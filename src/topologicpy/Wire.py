# Copyright (C) 2024
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

from binascii import a2b_base64
from re import A
import topologic_core as topologic
from topologicpy.Topology import Topology
import math
import itertools

class Wire():
    @staticmethod
    def Arc(startVertex, middleVertex, endVertex, sides: int = 16, close: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an arc. The base chord will be parallel to the x-axis and the height will point in the positive y-axis direction. 

        Parameters
        ----------
        startVertex : topologic_core.Vertex
            The start vertex of the arc.
        middleVertex : topologic_core.Vertex
            The middle vertex (apex) of the arc.
        endVertex : topologic_core.Vertex
            The end vertex of the arc.
        sides : int , optional
            The number of sides of the arc. The default is 16.
        close : bool , optional
            If set to True, the arc will be closed by connecting the last vertex to the first vertex. Otherwise, it will be left open.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created arc.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import numpy as np

        def circle_arc_points(p1, p2, p3, n):
            # Convert points to numpy arrays
            p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p1

            # Find the normal to the plane containing the three points
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Calculate midpoints of p1-p2 and p1-p3
            midpoint1 = (p1 + p2) / 2
            midpoint2 = (p1 + p3) / 2

            # Find the circumcenter using the perpendicular bisectors
            def perpendicular_bisector(pA, pB, midpoint):
                direction = np.cross(normal, pB - pA)
                direction = direction / np.linalg.norm(direction)
                return direction, midpoint

            direction1, midpoint1 = perpendicular_bisector(p1, p2, midpoint1)
            direction2, midpoint2 = perpendicular_bisector(p1, p3, midpoint2)

            # Solve for circumcenter
            A = np.array([direction1, -direction2]).T
            b = midpoint2 - midpoint1
            t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]
            
            circumcenter = midpoint1 + t1 * direction1

            # Calculate radius
            radius = np.linalg.norm(circumcenter - p1)

            # Helper function to rotate a point around an arbitrary axis
            def rotation_matrix_around_axis(axis, theta):
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                x, y, z = axis
                return np.array([
                    [cos_theta + x*x*(1 - cos_theta), x*y*(1 - cos_theta) - z*sin_theta, x*z*(1 - cos_theta) + y*sin_theta],
                    [y*x*(1 - cos_theta) + z*sin_theta, cos_theta + y*y*(1 - cos_theta), y*z*(1 - cos_theta) - x*sin_theta],
                    [z*x*(1 - cos_theta) - y*sin_theta, z*y*(1 - cos_theta) + x*sin_theta, cos_theta + z*z*(1 - cos_theta)]
                ])

            # Generate points along the arc
            def interpolate_on_arc(p_start, p_end, center, n_points):
                v_start = p_start - center
                v_end = p_end - center
                
                angle_between = np.arccos(np.dot(v_start, v_end) / (np.linalg.norm(v_start) * np.linalg.norm(v_end)))
                axis = np.cross(v_start, v_end)
                axis = axis / np.linalg.norm(axis)
                
                # Adjust for symmetry if n_points is even or odd
                if n_points % 2 == 0:
                    # For even n_points, generate n_points + 1 and skip the first point for symmetry
                    angles = np.linspace(0, angle_between, n_points + 1)
                    arc_points = [center + np.dot(rotation_matrix_around_axis(axis, angle), v_start) for angle in angles]
                    return [p_start]+arc_points[1:]  # Skip the first point
                else:
                    # For odd n_points, include both start, apex, and end points symmetrically
                    angles = np.linspace(0, angle_between, n_points)
                    arc_points = [center + np.dot(rotation_matrix_around_axis(axis, angle), v_start) for angle in angles]
                    return arc_points

            # Get points on the arc from p1 to p3 via p2
            if n <= 1: # Special case for number of edges == 1 or less.
                return [p1, p3]
            if n == 2: # Special case for number of edges == 2.
                return [p1, p2, p3]
            arc1 = interpolate_on_arc(p1, p2, circumcenter, (n+1) // 2)
            arc2 = interpolate_on_arc(p2, p3, circumcenter, (n+1) // 2)
            return np.vstack([arc1, arc2])
        
        if not Topology.IsInstance(startVertex, "Vertex"):
            if not silent:
                print("Wire.Arc - Error: The input startVertex is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(middleVertex, "Vertex"):
            if not silent:
                print("Wire.Arc - Error: The input middleVertex is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(endVertex, "Vertex"):
            if not silent:
                print("Wire.Arc - Error: The input endVertex is not a valid vertex. Returning None.")
            return None
        arc_points = circle_arc_points(np.array(Vertex.Coordinates(startVertex)), np.array(Vertex.Coordinates(middleVertex)), np.array(Vertex.Coordinates(endVertex)), sides)
        vertices = []
        for arc_point in arc_points:
            vertices.append(Vertex.ByCoordinates(list(arc_point)))
        arc = Wire.ByVertices(vertices, close=False, tolerance=tolerance)
        if not Topology.IsInstance(arc, "Wire"):
            if not silent:
                print("Wire.Arc - Error: Could not create an arc. Returning None.")
            return None
        return arc
    
    def ArcByEdge(edge, sagitta: float = 1, absolute: bool = True, sides: int = 16, close: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an arc. The base chord will be parallel to the x-axis and the height will point in the positive y-axis direction. 

        Parameters
        ----------
        edge : topologic_core.Edge
            The location of the start vertex of the arc.
        sagitta : float , optional
            The length of the sagitta. In mathematics, the sagitta is the line connecting the center of a chord to the apex (or highest point) of the arc subtended by that chord. The default is 1.
        absolute : bool , optional
            If set to True, the sagitta length is treated as an absolute value. Otherwise, it is treated as a ratio based on the length of the edge.
            For example, if the length of the edge is 10, the sagitta is set to 0.5, and absolute is set to False, the sagitta length will be 5. The default is True.
        sides : int , optional
            The number of sides of the arc. The default is 16.
        close : bool , optional
            If set to True, the arc will be closed by connecting the last vertex to the first vertex. Otherwise, it will be left open.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created arc.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Wire.ArcByEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        if sagitta <= 0:
            if not silent:
                print("Wire.ArcByEdge - Error: The input sagitta parameter is not a valid positive number. Returning None.")
            return None
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)
        if absolute == True:
            length = sagitta
        else:
            length = Edge.Length(edge)*sagitta
        norm = Edge.NormalEdge(edge, length=length, silent=silent)
        if norm == None:
            if not silent:
                print("Wire.ArcByEdge - Warning: Could not create an arc. Returning the original edge.")
            return edge
        cv = Edge.EndVertex(norm)
        return Wire.Arc(sv, cv, ev, sides=sides, close=close)
    
    @staticmethod
    def BoundingRectangle(topology, optimize: int = 0, mantissa: int = 6, tolerance=0.0001):
        """
        Returns a wire representing a bounding rectangle of the input topology. The returned wire contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting wire will become axis-aligned.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area.
            The minimum optimization number of 0 will result in an axis-aligned bounding rectangle.
            A maximum optimization number of 10 will attempt to reduce the bounding rectangle's area by 50%. The default is 0.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Wire
            The bounding rectangle of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from random import sample
        import time

        def br(topology):
            vertices = Topology.Vertices(topology)
            x = []
            y = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
            x_min = min(x)
            y_min = min(y)
            maxX = max(x)
            maxY = max(y)
            return [x_min, y_min, maxX, maxY]

        if not Topology.IsInstance(topology, "Topology"):
            return None

        vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
        if Vertex.AreCollinear(vertices, mantissa=mantissa, tolerance=tolerance):
            print("Wire.BoundingRectangle - Error: All vertices of the input topology parameter are collinear and thus no bounding rectangle can be created. Returning None.")
            return None
        start = time.time()
        period = 0
        result = True
        while result and period < 30:
            vList = sample(vertices, 3)
            result = Vertex.AreCollinear(vList)
            end = time.time()
            period = end - start
        if result == True:
            print("Wire.BoundingRectangle - Error: Could not find three vertices that are not colinear within 30 seconds. Returning None.")
            return None
        w = Wire.ByVertices(vList, close=True, tolerance=tolerance)
        if not Topology.IsInstance(w, "Wire"):
            print("Wire.BoundingRectangle - Error: Could not create wire from three vertices. Returning None.")
            return None
        f = Face.ByWire(w, tolerance=tolerance)
        if not Topology.IsInstance(f, "Face"):
            print("Wire.BoundingRectangle - Error: Could not create face from wire. Returning None.")
            return None
        f_origin = Topology.Centroid(f)
        normal = Face.Normal(f, mantissa=mantissa)
        topology = Topology.Flatten(topology, origin=f_origin, direction=normal)
        
        boundingRectangle = br(topology)
        x_min = boundingRectangle[0]
        y_min = boundingRectangle[1]
        maxX = boundingRectangle[2]
        maxY = boundingRectangle[3]
        w = abs(maxX - x_min)
        l = abs(maxY - y_min)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_br = boundingRectangle
        origin = Topology.Centroid(topology)
        optimize = min(max(optimize, 0), 10)
        if optimize > 0:
            factor = 1.0 - float(optimize)*0.05 # This will give a range of 0 to 0.5. Equivalent to a maximum 50% reduction in area.
            flag = False
            for n in range(10, 0, -1):
                if flag:
                    break
                za = n
                zb = 90+n
                zc = n
                for z in range(za,zb,zc):
                    if flag:
                        break
                    t = Topology.Rotate(topology, origin=origin, axis=[0, 0, 1], angle=z)
                    x_min, y_min, maxX, maxY = br(t)
                    w = abs(maxX - x_min)
                    l = abs(maxY - y_min)
                    area = l*w
                    if area <= orig_area*factor: # If new area is less than or equal to a certain percentage of the original area then break. e.g. if area is less than or qual to 50% of original area then break.
                        best_area = area
                        best_z = z
                        best_br = [x_min, y_min, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_br = [x_min, y_min, maxX, maxY]
                        
        else:
            best_br = boundingRectangle
        x_min, y_min, maxX, maxY = best_br
        vb1 = Vertex.ByCoordinates(x_min, y_min, 0)
        vb2 = Vertex.ByCoordinates(maxX, y_min, 0)
        vb3 = Vertex.ByCoordinates(maxX, maxY, 0)
        vb4 = Vertex.ByCoordinates(x_min, maxY, 0)

        boundingRectangle = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=origin, axis=[0, 0, 1], angle=-best_z)
        boundingRectangle = Topology.Unflatten(boundingRectangle, origin=f_origin, direction=normal)
        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        boundingRectangle = Topology.SetDictionary(boundingRectangle, dictionary)
        return boundingRectangle

    @staticmethod
    def ByEdges(edges: list, orient: bool = False, tolerance: float = 0.0001):
        """
        Creates a wire from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        orient : bool , optional
            If set to True the edges are oriented head to tail. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if Topology.IsInstance(x, "Edge")]
        if len(edgeList) == 0:
            print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
            return None
        if len(edgeList) == 1:
            wire = topologic.Wire.ByEdges(edgeList) # Hook to Core
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(edgeList), tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.ByEdges - Error: The operation failed. Returning None.")
            wire = None
        if Wire.IsManifold(wire):
            if orient == True:
                wire = Wire.OrientEdges(wire, Wire.StartVertex(wire), tolerance=tolerance)
        return wire

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        if not Topology.IsInstance(cluster, "Cluster"):
            print("Wire.ByEdges - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def ByOffset(wire, offset: float = 1.0, offsetKey: str = "offset", stepOffsetA: float = 0, stepOffsetB: float = 0, stepOffsetKeyA: str = "stepOffsetA", stepOffsetKeyB: str = "stepOffsetB", reverse: bool = False, bisectors: bool = False, transferDictionaries: bool = False, epsilon: float = 0.01, tolerance: float = 0.0001,  silent: bool = False, numWorkers: int = None):
        """
        Creates an offset wire from the input wire. A positive offset value results in an offset to the interior of an anti-clockwise wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. The default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. The default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. The default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. The default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. The default is "stepOffsetB".
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. The default is False.
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned wire. The default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the new wire. Otherwise, they are not. The default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). The default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        numWorkers : int , optional
            Number of workers run in parallel to process. If you set it to 1, no parallel processing will take place.
            The default is None which causes the algorithm to use twice the number of cpu cores in the host computer.

        
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ByOffset - Error: The input wire parameter is not a valid wire. Returning None.")
                return None
        
        #temp_face = Face.ByWire(wire)
        #original_area = Face.Area(temp_face)
        if reverse == True:
            fac = -1
        else:
            fac = 1
        origin = Topology.Centroid(wire)
        temp_vertices = [Topology.Vertices(wire)[0], Topology.Vertices(wire)[1], Topology.Centroid(wire)]
        temp_face = Face.ByWire(Wire.ByVertices(temp_vertices, close=True, tolerance=tolerance), silent=silent)
        temp_normal = Face.Normal(temp_face)
        flat_wire = Topology.Flatten(wire, direction=temp_normal, origin=origin)
        normal = Face.Normal(temp_face)
        flat_wire = Topology.Flatten(wire, direction=normal, origin=origin)
        original_edges = Topology.Edges(wire)
        edges = Topology.Edges(flat_wire)
        original_edges = Topology.Edges(wire)
        offsets = []
        offset_edges = []
        final_vertices = []
        bisectors_list = []
        edge_dictionaries = []
        for i, edge in enumerate(edges):
            d = Topology.Dictionary(original_edges[i])
            d_offset = Dictionary.ValueAtKey(d, offsetKey)
            if d_offset == None:
                d_offset = offset
            d_offset = d_offset*fac
            offsets.append(d_offset)
            offset_edge = Edge.ByOffset2D(edge, d_offset)
            offset_edges.append(offset_edge)
        for i in range(len(edges)):
            o_edge_a = offset_edges[i]
            v_a = Edge.StartVertex(edges[i])
            if i == 0:
                if Wire.IsClosed(wire) == False:
                    v1 = Edge.StartVertex(offset_edges[0])
                    if transferDictionaries == True:
                        v1 = Topology.SetDictionary(v1, Topology.Dictionary(v_a), silent=True)
                        edge_dictionaries.append(Topology.Dictionary(edges[i]))
                    final_vertices.append(v1)
                    if bisectors == True:
                        bisectors_list.append(Edge.ByVertices(v_a, v1))
                else:
                    prev_edge = offset_edges[-1]
                    v1 = Edge.Intersect2D(prev_edge, o_edge_a, silent=True)
                    if Topology.IsInstance(v1, "Vertex"):
                        if bisectors == True:
                            bisectors_list.append(Edge.ByVertices(v_a, v1))
                        if transferDictionaries == True:
                            v1 = Topology.SetDictionary(v1, Topology.Dictionary(v_a), silent=True)
                            edge_dictionaries.append(Topology.Dictionary(edges[i]))
                        final_vertices.append(v1)
                    else:
                        connection = Edge.Connection(prev_edge, o_edge_a)
                        if Topology.IsInstance(connection, "Edge"):
                            d = Topology.Dictionary(v_a)
                            d_stepOffsetA = Dictionary.ValueAtKey(d, stepOffsetKeyA)
                            if d_stepOffsetA == None:
                                d_stepOffsetA = stepOffsetA
                            d_stepOffsetB = Dictionary.ValueAtKey(d, stepOffsetKeyB)
                            if d_stepOffsetB == None:
                                d_stepOffsetB = stepOffsetB
                            v1_1 = Topology.TranslateByDirectionDistance(Edge.EndVertex(prev_edge),
                                                                        direction = Vector.Reverse(Edge.Direction(prev_edge)),
                                                                        distance = d_stepOffsetA)
                                                                                                    
                            v1_2 = Topology.TranslateByDirectionDistance(Edge.StartVertex(o_edge_a),
                                                                        direction = Edge.Direction(o_edge_a),
                                                                        distance = d_stepOffsetB)
                            bisectors_list.append(Edge.ByVertices(v_a, v1_1))
                            bisectors_list.append(Edge.ByVertices(v_a, v1_2))
                            final_vertices.append(v1_1)
                            final_vertices.append(v1_2)
                            if transferDictionaries == True:
                                v1_1 = Topology.SetDictionary(v1_1, Topology.Dictionary(v_a), silent=True)
                                v1_2 = Topology.SetDictionary(v1_2, Topology.Dictionary(v_a), silent=True)
                                edge_dictionaries.append(Topology.Dictionary(v_a))
                                edge_dictionaries.append(Topology.Dictionary(edges[i]))
            else:
                prev_edge = offset_edges[i-1]
                v1 = Edge.Intersect2D(prev_edge, o_edge_a, silent=True)
                if Topology.IsInstance(v1, "Vertex"):
                    if bisectors == True:
                        bisectors_list.append(Edge.ByVertices(v_a, v1))
                    if transferDictionaries == True:
                        d_temp = Topology.Dictionary(v_a)
                        v1 = Topology.SetDictionary(v1, Topology.Dictionary(v_a), silent=True)
                        edge_dictionaries.append(Topology.Dictionary(edges[i]))
                    final_vertices.append(v1)
                else:
                    connection = Edge.Connection(prev_edge, o_edge_a)
                    if Topology.IsInstance(connection, "Edge"):
                        d = Topology.Dictionary(v_a)
                        d_stepOffsetA = Dictionary.ValueAtKey(d, stepOffsetKeyA)
                        if d_stepOffsetA == None:
                            d_stepOffsetA = stepOffsetA
                        d_stepOffsetB = Dictionary.ValueAtKey(d, stepOffsetKeyB)
                        if d_stepOffsetB == None:
                            d_stepOffsetB = stepOffsetB
                        v1_1 = Topology.TranslateByDirectionDistance(Edge.EndVertex(prev_edge),
                                                                     direction = Vector.Reverse(Edge.Direction(prev_edge)),
                                                                     distance = d_stepOffsetA)
                                                                                                
                        v1_2 = Topology.TranslateByDirectionDistance(Edge.StartVertex(o_edge_a),
                                                                     direction = Edge.Direction(o_edge_a),
                                                                     distance = d_stepOffsetB)
                        if transferDictionaries == True:
                            v1_1 = Topology.SetDictionary(v1_1, Topology.Dictionary(v_a), silent=True)
                            v1_2 = Topology.SetDictionary(v1_2, Topology.Dictionary(v_a), silent=True)
                            edge_dictionaries.append(Topology.Dictionary(v_a))
                            edge_dictionaries.append(Topology.Dictionary(edges[i]))
                        bisectors_list.append(Edge.ByVertices(v_a, v1_1))
                        bisectors_list.append(Edge.ByVertices(v_a, v1_2))
                        final_vertices.append(v1_1)
                        final_vertices.append(v1_2)
        v_a = Edge.EndVertex(edges[-1])
        if Wire.IsClosed(wire) == False:
            v1 = Edge.EndVertex(offset_edges[-1])
            final_vertices.append(v1)
            if transferDictionaries == True:
                v1 = Topology.SetDictionary(v1, Topology.Dictionary(v_a), silent=True)
            if bisectors == True:
                bisectors_list.append(Edge.ByVertices(v_a, v1))
        
        
        # wire_edges = []
        # for i in range(len(final_vertices)-1):
        #     v1 = final_vertices[i]
        #     v2 = final_vertices[i+1]
        #     w_e = Edge.ByVertices(v1,v2)
        #     #w_e = Edge.SetLength(w_e, Edge.Length(w_e)+(2*epsilon), bothSides = True)
        #     wire_edges.append(w_e)
        # if Wire.IsClosed(wire):
        #     v1 = final_vertices[-1]
        #     v2 = final_vertices[0]
        #     #w_e = Edge.SetLength(w_e, Edge.Length(w_e)+(2*epsilon), bothSides = True)
        #     wire_edges.append(w_e)
        
        return_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        #wire_edges = Topology.Edges(wire_edges)
        wire_edges = [Edge.SetLength(w_e, Edge.Length(w_e)+(2*epsilon), bothSides=True) for w_e in Topology.Edges(return_wire)]
        return_wire_edges = Topology.Edges(return_wire)
        if transferDictionaries == True:
            if not len(wire_edges) == len(edge_dictionaries):
                if not silent:
                        print("Length of Wire Edges:", len(wire_edges))
                        print("Length of Edge Dictionaries:", len(edge_dictionaries))
                        print("Wire.ByOffset - Warning: The resulting wire is not well-formed, offsets may not be applied correctly. Please check your offsets.")
            for i, wire_edge in enumerate(wire_edges):
                if len(edge_dictionaries) > 0:
                    temp_dictionary = edge_dictionaries[min(i,len(edge_dictionaries)-1)]
                    wire_edge = Topology.SetDictionary(wire_edge, temp_dictionary, silent=True)
                    return_wire_edges[i] = Topology.SetDictionary(return_wire_edges[i], temp_dictionary, silent=True)
        if bisectors == True:
            temp_return_wire = Topology.SelfMerge(Cluster.ByTopologies(wire_edges+bisectors_list))
            if transferDictionaries == True:
                sel_vertices = Topology.Vertices(return_wire)
                sel_vertices += Topology.Vertices(flat_wire)
                edges = Topology.Edges(return_wire)
                sel_edges = []
                for edge in edges:
                    d = Topology.Dictionary(edge)
                    c = Topology.Centroid(edge)
                    c = Topology.SetDictionary(c, d, silent=True)
                    sel_edges.append(c)
                temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_vertices, tranVertices=True, numWorkers=numWorkers)
                temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_edges, tranEdges=True, numWorkers=numWorkers)
                
            return_wire = temp_return_wire
        
        if not Topology.IsInstance(return_wire, "Wire"):
            if not silent:
                print("Wire.ByOffset - Warning: The resulting wire is not well-formed, please check your offsets.")
        else:
            if not Wire.IsManifold(return_wire) and bisectors == False:
                if not silent:
                    print("Wire.ByOffset - Warning: The resulting wire is non-manifold, please check your offsets.")
                    print("Wire.ByOffset - Warning: Pursuing a workaround, but it might take longer to complete.")
                
                #cycles = Wire.Cycles(return_wire, maxVertices = len(final_vertices))
                temp_wire = Topology.SelfMerge(Cluster.ByTopologies(wire_edges))
                cycles = Wire.Cycles(temp_wire, maxVertices = len(final_vertices))
                if len(cycles) > 0:
                    distances = []
                    for cycle in cycles:
                        cycle_centroid = Topology.Centroid(cycle)
                        distance = Vertex.Distance(origin, cycle_centroid)
                        distances.append(distance)
                    cycles = Helper.Sort(cycles, distances)
                    # Get the top three or less
                    cycles = cycles[:min(3, len(cycles))]
                    areas = [Face.Area(Face.ByWire(cycle)) for cycle in cycles]
                    cycles = Helper.Sort(cycles, areas)
                    return_cycle = Wire.Reverse(cycles[-1])
                    test_cycle = Wire.Simplify(return_cycle, tolerance=epsilon)
                    if Topology.IsInstance(test_cycle, "Wire"):
                        return_cycle = test_cycle
                    return_cycle = Wire.RemoveCollinearEdges(return_cycle, silent=silent)
                    sel_edges = []
                    for temp_edge in wire_edges:
                        x = Topology.Centroid(temp_edge)
                        d = Topology.Dictionary(temp_edge)
                        x = Topology.SetDictionary(x, d, silent=True)
                        sel_edges.append(x)
                    return_cycle = Topology.TransferDictionariesBySelectors(return_cycle, Topology.Vertices(return_wire), tranVertices=True, tolerance=tolerance, numWorkers=numWorkers)
                    return_cycle = Topology.TransferDictionariesBySelectors(return_cycle, sel_edges, tranEdges=True, tolerance=tolerance, numWorkers=numWorkers)
                    return_wire = return_cycle

        return_wire = Topology.Unflatten(return_wire, direction=normal, origin=origin)
        if transferDictionaries == True:
            return_wire = Topology.SetDictionary(return_wire, Topology.Dictionary(wire), silent=True)
        return return_wire

    @staticmethod
    def ByOffsetArea(wire,
                    area,
                    offsetKey="offset",
                    minOffsetKey="minOffset",
                    maxOffsetKey="maxOffset",
                    defaultMinOffset=0,
                    defaultMaxOffset=1,
                    maxIterations = 1,
                    tolerance=0.0001,
                    silent = False,
                    numWorkers = None):
        """
        Creates an offset wire from the input wire based on the input area.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        area : float
            The desired area of the created wire.
        offsetKey : str , optional
            The edge dictionary key under which to store the offset value. The default is "offset".
        minOffsetKey : str , optional
            The edge dictionary key under which to find the desired minimum edge offset value. If a value cannot be found, the defaultMinOffset input parameter value is used instead. The default is "minOffset".
        maxOffsetKey : str , optional
            The edge dictionary key under which to find the desired maximum edge offset value. If a value cannot be found, the defaultMaxOffset input parameter value is used instead. The default is "maxOffset".
        defaultMinOffset : float , optional
            The desired minimum edge offset distance. The default is 0.
        defaultMaxOffset : float , optional
            The desired maximum edge offset distance. The default is 1.
        maxIterations: int , optional
            The desired maximum number of iterations to attempt to converge on a solution. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        numWorkers : int , optional
            Number of workers run in parallel to process. If you set it to 1, no parallel processing will take place.
            The default is None which causes the algorithm to use twice the number of cpu cores in the host computer.
        
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import numpy as np
        from scipy.optimize import minimize

        def compute_offset_amounts(wire,
                                area,
                                offsetKey="offset",
                                minOffsetKey="minOffset",
                                maxOffsetKey="maxOffset",
                                defaultMinOffset=0,
                                defaultMaxOffset=1,
                                maxIterations = 10000,
                                tolerance=0.0001):
            
            initial_offsets = []
            bounds = []
            for edge in edges:
                d = Topology.Dictionary(edge)
                minOffset = Dictionary.ValueAtKey(d, minOffsetKey) or defaultMinOffset
                maxOffset = Dictionary.ValueAtKey(d, maxOffsetKey) or defaultMaxOffset
                # Initial guess: small negative offsets to shrink the polygon, within the constraints
                initial_offsets.append((minOffset + maxOffset) / 2)
                # Bounds based on the constraints for each edge
                bounds.append((minOffset, maxOffset))

            # Convert initial_offsets to np.array for efficiency
            initial_offsets = np.array(initial_offsets)
            iteration_count = [0]  # List to act as a mutable counter

            def objective_function(offsets):
                for i, edge in enumerate(edges):
                    d = Topology.Dictionary(edge)
                    d = Dictionary.SetValueAtKey(d, offsetKey, offsets[i])
                    edge = Topology.SetDictionary(edge, d)
                
                # Offset the wire
                new_wire = Wire.ByOffset(wire, offsetKey=offsetKey, silent=silent, numWorkers=numWorkers)
                # Check for an illegal wire. In that case, return a very large loss value.
                if not Topology.IsInstance(new_wire, "Wire"):
                    return (float("inf"))
                if not Wire.IsManifold(new_wire):
                    return (float("inf"))
                if not Wire.IsClosed(new_wire):
                    return (float("inf"))
                new_face = Face.ByWire(new_wire)
                # Calculate the area of the new wire/face
                new_area = Face.Area(new_face)
                
                # The objective is the difference between the target hole area and the actual hole area
                # We want this difference to be as close to 0 as possible
                loss = (new_area - area) ** 2
                # If the loss is less than the tolerance, accept the result and return a loss of 0.
                if loss <= tolerance:
                    return 0
                # Otherwise, return the actual loss value.
                return loss 
            
            # Callback function to track and display iteration number
            def iteration_callback(xk):
                iteration_count[0] += 1  # Increment the counter
                if not silent:
                    print(f"Wire.ByOffsetArea - Information: Iteration {iteration_count[0]}")
            
            # Use scipy optimization/minimize to find the correct offsets, respecting the min/max bounds
            result = minimize(objective_function,
                            initial_offsets,
                            method = "Powell",
                            bounds=bounds,
                            options={ 'maxiter': maxIterations},
                            callback=iteration_callback
                            )

            # Return the offsets
            return result.x
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.OffsetByArea - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.OffsetByArea - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        
        if not Wire.IsClosed(wire):
            if not silent:
                print("Wire.OffsetByArea - Error: The input wire parameter is not a closed wire. Returning None.")
            return None
        
        edges = Topology.Edges(wire)
        # Compute the offset amounts
        offsets = compute_offset_amounts(wire,
                                area = area,
                                offsetKey = offsetKey,
                                minOffsetKey = minOffsetKey,
                                maxOffsetKey = maxOffsetKey,
                                defaultMinOffset = defaultMinOffset,
                                defaultMaxOffset = defaultMaxOffset,
                                maxIterations = maxIterations,
                                tolerance = tolerance)
        # Set the edge dictionaries correctly according to the specified offsetKey
        for i, edge in enumerate(edges):
            d = Topology.Dictionary(edge)
            d = Dictionary.SetValueAtKey(d, offsetKey, offsets[i])
            edge = Topology.SetDictionary(edge, d)
                
        # Offset the wire
        return_wire = Wire.ByOffset(wire, offsetKey=offsetKey, silent=silent, numWorkers=numWorkers)
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.OffsetByArea - Error: Could not create the offset wire. Returning None.")
            return None
        return return_wire

    @staticmethod
    def ByVertices(vertices: list, close: bool = True, tolerance: float = 0.0001):
        """
        Creates a wire from the input list of vertices.

        Parameters
        ----------
        vertices : list
            the input list of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            print("Wire.ByVertices - Error: The number of vertices is less than 2. Returning None.")
            return None
        edges = []
        for i in range(len(vertexList)-1):
            v1 = vertexList[i]
            v2 = vertexList[i+1]
            e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
        if close:
            v1 = vertexList[-1]
            v2 = vertexList[0]
            e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
        if len(edges) < 1:
            print("Wire.ByVertices - Error: The number of edges is less than 1. Returning None.")
            return None
        elif len(edges) == 1:
            wire = Wire.ByEdges(edges, orient=False)
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
        return wire

    @staticmethod
    def ByVerticesCluster(cluster, close: bool = True, tolerance: float = 0.0001):
        """
        Creates a wire from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.cluster
            the input cluster of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        vertices = Topology.Vertices(cluster)
        return Wire.ByVertices(vertices, close=close, tolerance=tolerance)

    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 0.5.
        sides : int , optional
            The desired number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created circle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Circle - Error: The input origin parameter is not a valid Vertex. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Circle - Error: The input placement parameter is not a recognized string. Returning None.")
            return None
        radius = abs(radius)
        if radius <= tolerance:
            return None
        
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) <= tolerance:
            return None
        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*radius + Vertex.X(origin)
            y = math.cos(angle)*radius + Vertex.Y(origin)
            z = Vertex.Z(origin)
            xList.append(x)
            yList.append(y)
            baseV.append(Vertex.ByCoordinates(x, y, z))

        if angleRange == 360:
            baseWire = Wire.ByVertices(baseV[::-1], close=False, tolerance=tolerance) #reversing the list so that the normal points up in Blender
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close, tolerance=tolerance) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, radius, radius, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -radius, radius, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire
    
    @staticmethod
    def Close(wire, mantissa=6, tolerance=0.0001):
        """
        Closes the input wire

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The closed version of the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        
        def nearest_vertex(vertex, vertices):
            distances = []
            for v in vertices:
                distances.append(Vertex.Distance(vertex, v))
            new_vertices = Helper.Sort(vertices, distances)
            return new_vertices[1] #The first item is the same vertex, so return the next nearest vertex.
        
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Close - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if Wire.IsClosed(wire):
            return wire
        vertices = Topology.Vertices(wire)
        ends = [v for v in vertices if Vertex.Degree(v, wire) == 1]
        if len(ends) < 2:
            print("Wire.Close - Error: The input wire parameter contains less than two open end vertices. Returning None.")
            return None
        geometry = Topology.Geometry(wire, mantissa=mantissa)
        g_vertices = geometry['vertices']
        g_edges = geometry['edges']
        used = []
        for end in ends:
            nearest = nearest_vertex(end, ends)
            if not nearest in used:
                d = Vertex.Distance(end, nearest)
                i1 = Vertex.Index(end, vertices, tolerance=tolerance)
                i2 = Vertex.Index(nearest, vertices, tolerance=tolerance)
                if i1 == None or i2 == None:
                    print("Wire.Close - Error: Something went wrong. Returning None.")
                    return None
                if d <= tolerance:
                    g_vertices[i1] = Vertex.Coordinates(end)
                    g_vertices[i2] = Vertex.Coordinates(end)
                else:
                    if not(([i1, i2] in g_edges) or ([i2, i1] in g_edges)):
                        g_edges.append([i1, i2])
                used.append(end)
        new_wire = Topology.SelfMerge(Topology.ByGeometry(vertices=g_vertices, edges=g_edges, faces=[]))
        return new_wire



    @staticmethod
    def ConcaveHull(topology, k: int = 3, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a wire representing the 2D concave hull of the input topology. The vertices of the topology are assumed to be coplanar.
        Code based on Moreira, A and Santos, M Y, "CONCAVE HULL: A K-NEAREST NEIGHBOURS APPROACH FOR THE COMPUTATION OF THE REGION OCCUPIED BY A SET OF POINTS"
        GRAPP 2007 - International Conference on Computer Graphics Theory and Applications.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        k : int, optional
            The number of nearest neighbors to consider for each point when building the hull. 
            Must be at least 3 for the algorithm to function correctly. Increasing `k` will produce a smoother, 
            less concave hull, while decreasing `k` may yield a more detailed, concave shape. The default is 3.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The concave hull of the input topology.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from math import atan2, sqrt, pi
        from random import sample

        # Helper function to clean the list by removing duplicate points
        def clean_list(points_list):
            return list(set(points_list))

        # Helper function to find the point with the minimum Y-coordinate
        def find_min_y_point(points):
            return min(points, key=lambda p: [p[1], p[0]])

        # Helper function to find the k-nearest neighbors to a given point
        def nearest_points(points, reference_point, k):
            # Sort points by distance from the reference point and select the first k points
            sorted_points = sorted(points, key=lambda p: sqrt((p[0] - reference_point[0]) ** 2 + (p[1] - reference_point[1]) ** 2))
            return sorted_points[:k]

        # Helper function to sort points by the angle relative to the previous direction
        def sort_by_angle(points, current_point, prev_angle):
            def angle_to(p):
                angle = atan2(p[1] - current_point[1], p[0] - current_point[0])
                angle_diff = (angle - prev_angle + 2 * pi) % (2 * pi)
                return angle_diff
            return sorted(points, key=angle_to)

        # Helper function to check if two line segments intersect
        def intersects_q(line1, line2):
            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0: return 0
                return 1 if val > 0 else 2

            p1, q1 = line1
            p2, q2 = line2
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True
            return False

        # Helper function to check if point q lies on segment pr
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        # Helper function to calculate the angle between two points
        def angle(p1, p2):
            return atan2(p2[1] - p1[1], p2[0] - p1[0])

        # Helper function to determine if a point is inside a polygon (Ray Casting method)
        def point_in_polygon_q(point, polygon):
            x, y = point
            inside = False
            n = len(polygon)
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        def concave_hull(points_list, k: int = 3):
            # Ensure k >= 3
            kk = max(k, 3)
            
            # Remove duplicate points
            dataset = clean_list(points_list)
            
            # If there are fewer than 3 unique points, no polygon can be formed
            if len(dataset) < 3:
                return None
            elif len(dataset) == 3:
                return dataset  # If exactly 3 points, they form the polygon

            # Ensure we have enough neighbors
            kk = min(kk, len(dataset) - 1)
            
            # Find starting point (minimum Y value) and initialize hull
            first_point = find_min_y_point(dataset)
            hull = [first_point]
            current_point = first_point
            dataset.remove(first_point)
            prev_angle = 0
            step = 2
            
            # Original code logic, with an update to calculate prev_angle
            while (current_point != first_point or step == 2) and len(dataset) > 0:
                # After 4 steps, re-add the starting point to check for closure
                if step == 5:
                    dataset.append(first_point)
                
                # Find the k-nearest points
                k_nearest_points = nearest_points(dataset, current_point, kk)
                
                # Sort candidates based on angle
                c_points = sort_by_angle(k_nearest_points, current_point, prev_angle)
                
                intersection_found = True
                i = 0
                
                # Select the first candidate that does not intersect any polygon edges
                while intersection_found and i < len(c_points):
                    candidate_point = c_points[i]
                    i += 1
                    
                    if candidate_point == first_point:
                        last_point_check = 1
                    else:
                        last_point_check = 0

                    # Check for intersections with the existing edges
                    j = 2
                    intersection_found = False
                    while not intersection_found and j < len(hull) - last_point_check:
                        # Using hull[-1] and hull[-2] for last and second-to-last points
                        intersection_found = intersects_q(
                            (hull[-1], candidate_point),
                            (hull[-1 - j], hull[-j])
                        )
                        j += 1

                # If all candidates intersect, retry with a higher number of neighbors
                if intersection_found:
                    return concave_hull(points_list, kk + 1)
                
                # Update the hull with the selected candidate point
                current_point = candidate_point
                hull.append(current_point)

                # Calculate the angle between the last two points in the hull to set `prev_angle`
                if len(hull) > 1:
                    prev_angle = angle(hull[-1], hull[-2])
                    
                dataset.remove(current_point)
                step += 1


            # Check if all points are inside the constructed hull
            all_inside = True
            i = len(dataset) - 1
            while all_inside and i >= 0:
                all_inside = point_in_polygon_q(dataset[i], hull)
                i -= 1

            # If any points are outside the hull, retry with a higher number of neighbors
            if not all_inside:
                return concave_hull(points_list, kk + 1)
            
            # Return the completed hull if all points are inside
            return hull

        f = None
        # Create a sample face and flatten
        while not Topology.IsInstance(f, "Face"):
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v, tolerance=tolerance)
            f = Face.ByWire(w, tolerance=tolerance, silent=True)
            if not f == None:
                origin = Topology.Centroid(f)
                normal = Face.Normal(f, mantissa=mantissa)
                f = Topology.Flatten(f, origin=origin, direction=normal)
        flat_topology = Topology.Flatten(topology, origin=origin, direction=normal)
        vertices = Topology.Vertices(flat_topology)
        points = []
        for v in vertices:
            points.append((Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa)))
        hull = concave_hull(points, k=k)
        hull_vertices = []
        for p in hull:
            hull_vertices.append(Vertex.ByCoordinates(p[0], p[1], 0))
        ch = Wire.ByVertices(hull_vertices, close=True, tolerance=tolerance)
        ch = Topology.Unflatten(ch, origin=origin, direction=normal)
        return ch

    @staticmethod
    def ConvexHull(topology, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a wire representing the 2D convex hull of the input topology. The vertices of the topology are assumed to be coplanar.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The convex hull of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from random import sample

        def Left_index(points):
            
            '''
            Finding the left most point
            '''
            minn = 0
            for i in range(1,len(points)):
                if points[i][0] < points[minn][0]:
                    minn = i
                elif points[i][0] == points[minn][0]:
                    if points[i][1] > points[minn][1]:
                        minn = i
            return minn

        def orientation(p, q, r):
            '''
            To find orientation of ordered triplet (p, q, r). 
            The function returns following values 
            0 --> p, q and r are collinear 
            1 --> Clockwise 
            2 --> Counterclockwise 
            '''
            val = (q[1] - p[1]) * (r[0] - q[0]) - \
                (q[0] - p[0]) * (r[1] - q[1])
        
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2
        
        def convex_hull(points, n):
            
            # There must be at least 3 points 
            if n < 3:
                return
        
            # Find the leftmost point
            l = Left_index(points)
        
            hull = []
            
            '''
            Start from leftmost point, keep moving counterclockwise 
            until reach the start point again. This loop runs O(h) 
            times where h is number of points in result or output. 
            '''
            p = l
            q = 0
            while(True):
                
                # Add current point to result 
                hull.append(p)
        
                '''
                Search for a point 'q' such that orientation(p, q, 
                x) is counterclockwise for all points 'x'. The idea 
                is to keep track of last visited most counterclock- 
                wise point in q. If any point 'i' is more counterclock- 
                wise than q, then update q. 
                '''
                q = (p + 1) % n
        
                for i in range(n):
                    
                    # If i is more counterclockwise 
                    # than current q, then update q 
                    if(orientation(points[p], 
                                points[i], points[q]) == 2):
                        q = i
        
                '''
                Now q is the most counterclockwise with respect to p 
                Set p as q for next iteration, so that q is added to 
                result 'hull' 
                '''
                p = q
        
                # While we don't come to first point
                if(p == l):
                    break
        
            # Print Result 
            return hull

        f = None
        # Create a sample face and flatten
        while not Topology.IsInstance(f, "Face"):
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v, tolerance=tolerance)
            f = Face.ByWire(w, tolerance=tolerance)
            origin = Topology.Centroid(f)
            normal = Face.Normal(f, mantissa=mantissa)
            f = Topology.Flatten(f, origin=origin, direction=normal)
        flat_topology = Topology.Flatten(topology, origin=origin, direction=normal)
        vertices = Topology.Vertices(flat_topology)
        points = []
        for v in vertices:
            points.append((Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa)))
        hull = convex_hull(points, len(points))
        hull_vertices = []
        for p in hull:
            hull_vertices.append(Vertex.ByCoordinates(points[p][0], points[p][1], 0))
        ch = Wire.ByVertices(hull_vertices, tolerance=tolerance)
        ch = Topology.Unflatten(ch, origin=origin, direction=normal)
        return ch

    @staticmethod
    def CrossShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c=None,
            d=None,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a Cross-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. The default is None which results in the Cross-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the Cross-shape. The default is 1.0.
        length : float , optional
            The overall length of the Cross-shape. The default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the Cross-shape. The default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the Cross-shape. The default is 0.25.
        c : float , optional
            The distance of the vertical symmetry axis measured from the left side of the Cross-shape. The default is None which results in the Cross-shape being symmetrical on the Y-axis.
        d : float , optional
            The distance of the horizontal symmetry axis measured from the bottom side of the Cross-shape. The default is None which results in the Cross-shape being symmetrical on the X-axis.
        direction : list , optional
            The vector representing the up direction of the Cross-shape. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the Cross-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created Cross-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Wire.CrossShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Wire.CrossShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Wire.CrossShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Wire.CrossShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if c == None:
            c = width/2
        if d == None:
            d = length/2
        if not isinstance(c, int) and not isinstance(c, float):
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(d, int) and not isinstance(d, float):
            if not silent:
                print("Wire.CrossShape - Error: The d input parameter is not a valid number. Returning None.")
        if width <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if d <= tolerance:
            if not silent:
                print("Wire.CrossShape - Error: The d input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance*2):
            if not silent:
                print("Wire.CrossShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance*2):
            if not silent:
                print("Wire.CrossShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if c <= (tolerance + a/2):
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter must be more than half the a input parameter. Returning None.")
            return None
        if d <= (tolerance + b/2):
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter must be more than half the b input parameter. Returning None.")
            return None
        if c >= (width - tolerance - a/2):
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter must be less than the width minus half the a input parameter. Returning None.")
            return None
        if d >= (length - tolerance - b/2):
            if not silent:
                print("Wire.CrossShape - Error: The c input parameter must be less than the width minus half the b input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.CrossShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.CrossShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.CrossShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the Cross-shape (counterclockwise)
        v1 = Vertex.ByCoordinates(c-a/2, 0)
        v2 = Vertex.ByCoordinates(c+a/2, 0)
        v3 = Vertex.ByCoordinates(c+a/2, d-b/2)
        v4 = Vertex.ByCoordinates(width, d-b/2)
        v5 = Vertex.ByCoordinates(width, d+b/2)
        v6 = Vertex.ByCoordinates(c+a/2, d+b/2)
        v7 = Vertex.ByCoordinates(c+a/2, length)
        v8 = Vertex.ByCoordinates(c-a/2, length)  # Top of vertical arm
        v9 = Vertex.ByCoordinates(c-a/2, d+b/2)  # Top of vertical arm
        v10 = Vertex.ByCoordinates(0, d+b/2)  # Top of vertical arm
        v11 = Vertex.ByCoordinates(0, d-b/2)  # Top of vertical arm
        v12 = Vertex.ByCoordinates(c-a/2, d-b/2)  # Top of vertical arm

        # Create the T-shaped wire
        cross_shape = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8, v9,v10, v11, v12], close=True, tolerance=tolerance)
        cross_shape = Topology.Translate(cross_shape, -width/2, -length/2, 0)
        cross_shape = Topology.Translate(cross_shape, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        reverse = False
        if flipHorizontal == True:
            xScale = -1
            reverse = not reverse
        else:
            xScale = 1
        if flipVertical == True:
            yScale = -1
            reverse = not reverse
        else:
            yScale = 1
        if xScale == -1 or yScale == -1:
            cross_shape = Topology.Scale(cross_shape, origin=origin, x=xScale, y=yScale, z=1)
            if reverse == True:
                cross_shape = Wire.Reverse(cross_shape)
        if placement.lower() == "lowerleft":
            cross_shape = Topology.Translate(cross_shape, width/2, length/2, 0)
        elif placement.lower() == "upperright":
            cross_shape = Topology.Translate(cross_shape, -width/2, -length/2, 0)
        elif placement.lower() == "upperleft":
            cross_shape = Topology.Translate(cross_shape, width/2, -length/2, 0)
        elif placement.lower() == "lowerright":
            cross_shape = Topology.Translate(cross_shape, -width/2, length/2, 0)
        
        if direction != [0, 0, 1]:
            cross_shape = Topology.Orient(cross_shape, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return cross_shape
    
    @staticmethod
    def CShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c =0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a C-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the C-shape. The default is None which results in the C-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the C-shape. The default is 1.0.
        length : float , optional
            The overall length of the C-shape. The default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the C-shape. The default is 0.25.
        b : float , optional
            The vertical thickness of the lower horizontal arm of the C-shape. The default is 0.25.
        c : float , optional
            The vertical thickness of the upper horizontal arm of the C-shape. The default is 0.25.
        direction : list , optional
            The vector representing the up direction of the C-shape. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the C-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created C-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Wire.CShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Wire.CShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Wire.CShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Wire.CShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Wire.CShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Wire.CShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Wire.CShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Wire.CShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Wire.CShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Wire.CShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Wire.CShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.CShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.CShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.CShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the C-shape (counterclockwise)
        v1 = Vertex.Origin()  # Base origin
        v2 = Vertex.ByCoordinates(width, 0)
        v3 = Vertex.ByCoordinates(width, b)
        v4 = Vertex.ByCoordinates(a, b)
        v5 = Vertex.ByCoordinates(a, length-c)
        v6 = Vertex.ByCoordinates(width, length-c)
        v7 = Vertex.ByCoordinates(width, length)
        v8 = Vertex.ByCoordinates(0, length)

        # Create the C-shaped wire
        c_shape = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8], close=True, tolerance=tolerance)
        c_shape = Topology.Translate(c_shape, -width/2, -length/2, 0)
        c_shape = Topology.Translate(c_shape, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        reverse = False
        if flipHorizontal == True:
            xScale = -1
            reverse = not reverse
        else:
            xScale = 1
        if flipVertical == True:
            yScale = -1
            reverse = not reverse
        else:
            yScale = 1
        if xScale == -1 or yScale == -1:
            c_shape = Topology.Scale(c_shape, origin=origin, x=xScale, y=yScale, z=1)
            if reverse == True:
                c_shape = Wire.Reverse(c_shape)
        if placement.lower() == "lowerleft":
            c_shape = Topology.Translate(c_shape, width/2, length/2, 0)
        elif placement.lower() == "upperright":
            c_shape = Topology.Translate(c_shape, -width/2, -length/2, 0)
        elif placement.lower() == "upperleft":
            c_shape = Topology.Translate(c_shape, width/2, -length/2, 0)
        elif placement.lower() == "lowerright":
            c_shape = Topology.Translate(c_shape, -width/2, length/2, 0)
        
        if direction != [0, 0, 1]:
            c_shape = Topology.Orient(c_shape, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return c_shape

    @staticmethod
    def Cycles(wire, maxVertices: int = 4, tolerance: float = 0.0001) -> list:
        """
        Returns the closed circuits of wires found within the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        maxVertices : int , optional
            The maximum number of vertices of the circuits to be searched. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of circuits (closed wires) found within the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        def vIndex(v, vList, tolerance=0.0001):
            for i in range(len(vList)):
                if Vertex.Distance(v, vList[i]) <= tolerance:
                    return i+1
            return None
        
        #  rotate cycle path such that it begins with the smallest node
        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:]+path[:n]

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def isNew(cycles, path):
            return not path in cycles

        def visited(node, path):
            return node in path

        def findNewCycles(graph, cycles, path, maxVertices):
            if len(path) > maxVertices:
                return
            start_node = path[0]
            next_node= None
            sub = []

            #visit each edge and each node of each edge
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                        if node1 == start_node:
                            next_node = node2
                        else:
                            next_node = node1
                        if not visited(next_node, path):
                                # neighbor node not on path yet
                                sub = [next_node]
                                sub.extend(path)
                                # explore extended path
                                findNewCycles(graph, cycles, sub, maxVertices);
                        elif len(path) > 2  and next_node == path[-1]:
                                # cycle found
                                p = rotate_to_smallest(path);
                                inv = invert(p)
                                if isNew(cycles, p) and isNew(cycles, inv):
                                    cycles.append(p)

        def main(graph, cycles, maxVertices):
            returnValue = []
            for edge in graph:
                for node in edge:
                    findNewCycles(graph, cycles, [node], maxVertices)
            for cy in cycles:
                row = []
                for node in cy:
                    row.append(node)
                returnValue.append(row)
            return returnValue

        tEdges = []
        _ = wire.Edges(None, tEdges)
        tVertices = Topology.Vertices(wire)
        tVertices = tVertices

        graph = []
        for anEdge in tEdges:
            graph.append([vIndex(Edge.StartVertex(anEdge), tVertices, tolerance), vIndex(Edge.EndVertex(anEdge), tVertices, tolerance)]) # Hook to Core

        cycles = []
        resultingCycles = main(graph, cycles, maxVertices)

        result = []
        for aRow in resultingCycles:
            row = []
            for anIndex in aRow:
                row.append(tVertices[anIndex-1])
            result.append(row)

        resultWires = []
        for i in range(len(result)):
            c = result[i]
            resultEdges = []
            for j in range(len(c)-1):
                v1 = c[j]
                v2 = c[j+1]
                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
                resultEdges.append(e)
            e = Edge.ByStartVertexEndVertex(c[len(c)-1], c[0], tolerance=tolerance, silent=True)
            resultEdges.append(e)
            resultWire = Wire.ByEdges(resultEdges, tolerance=tolerance)
            resultWires.append(resultWire)
        return resultWires

    @staticmethod
    def Edges(wire) -> list:
        """
        Returns the edges of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of edges.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None
        edges = []
        _ = wire.Edges(None, edges)
        return edges

    @staticmethod
    def Einstein(origin= None, radius: float = 0.5, direction: list = [0, 0, 1], placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physicist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the tile. The default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import math

        def cos(angle):
            return math.cos(math.radians(angle))
        def sin(angle):
            return math.sin(math.radians(angle))
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        d = cos(30)*radius
        v1 = Vertex.ByCoordinates(0, 0, 0)
        v2 = Vertex.ByCoordinates(cos(30)*d, sin(30)*d, 0)
        v3 = Vertex.ByCoordinates(radius, 0)
        v4 = Vertex.ByCoordinates(2*radius, 0)
        v5 = Vertex.ByCoordinates(2*radius+cos(60)*radius*0.5, sin(30)*d, 0)
        v6 = Vertex.ByCoordinates(1.5*radius, d)
        v7 = Vertex.ByCoordinates(1.5*radius, 2*d)
        v8 = Vertex.ByCoordinates(radius, 2*d)
        v9 = Vertex.ByCoordinates(radius-cos(60)*0.5*radius, 2*d+sin(60)*0.5*radius)
        v10 = Vertex.ByCoordinates(0, 2*d)
        v11 = Vertex.ByCoordinates(0, d)
        v12 = Vertex.ByCoordinates(-radius*0.5, d)
        v13 = Vertex.ByCoordinates(-cos(30)*d, sin(30)*d, 0)
        vertices = [v1, v13, v12, v11, v10, v9, v8, v7, v6, v5, v4, v3, v2]
        # [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
        einstein = Wire.ByVertices(vertices, close=True, tolerance=tolerance)

        einstein = Topology.Rotate(einstein, origin=origin, axis=[1,0,0], angle=180)
        
        if placement.lower() == "lowerleft":
            einstein = Topology.Translate(einstein, radius, d, 0)
        dx = Vertex.X(origin, mantissa=mantissa)
        dy = Vertex.Y(origin, mantissa=mantissa)
        dz = Vertex.Z(origin, mantissa=mantissa)
        einstein = Topology.Translate(einstein, dx, dy, dz)
        if direction != [0, 0, 1]:
            einstein = Topology.Orient(einstein, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return einstein
    
    @staticmethod
    def Ellipse(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: float = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created ellipse

        """
        ellipseAll = Wire.EllipseAll(origin=origin, inputMode=inputMode, width=width, length=length, focalLength=focalLength, eccentricity=eccentricity, majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close, direction=direction, placement=placement, tolerance=tolerance)
        return ellipseAll["ellipse"]

    @staticmethod
    def EllipseAll(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "ellipse" : The ellipse (topologic_core.Wire)
            2. "foci" : The two focal points (topologic_core.Cluster containing two vertices)
            3. "a" : The major axis length
            4. "b" : The minor axis length
            5. "c" : The focal length
            6. "e" : The eccentricity
            7. "width" : The width
            8. "length" : The length

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        if inputMode not in [1, 2, 3, 4]:
            return None
        if placement.lower() not in ["center", "lowerleft"]:
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            return None
        width = abs(width)
        length = abs(length)
        focalLength= abs(focalLength)
        eccentricity=abs(eccentricity)
        majorAxisLength=abs(majorAxisLength)
        minorAxisLength=abs(minorAxisLength)
        sides = abs(sides)
        if width <= tolerance or length <= tolerance or focalLength <= tolerance or eccentricity <= tolerance or majorAxisLength <= tolerance or minorAxisLength <= tolerance or sides < 3:
            return None
        if inputMode == 1:
            w = width
            l = length
            a = width/2
            b = length/2
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
        elif inputMode == 2:
            c = focalLength
            e = eccentricity
            a = c/e
            b = math.sqrt(abs(a**2 - c**2))
            w = a*2
            l = b*2
        elif inputMode == 3:
            c = focalLength
            b = minorAxisLength
            a = math.sqrt(abs(b**2 + c**2))
            e = c/a
            w = a*2
            l = b*2
        elif inputMode == 4:
            a = majorAxisLength
            b = minorAxisLength
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
            w = a*2
            l = b*2
        else:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle - fromAngle) <= tolerance:
            return None

        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*a + Vertex.X(origin)
            y = math.cos(angle)*b + Vertex.Y(origin)
            z = Vertex.Z(origin)
            xList.append(x)
            yList.append(y)
            baseV.append(Vertex.ByCoordinates(x, y, z))

        if angleRange == 360:
            baseWire = Wire.ByVertices(baseV[::-1], close=False, tolerance=tolerance) #reversing the list so that the normal points up in Blender
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close, tolerance=tolerance) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, a, b, 0)
        baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        # Create a Cluster of the two foci
        v1 = Vertex.ByCoordinates(c+Vertex.X(origin), 0+Vertex.Y(origin), 0)
        v2 = Vertex.ByCoordinates(-c+Vertex.X(origin), 0+Vertex.Y(origin), 0)
        foci = Cluster.ByTopologies([v1, v2])
        if placement.lower() == "lowerleft":
            foci = Topology.Translate(foci, a, b, 0)
        foci = Topology.Orient(foci, origin=origin, dirA=[0, 0, 1], dirB=direction)
        d = {}
        d['ellipse'] = baseWire
        d['foci'] = foci
        d['a'] = a
        d['b'] = b
        d['c'] = c
        d['e'] = e
        d['w'] = w
        d['l'] = l
        return d

    @staticmethod
    def EndVertex(wire):
        """
        Returns the end vertex of the input wire. The wire must be manifold and open.

        """
        sv, ev = Wire.StartEndVertices(wire)
        return ev
    
    @staticmethod
    def ExteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6) -> list:
        """
        Returns the exterior angles of the input wire in degrees. The wire must be planar, manifold, and closed.
        
        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        mantissa : int , optional
            The length of the desired mantissa. The default is 6.
        
        Returns
        -------
        list
            The list of exterior angles.
        
        """
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None")
            return None
        if not Wire.IsClosed(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None")
            return None
        
        interior_angles = Wire.InteriorAngles(wire, mantissa=mantissa)
        exterior_angles = [round(360-a, mantissa) for a in interior_angles]
        return exterior_angles
    
    @staticmethod
    def ExternalBoundary(wire):
        """
        Returns the external boundary (cluster of vertices where degree == 1) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input wire. This is a cluster of vertices of degree == 1.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.ExternalBoundary - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        return Cluster.ByTopologies([v for v in Topology.Vertices(wire) if (Vertex.Degree(v, wire, topologyType="edge") == 1)])
    
    @staticmethod
    def Fillet(wire, radius: float = 0, sides: int = 16, radiusKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input wire given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        radius : float
            The desired radius of the fillet.
        radiusKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired fillet radius. The default is None.
        sides : int , optional
            The number of sides (segments) of the fillet. The default is 16.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The filleted wire.

        """
        def start_from(edge, v):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(v, ev) < Vertex.Distance(v, sv):
                return Edge.Reverse(edge)
            return edge
        
        def compute_kite_edges(alpha, r):
            # Convert angle to radians
            alpha = math.radians(alpha) *0.5
            h = r/math.cos(alpha)
            a = math.sqrt(h*h - r*r)
            return [a,h]
        
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not manifold. Returning None.")
            return None
        if not Topology.IsPlanar(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not planar. Returning None.")
            return None

        orig_radius = radius
        f = Face.BoundingRectangle(wire, tolerance=tolerance)
        normal = Face.Normal(f)
        flat_wire = Topology.Flatten(wire, origin=Vertex.Origin(), direction=normal)
        vertices = Topology.Vertices(flat_wire)
        final_vertices = []
        for v in vertices:
            radius = orig_radius
            edges = Topology.SuperTopologies(v, flat_wire, topologyType="edge")
            if len(edges) == 2:
                for edge in edges:
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(v, ev) <= tolerance:
                        edge0 = edge
                    else:
                        edge1 = edge
                ang = Edge.Angle(edge0, edge1)
                e1 = start_from(edge0, v)
                e2 = start_from(edge1, v)

                dir1 = Edge.Direction(e1)
                dir2 = Edge.Direction(e2)
                if Vector.IsParallel(dir1, dir2) or Vector.IsAntiParallel(dir1, dir2):
                    pass
                else:
                    if isinstance(radiusKey, str):
                        d = Topology.Dictionary(v)
                        if Topology.IsInstance(d, "Dictionary"):
                            v_radius = Dictionary.ValueAtKey(d, radiusKey)
                            if isinstance(v_radius, float) or isinstance(v_radius, int):
                                if v_radius >= 0:
                                    radius = v_radius
                    if radius > 0:
                        dir_bisector = Vector.Bisect(dir1,dir2)
                        a, h = compute_kite_edges(ang, radius)
                        if a <= Edge.Length(e1) and a <= Edge.Length(e2):
                            v1 = Topology.TranslateByDirectionDistance(v, dir1, a)
                            center = Topology.TranslateByDirectionDistance(v, dir_bisector, h)
                            v2 = Topology.TranslateByDirectionDistance(v, dir2, a)
                            fillet = Wire.Circle(origin=center, radius=radius, close=True)
                            bisector = Edge.ByVertices(v, center)
                            mid_vertex = Topology.Slice(bisector, fillet)
                            mid_vertex = Topology.Vertices(mid_vertex)[1]
                            fillet = Wire.Arc(v1, mid_vertex, v2, sides=sides, close= False)
                            f_sv = Wire.StartVertex(fillet)
                            if Vertex.Distance(f_sv, edge1) < Vertex.Distance(f_sv, edge0):
                                fillet = Wire.Reverse(fillet)
                            final_vertices += Topology.Vertices(fillet)
                        else:
                            if not silent:
                                print("Wire.Fillet - Error: The specified fillet radius is too large to be applied. Skipping.")
                    else:
                        final_vertices.append(v)
            else:
                final_vertices.append(v)
        flat_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        # Unflatten the wire
        return_wire = Topology.Unflatten(flat_wire, origin=Vertex.Origin(), direction=normal)
        return return_wire
    
    @staticmethod
    def InteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6) -> list:
        """
        Returns the interior angles of the input wire in degrees. The wire must be planar, manifold, and closed.
        This code has been contributed by Yidan Xue.
        
        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        list
            The list of interior angles.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None")
            return None
        if not Wire.IsClosed(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None")
            return None
        
        f = Face.ByWire(wire)
        normal = Face.Normal(f)
        origin = Topology.Centroid(f)
        w = Topology.Flatten(wire, origin=origin, direction=normal)
        angles = []
        edges = Topology.Edges(w)
        e1 = edges[len(edges)-1]
        e2 = edges[0]
        a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
        angles.append(a)
        for i in range(len(edges)-1):
            e1 = edges[i]
            e2 = edges[i+1]
            a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
            angles.append(round(a, mantissa))
        if abs(sum(angles)-(len(angles)-2)*180)<tolerance:
            return angles
        else:
            angles = [360-ang for ang in angles]
            return angles

    @staticmethod
    def Interpolate(wires: list, n: int = 5, outputType: str = "default", mapping: str = "default", tolerance: float = 0.0001):
        """
        Creates *n* number of wires that interpolate between wireA and wireB.

        Parameters
        ----------
        wireA : topologic_core.Wire
            The first input wire.
        wireB : topologic_core.Wire
            The second input wire.
        n : int , optional
            The number of intermediate wires to create. The default is 5.
        outputType : str , optional
            The desired type of output. The options are case insensitive. The default is "contour". The options are:
                - "Default" or "Contours" (wires are not connected)
                - "Raster or "Zigzag" or "Toolpath" (the wire ends are connected to create a continuous path)
                - "Grid" (the wire ends are connected to create a grid). 
        mapping : str , optional
            The desired type of mapping for wires with different number of vertices. It is case insensitive. The default is "default". The options are:
                - "Default" or "Repeat" which repeats the last vertex of the wire with the least number of vertices
                - "Nearest" which maps the vertices of one wire to the nearest vertex of the next wire creating a list of equal number of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Topology
            The created interpolated wires as well as the input wires. The return type can be a topologic_core.Cluster or a topologic_core.Wire based on options.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Helper import Helper
        
        outputType = outputType.lower()
        if outputType not in ["default", "contours", "raster", "zigzag", "toolpath", "grid"]:
            return None
        if outputType == "default" or outputType == "contours":
            outputType = "contours"
        if outputType == "raster" or outputType == "zigzag" or outputType == "toolpath":
            outputType = "zigzag"
        
        mapping = mapping.lower()
        if mapping not in ["default", "nearest", "repeat"]:
            print("Wire.Interpolate - Error: The mapping input parameter is not recognized. Returning None.")
            return None
        
        def nearestVertex(v, vertices):
            distances = [Vertex.Distance(v, vertex) for vertex in vertices]
            return vertices[distances.index(sorted(distances)[0])]
        
        def replicate(vertices, mapping="default"):
            vertices = Helper.Repeat(vertices)
            finalList = vertices
            if mapping == "nearest":
                finalList = [vertices[0]]
                for i in range(len(vertices)-1):
                    loopA = vertices[i]
                    loopB = vertices[i+1]
                    nearestVertices = []
                    for j in range(len(loopA)):
                        nv = nearestVertex(loopA[j], loopB)
                        nearestVertices.append(nv)
                    finalList.append(nearestVertices)
            return finalList
        
        def process(verticesA, verticesB, n=5):
            contours = [verticesA]
            for i in range(1, n+1):
                u = float(i)/float(n+1)
                temp_vertices = []
                for j in range(len(verticesA)):
                    temp_v = Edge.VertexByParameter(Edge.ByVertices([verticesA[j], verticesB[j]], tolerance=tolerance), u)
                    temp_vertices.append(temp_v)
                contours.append(temp_vertices)
            return contours
        
        if len(wires) < 2:
            return None
        
        vertices = []
        for wire in wires:
            vertices.append(Topology.SubTopologies(wire, subTopologyType="vertex"))
        vertices = replicate(vertices, mapping=mapping)
        contours = []
        
        finalWires = []
        for i in range(len(vertices)-1):
            verticesA = vertices[i]
            verticesB = vertices[i+1]
            contour = process(verticesA=verticesA, verticesB=verticesB, n=n)
            contours += contour
            for c in contour:
                finalWires.append(Wire.ByVertices(c, close=Wire.IsClosed(wires[i], tolerance=tolerance)))

        contours.append(vertices[-1])
        finalWires.append(wires[-1])
        ridges = []
        if outputType == "grid" or outputType == "zigzag":
            for i in range(len(contours)-1):
                verticesA = contours[i]
                verticesB = contours[i+1]
                if outputType == "grid":
                    for j in range(len(verticesA)):
                        ridges.append(Edge.ByVertices([verticesA[j], verticesB[j]], tolerance=tolerance))
                elif outputType == "zigzag":
                    if i%2 == 0:
                        sv = verticesA[-1]
                        ev = verticesB[-1]
                        ridges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))
                    else:
                        sv = verticesA[0]
                        ev = verticesB[0]
                        ridges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))

        return Topology.SelfMerge(Cluster.ByTopologies(finalWires+ridges), tolerance=tolerance)
    
    @staticmethod
    def Invert(wire, tolerance: float = 0.0001):
        """
        Creates a wire that is an inverse (mirror) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The inverted wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            return None
        vertices = Topology.Vertices(wire)
        reversed_vertices = vertices[::-1]
        return Wire.ByVertices(reversed_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)

    @staticmethod
    def IsClosed(wire) -> bool:
        """
        Returns True if the input wire is closed. Returns False otherwise.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        bool
            True if the input wire is closed. False otherwise.

        """
        status = None
        if wire:
            if Topology.IsInstance(wire, "Wire"):
                status = wire.IsClosed()
        return status
    
    @staticmethod
    def IsManifold(wire, silent: bool = False) -> bool:
        """
        Returns True if the input wire is manifold. Returns False otherwise. A manifold wire is one where its vertices have a degree of 1 or 2.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        bool
            True if the input wire is manifold. False otherwise.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.IsManifold - Error: The input wire parameter is not a valid topologic wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        
        vertices = Topology.Vertices(wire)
        for v in vertices:
            if Vertex.Degree(v, hostTopology=wire) > 2:
                return False
        return True

    @staticmethod
    def IsSimilar(wireA, wireB, angTolerance: float = 0.1, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input wires are similar. Returns False otherwise. The wires must be closed.

        Parameters
        ----------
        wireA : topologic_core.Wire
            The first input wire.
        wireB : topologic_core.Wire
            The second input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two input wires are similar. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        
        def isCyclicallyEquivalent(u, v, lengthTolerance, angleTolerance):
            n, i, j = len(u), 0, 0
            if n != len(v):
                return False
            while i < n and j < n:
                if (i % 2) == 0:
                    tol = lengthTolerance
                else:
                    tol = angleTolerance
                k = 1
                while k <= n and math.fabs(u[(i + k) % n]- v[(j + k) % n]) <= tol:
                    k += 1
                if k > n:
                    return True
                if math.fabs(u[(i + k) % n]- v[(j + k) % n]) > tol:
                    i += k
                else:
                    j += k
            return False

        def angleBetweenEdges(e1, e2, tolerance=0.0001):
            a = Vertex.X(Edge.EndVertex(e1)) - Vertex.X(Edge.StartVertex(e1))
            b = Vertex.Y(Edge.EndVertex(e1)) - Vertex.Y(Edge.StartVertex(e1))
            c = Vertex.Z(Edge.EndVertex(e1)) - Vertex.Z(Edge.StartVertex(e1))
            d = Vertex.Distance(Edge.EndVertex(e1), Edge.StartVertex(e2))
            if d <= tolerance:
                d = Vertex.X(Edge.StartVertex(e2)) - Vertex.X(Edge.EndVertex(e2))
                e = Vertex.Y(Edge.StartVertex(e2)) - Vertex.Y(Edge.EndVertex(e2))
                f = Vertex.Z(Edge.StartVertex(e2)) - Vertex.Z(Edge.EndVertex(e2))
            else:
                d = Vertex.X(Edge.EndVertex(e2)) - Vertex.X(Edge.StartVertex(e2))
                e = Vertex.Y(Edge.EndVertex(e2)) - Vertex.Y(Edge.StartVertex(e2))
                f = Vertex.Z(Edge.EndVertex(e2)) - Vertex.Z(Edge.StartVertex(e2))
            dotProduct = a*d + b*e + c*f
            modOfVector1 = math.sqrt( a*a + b*b + c*c)*math.sqrt(d*d + e*e + f*f) 
            angle = dotProduct/modOfVector1
            angleInDegrees = math.degrees(math.acos(angle))
            return angleInDegrees

        def getInteriorAngles(edges, tolerance=0.0001):
            angles = []
            for i in range(len(edges)-1):
                e1 = edges[i]
                e2 = edges[i+1]
                angles.append(angleBetweenEdges(e1, e2, tolerance=tolerance))
            return angles

        def getRep(edges, tolerance=0.0001):
            angles = getInteriorAngles(edges, tolerance=tolerance)
            lengths = []
            for anEdge in edges:
                lengths.append(Edge.Length(anEdge))
            minLength = min(lengths)
            normalizedLengths = []
            for aLength in lengths:
                normalizedLengths.append(aLength/minLength)
            return [x for x in itertools.chain(*itertools.zip_longest(normalizedLengths, angles)) if x is not None]
        
        if (wireA.IsClosed() == False):
            return None
        if (wireB.IsClosed() == False):
            return None
        edgesA = []
        _ = wireA.Edges(None, edgesA)
        edgesB = []
        _ = wireB.Edges(None, edgesB)
        if len(edgesA) != len(edgesB):
            return False
        repA = getRep(list(edgesA), tolerance=tolerance)
        repB = getRep(list(edgesB), tolerance=tolerance)
        if isCyclicallyEquivalent(repA, repB, tolerance, angTolerance):
            return True
        if isCyclicallyEquivalent(repA, repB[::-1], tolerance, angTolerance):
            return True
        return False

    @staticmethod
    def IShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c =0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates an I-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the I-shape. The default is None which results in the I-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the I-shape. The default is 1.0.
        length : float , optional
            The overall length of the I-shape. The default is 1.0.
        a : float , optional
            The hortizontal thickness of the central vertical arm of the I-shape. The default is 0.25.
        b : float , optional
            The vertical thickness of the lower horizontal arm of the I-shape. The default is 0.25.
        c : float , optional
            The vertical thickness of the upper horizontal arm of the I-shape. The default is 0.25.
        direction : list , optional
            The vector representing the up direction of the I-shape. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the I-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created I-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Wire.IShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Wire.IShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Wire.IShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Wire.IShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Wire.IShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Wire.IShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Wire.IShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Wire.IShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Wire.IShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Wire.IShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Wire.IShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.IShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.IShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.IShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the I-shape (counterclockwise)
        v1 = Vertex.Origin()  # Base origin
        v2 = Vertex.ByCoordinates(width, 0)
        v3 = Vertex.ByCoordinates(width, b)
        v4 = Vertex.ByCoordinates(width/2+a/2, b)
        v5 = Vertex.ByCoordinates(width/2+a/2, length-c)
        v6 = Vertex.ByCoordinates(width, length-c)
        v7 = Vertex.ByCoordinates(width, length)
        v8 = Vertex.ByCoordinates(0, length)
        v9 = Vertex.ByCoordinates(0, length-c)
        v10 = Vertex.ByCoordinates(width/2-a/2, length-c)
        v11 = Vertex.ByCoordinates(width/2-a/2, b)
        v12 = Vertex.ByCoordinates(0,b)

        # Create the I-shaped wire
        i_shape = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], close=True, tolerance=tolerance)
        i_shape = Topology.Translate(i_shape, -width/2, -length/2, 0)
        i_shape = Topology.Translate(i_shape, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        reverse = False
        if flipHorizontal == True:
            xScale = -1
            reverse = not reverse
        else:
            xScale = 1
        if flipVertical == True:
            yScale = -1
            reverse = not reverse
        else:
            yScale = 1
        if xScale == -1 or yScale == -1:
            i_shape = Topology.Scale(i_shape, origin=origin, x=xScale, y=yScale, z=1)
            if reverse == True:
                i_shape = Wire.Reverse(i_shape)
        if placement.lower() == "lowerleft":
            i_shape = Topology.Translate(i_shape, width/2, length/2, 0)
        elif placement.lower() == "upperright":
            i_shape = Topology.Translate(i_shape, -width/2, -length/2, 0)
        elif placement.lower() == "upperleft":
            i_shape = Topology.Translate(i_shape, width/2, -length/2, 0)
        elif placement.lower() == "lowerright":
            i_shape = Topology.Translate(i_shape, -width/2, length/2, 0)
        
        if direction != [0, 0, 1]:
            i_shape = Topology.Orient(i_shape, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return i_shape

    @staticmethod
    def Length(wire, mantissa: int = 6) -> float:
        """
        Returns the length of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The length of the input wire. Test

        """
        from topologicpy.Edge import Edge

        if not wire:
            return None
        if not Topology.IsInstance(wire, "Wire"):
            return None
        totalLength = None
        try:
            edges = []
            _ = wire.Edges(None, edges)
            totalLength = 0
            for anEdge in edges:
                totalLength = totalLength + Edge.Length(anEdge)
            totalLength = round(totalLength, mantissa)
        except:
            totalLength = None
        return totalLength

    @staticmethod
    def Line(origin= None, length: float = 1, direction: list = [1, 0, 0], sides: int = 2, placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a straight line wire using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. The default is None which results in the edge being placed at (0, 0, 0).
        length : float , optional
            The desired length of the edge. The default is 1.0.
        direction : list , optional
            The desired direction (vector) of the edge. The default is [1, 0, 0] (along the X-axis).
        sides : int , optional
            The desired number of sides/segments. The minimum number of sides is 2. The default is 2.
        placement : str , optional
            The desired placement of the edge. The options are:
            1. "center" which places the center of the edge at the origin.
            2. "start" which places the start of the edge at the origin.
            3. "end" which places the end of the edge at the origin.
            The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Edge
            The created edge
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Line - Error: The input origin is not a valid vertex. Returning None.")
            return None
        if length <= 0:
            print("Wire.Line - Error: The input length is less than or equal to zero. Returning None.")
            return None
        if not isinstance(direction, list):
            print("Wire.Line - Error: The input direction is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            print("Wire.Line - Error: The length of the input direction is not equal to three. Returning None.")
            return None
        if sides < 2:
            print("Wire.Line - Error: The number of sides cannot be less than two. Consider using Edge.Line() instead. Returning None.")
            return None
        edge = Edge.Line(origin=origin, length=length, direction=direction, placement=placement)
        vertices = [Edge.StartVertex(edge)]
        unitDistance = float(1)/float(sides)
        for i in range(1, sides):
            vertices.append(Edge.VertexByParameter(edge, i*unitDistance))
        vertices.append(Edge.EndVertex(edge))
        return Wire.ByVertices(vertices, closed=False, tolerance=tolerance)

    @staticmethod
    def LShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates an L-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the L-shape. The default is None which results in the L-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the L-shape. The default is 1.0.
        length : float , optional
            The overall length of the L-shape. The default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the L-shape. The default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the L-shape. The default is 0.25.
        direction : list , optional
            The vector representing the up direction of the L-shape. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the L-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created L-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Wire.LShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Wire.LShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Wire.LShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Wire.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Wire.LShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance):
            if not silent:
                print("Wire.LShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.LShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.LShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.LShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the L-shape (counterclockwise)
        v1 = Vertex.Origin()  # Base origin
        v2 = Vertex.ByCoordinates(width, 0)  # End of horizontal arm
        v3 = Vertex.ByCoordinates(width, b)  # Top of horizontal arm
        v4 = Vertex.ByCoordinates(a, b)  # Transition to vertical arm
        v5 = Vertex.ByCoordinates(a, length)  # End of vertical arm
        v6 = Vertex.ByCoordinates(0, length)  # Top of vertical arm

        # Create the L-shaped wire
        l_shape = Wire.ByVertices([v1, v2, v3, v4, v5, v6], close=True, tolerance=tolerance)
        l_shape = Topology.Translate(l_shape, -width/2, -length/2, 0)
        l_shape = Topology.Translate(l_shape, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        reverse = False
        if flipHorizontal == True:
            xScale = -1
            reverse = not reverse
        else:
            xScale = 1
        if flipVertical == True:
            yScale = -1
            reverse = not reverse
        else:
            yScale = 1
        if xScale == -1 or yScale == -1:
            l_shape = Topology.Scale(l_shape, origin=origing, x=xScale, y=yScale, z=1)
            if reverse == True:
                l_shape = Wire.Reverse(l_shape)
        if placement.lower() == "lowerleft":
            l_shape = Topology.Translate(l_shape, width/2, length/2, 0)
        elif placement.lower() == "upperright":
            l_shape = Topology.Translate(l_shape, -width/2, -length/2, 0)
        elif placement.lower() == "upperleft":
            l_shape = Topology.Translate(l_shape, width/2, -length/2, 0)
        elif placement.lower() == "lowerright":
            l_shape = Topology.Translate(l_shape, -width/2, length/2, 0)
        
        if direction != [0, 0, 1]:
            l_shape = Topology.Orient(l_shape, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return l_shape

    @staticmethod
    def Miter(wire, offset: float = 0, offsetKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input wire given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float
            The desired offset length of the miter along each edge.
        offsetKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired offset length. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The filleted wire.

        """
        def start_from(edge, v):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(v, ev) < Vertex.Distance(v, sv):
                return Edge.Reverse(edge)
            return edge
        
        def compute_kite_edges(alpha, r):
            # Convert angle to radians
            alpha = math.radians(alpha) *0.5
            h = r/math.cos(alpha)
            a = math.sqrt(h*h - r*r)
            return [a,h]
        
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not manifold. Returning None.")
            return None
        if not Topology.IsPlanar(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not planar. Returning None.")
            return None

        orig_offset = offset
        f = Face.BoundingRectangle(wire, tolerance=tolerance)
        normal = Face.Normal(f)
        flat_wire = Topology.Flatten(wire, origin=Vertex.Origin(), direction=normal)
        vertices = Topology.Vertices(flat_wire)
        final_vertices = []
        miters = []
        for v in vertices:
            offset = orig_offset
            edges = Topology.SuperTopologies(v, flat_wire, topologyType="edge")
            if len(edges) == 2:
                for edge in edges:
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(v, ev) <= tolerance:
                        edge0 = edge
                    else:
                        edge1 = edge
                ang = Edge.Angle(edge0, edge1)
                e1 = start_from(edge0, v)
                e2 = start_from(edge1, v)

                dir1 = Edge.Direction(e1)
                dir2 = Edge.Direction(e2)
                if Vector.IsParallel(dir1, dir2) or Vector.IsAntiParallel(dir1, dir2):
                    pass
                else:
                    if isinstance(offsetKey, str):
                        d = Topology.Dictionary(v)
                        if Topology.IsInstance(d, "Dictionary"):
                            v_offset = Dictionary.ValueAtKey(d, offsetKey)
                            if isinstance(v_offset, float) or isinstance(v_offset, int):
                                if v_offset >= 0:
                                    offset = v_offset
                    if offset > 0 and offset <= Edge.Length(e1) and offset <=Edge.Length(e2):
                        v1 = Topology.TranslateByDirectionDistance(v, dir1, offset)
                        v2 = Topology.TranslateByDirectionDistance(v, dir2, offset)
                        final_vertices += [v1,v2]
                    else:
                        print("Wire.Fillet - Warning: The input offset parameter is greater than the length of the edge. Skipping.")
                        final_vertices.append(v)
            else:
                final_vertices.append(v)
        flat_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        # Unflatten the wire
        return_wire = Topology.Unflatten(flat_wire, origin=Vertex.Origin(), direction=normal)
        return return_wire
    
    @staticmethod
    def Normal(wire, outputType="xyz", mantissa=6):
        """
        Returns the normal vector to the input wire. A normal vector of a wire is a vector perpendicular to it.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The normal vector to the input face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from random import sample
        import time
        import os
        import warnings

        try:
            import numpy as np
        except:
            print("Wire.Normal - Warning: Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except:
                os.system("pip install numpy --user")
            try:
                import numpy as np
                print("Wire.Normal - Warning: numpy library installed correctly.")
            except:
                warnings.warn("Wire.Normal - Error: Could not import numpy. Please try to install numpy manually. Returning None.")
                return None

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Normal - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        
        vertices = Topology.Vertices(wire)
        result = True
        start = time.time()
        period = 0
        while result and period < 30:
            vList = sample(vertices, 3)
            result = Vertex.AreCollinear(vList)
            end = time.time()
            period = end - start
        if result == True:
            print("Wire.BoundingRectangle - Error: Could not find three vertices that are not colinear within 30 seconds. Returning None.")
            return None
        vertices = [Vertex.Coordinates(v, mantissa=mantissa) for v in vList]
        
        if len(vertices) < 3:
            print("Wire.Normal - Error: At least three vertices are required to define a plane. Returning None.")
            return None
        
        # Convert vertices to numpy array for easier manipulation
        vertices = np.array(vertices)
        
        # Try to find two non-collinear edge vectors
        vec1 = None
        vec2 = None
        for i in range(1, len(vertices)):
            for j in range(i + 1, len(vertices)):
                temp_vec1 = vertices[i] - vertices[0]
                temp_vec2 = vertices[j] - vertices[0]
                cross_product = np.cross(temp_vec1, temp_vec2)
                if np.linalg.norm(cross_product) > 1e-6:  # Check if the cross product is not near zero
                    vec1 = temp_vec1
                    vec2 = temp_vec2
                    break
            if vec1 is not None and vec2 is not None:
                break
        
        if vec1 is None or vec2 is None:
            print("Wire.Normal - Error: The given vertices do not form a valid plane (all vertices might be collinear). Returning None.")
            return None
        
        # Calculate the cross product of the two edge vectors
        normal = np.cross(vec1, vec2)

        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            print("Wire.Normal - Error: The given vertices do not form a valid plane (cross product resulted in a zero vector). Returning None.")
            return None
        
        normal = normal / normal_length
        normal = normal.tolist()
        normal = [round(x, mantissa) for x in normal]
        return_normal = []
        outputType = list(outputType.lower())
        for axis in outputType:
            if axis == "x":
                return_normal.append(normal[0])
            elif axis == "y":
                return_normal.append(normal[1])
            elif axis == "z":
                return_normal.append(normal[2])
        return return_normal
    
    @staticmethod
    def OrientEdges(wire, vertexA, transferDictionaries = False, tolerance=0.0001):
        """
        Returns a correctly oriented head-to-tail version of the input wire. The input wire must be manifold.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertexA : topologic_core.Vertex
            The desired start vertex of the wire.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire are transfered to the new wire. Otherwise, they are not. The default is False.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The oriented wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.OrientEdges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Wire.OrientEdges - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.OrientEdges - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        oriented_edges = []
        remaining_edges = Topology.Edges(wire)
        original_vertices = Topology.Vertices(wire)
        if transferDictionaries:
            edge_selectors = []
            for i, e_s in enumerate(remaining_edges):
                s = Topology.Centroid(e_s)
                d = Topology.Dictionary(e_s)
                s = Topology.SetDictionary(s, d)
                edge_selectors.append(s)

        current_vertex = vertexA
        while remaining_edges:
            next_edge = None
            for edge in remaining_edges:
                if Vertex.Distance(Edge.StartVertex(edge), current_vertex) <= tolerance:
                    next_edge = edge
                    break
                elif Vertex.Distance(Edge.EndVertex(edge), current_vertex) <= tolerance:
                    next_edge = Edge.Reverse(edge)
                    break

            if next_edge:
                oriented_edges.append(next_edge)
                remaining_edges.remove(next_edge)
                current_vertex = Edge.EndVertex(next_edge)
            else:
                # Unable to find a next edge connected to the current vertex
                break
        vertices = [Edge.StartVertex(oriented_edges[0])]
        for i, edge in enumerate(oriented_edges):
            vertices.append(Edge.EndVertex(edge))
            
        return_wire = Wire.ByVertices(vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        if transferDictionaries:
            return_wire = Topology.TransferDictionariesBySelectors(return_wire, selectors=edge_selectors, tranEdges=True)
            return_wire = Topology.TransferDictionariesBySelectors(return_wire, selectors=original_vertices, tranVertices=True)
        return_wire = Topology.SetDictionary(return_wire, Topology.Dictionary(wire), silent=True)
        return return_wire

    @staticmethod
    def Planarize(wire, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a planarized version of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.
        origin : topologic_core.Vertex , optional
            The desired origin of the plane unto which the planar wire will be projected. If set to None, the centroid of the input wire will be chosen. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        topologic_core.Wire
            The planarized wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Planarize - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Planarize - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        vertices = Topology.Vertices(wire)
        edges = Topology.Edges(wire)
        plane_equation = Vertex.PlaneEquation(vertices, mantissa=mantissa)
        rect = Face.RectangleByPlaneEquation(origin=origin , equation=plane_equation, tolerance=tolerance)
        new_vertices = [Vertex.Project(v, rect, mantissa=mantissa) for v in vertices]
        new_vertices = Vertex.Fuse(new_vertices, mantissa=mantissa, tolerance=tolerance)
        new_edges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            sv1 = Vertex.Project(sv, rect)
            i = Vertex.Index(sv1, new_vertices, tolerance=tolerance)
            if i:
                sv1 = new_vertices[i]
            ev1 = Vertex.Project(ev, rect)
            i = Vertex.Index(ev1, new_vertices, tolerance=tolerance)
            if i:
                ev1 = new_vertices[i]
            new_edges.append(Edge.ByVertices([sv1, ev1]))
        return Topology.SelfMerge(Cluster.ByTopologies(new_edges), tolerance=tolerance)

    @staticmethod
    def Project(wire, face, direction: list = None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a projection of the input wire unto the input face.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        face : topologic_core.Face
            The face unto which to project the input wire.
        direction : list, optional
            The vector representing the direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The projected wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not wire:
            return None
        if not Topology.IsInstance(wire, "Wire"):
            return None
        if not face:
            return None
        if not Topology.IsInstance(face, "Face"):
            return None
        if not direction:
            direction = -1*Face.Normal(face, outputType="xyz", mantissa=mantissa)
        large_face = Topology.Scale(face, Topology.CenterOfMass(face), 500, 500, 500)
        edges = []
        _ = wire.Edges(None, edges)
        projected_edges = []

        if large_face:
            if (Topology.Type(large_face) == Topology.TypeID("Face")):
                for edge in edges:
                    if edge:
                        if (Topology.Type(edge) == Topology.TypeID("Edge")):
                            sv = Edge.StartVertex(edge)
                            ev = Edge.EndVertex(edge)

                            psv = Vertex.Project(vertex=sv, face=large_face, direction=direction)
                            pev = Vertex.Project(vertex=ev, face=large_face, direction=direction)
                            if psv and pev:
                                try:
                                    pe = Edge.ByVertices([psv, pev], tolerance=tolerance)
                                    projected_edges.append(pe)
                                except:
                                    continue
        w = Wire.ByEdges(projected_edges, tolerance=tolerance)
        return w

    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created rectangle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Rectangle - Error: specified origin is not a topologic vertex. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Rectangle - Error: Could not find placement in the list of placements. Returning None.")
            return None
        width = abs(width)
        length = abs(length)
        if width <= tolerance or length <= tolerance:
            print("Wire.Rectangle - Error: One or more of the specified dimensions is below the tolerance value. Returning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            print("Wire.Rectangle - Error: The direction vector magnitude is below the tolerance value. Returning None.")
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5

        vb1 = Vertex.ByCoordinates(Vertex.X(origin)-width*0.5+xOffset,Vertex.Y(origin)-length*0.5+yOffset,Vertex.Z(origin))
        vb2 = Vertex.ByCoordinates(Vertex.X(origin)+width*0.5+xOffset,Vertex.Y(origin)-length*0.5+yOffset,Vertex.Z(origin))
        vb3 = Vertex.ByCoordinates(Vertex.X(origin)+width*0.5+xOffset,Vertex.Y(origin)+length*0.5+yOffset,Vertex.Z(origin))
        vb4 = Vertex.ByCoordinates(Vertex.X(origin)-width*0.5+xOffset,Vertex.Y(origin)+length*0.5+yOffset,Vertex.Z(origin))

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True, tolerance=tolerance)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def RemoveCollinearEdges(wire, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Removes any collinear edges in the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        angTolerance : float, optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.
        silent : bool, optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The wire without collinear edges, or the original wire if no modifications were necessary.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def cleanup(wire):
            """Fuses vertices and removes edges below the tolerance distance."""
            vertices = Vertex.Fuse(Topology.Vertices(wire), tolerance=tolerance)
            edges = Topology.Edges(wire)
            new_edges = []

            for edge in edges:
                sv = vertices[Vertex.Index(Edge.StartVertex(edge), vertices, tolerance=tolerance)]
                ev = vertices[Vertex.Index(Edge.EndVertex(edge), vertices, tolerance=tolerance)]
                if Vertex.Distance(sv, ev) > tolerance:
                    new_edges.append(Edge.ByVertices([sv, ev]))

            return Topology.SelfMerge(Cluster.ByTopologies(new_edges, silent=silent), tolerance=tolerance) if new_edges else wire

        def remove_collinear_vertices(wire):
            """Removes collinear vertices from a wire."""
            if not Topology.IsInstance(wire, "Wire"):
                return wire

            vertices = Topology.Vertices(wire)
            filtered_vertices = []

            for i, vertex in enumerate(vertices):
                edges = Topology.SuperTopologies(topology=vertex, hostTopology=wire, topologyType="edge")

                if len(edges) != 2:
                    filtered_vertices.append(vertex)
                elif not Edge.IsCollinear(edges[0], edges[1], tolerance=tolerance):
                    filtered_vertices.append(vertex)

            if len(filtered_vertices) > 2:
                return Wire.ByVertices(filtered_vertices, close=wire.IsClosed(), tolerance=tolerance)
            elif len(filtered_vertices) == 2:
                return Edge.ByStartVertexEndVertex(filtered_vertices[0], filtered_vertices[1], tolerance=tolerance, silent=True)
            else:
                return wire

        # Main function logic
        if Topology.IsInstance(wire, "Cluster"):
            wires = Topology.Wires(wire)
            processed_wires = [Wire.RemoveCollinearEdges(w, angTolerance, tolerance, silent) for w in wires]
            if len(processed_wires) == 0:
                if not silent:
                    print("Wire.RemoveCollinearEdges - Error: No wires were produced. Returning None.")
                return None
            elif len(processed_wires) == 1:
                return Topology.SelfMerge(processed_wires[0])
            else:
                return Topology.SelfMerge(Cluster.ByTopologies(processed_wires, silent=silent))

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print(f"Wire.RemoveCollinearEdges - Error: Input is not a valid wire. Returning None.")
            return None

        new_wire = cleanup(wire)
        wires = Wire.Split(new_wire) if not Wire.IsManifold(new_wire) else [new_wire]

        processed_wires = [remove_collinear_vertices(w) for w in wires]

        if len(processed_wires) == 0:
            return wire
        elif len(processed_wires) == 1:
            return Topology.SelfMerge(processed_wires[0])
        else:
            return Topology.SelfMerge(Cluster.ByTopologies(processed_wires, silent=silent))

    @staticmethod
    def Representation(wire, normalize: bool = True, rotate: bool = True, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a normalized representation of a closed wire with alternating edge lengths and interior angles.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        normalize : bool , optional
            If set to True, the lengths in the list are normalized so that the shortest edge has a length of 1. the default is True.
        rotate : bool , optional
            If set to True, the list is rotated such that the shortest edge appears first.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The representation list.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        import math

        def angleBetweenEdges(e1, e2, tolerance=0.0001):
            a = Vertex.X(Edge.EndVertex(e1)) - Vertex.X(Edge.StartVertex(e1))
            b = Vertex.Y(Edge.EndVertex(e1)) - Vertex.Y(Edge.StartVertex(e1))
            c = Vertex.Z(Edge.EndVertex(e1)) - Vertex.Z(Edge.StartVertex(e1))
            d = Vertex.Distance(Edge.EndVertex(e1), Edge.StartVertex(e2))
            if d <= tolerance:
                d = Vertex.X(Edge.StartVertex(e2)) - Vertex.X(Edge.EndVertex(e2))
                e = Vertex.Y(Edge.StartVertex(e2)) - Vertex.Y(Edge.EndVertex(e2))
                f = Vertex.Z(Edge.StartVertex(e2)) - Vertex.Z(Edge.EndVertex(e2))
            else:
                d = Vertex.X(Edge.EndVertex(e2)) -  Vertex.X(Edge.StartVertex(e2))
                e = Vertex.Y(Edge.EndVertex(e2)) - Vertex.Y(Edge.StartVertex(e2))
                f = Vertex.Z(Edge.EndVertex(e2)) - Vertex.Z(Edge.StartVertex(e2))
            dotProduct = a*d + b*e + c*f
            modOfVector1 = math.sqrt( a*a + b*b + c*c)*math.sqrt(d*d + e*e + f*f) 
            angle = dotProduct/modOfVector1
            angleInDegrees = math.degrees(math.acos(angle))
            return angleInDegrees

        def getInteriorAngles(edges, tolerance=0.0001):
            angles = []
            for i in range(len(edges)-1):
                e1 = edges[i]
                e2 = edges[i+1]
                angles.append(angleBetweenEdges(e1, e2, tolerance=tolerance))
            return angles

        def rotate_list_to_minimum(nums):
            if not nums:
                return nums  # Return the empty list as-is

            min_index = nums.index(min(nums))
            return nums[min_index:] + nums[:min_index]
        
        def getRep(edges, normalize=True, rotate=True, tolerance=0.0001):
            angles = getInteriorAngles(edges, tolerance=tolerance)
            lengths = []
            normalizedLengths = []
            for anEdge in edges:
                lengths.append(Edge.Length(anEdge))
            if normalize == True:
                minLength = min(lengths)
            else:
                minLength = 1
            for aLength in lengths:
                normalizedLengths.append(aLength/minLength)
            if rotate == True:
                return rotate_list_to_minimum([x for x in itertools.chain(*itertools.zip_longest(normalizedLengths, angles)) if x is not None])
            return [x for x in itertools.chain(*itertools.zip_longest(normalizedLengths, angles)) if x is not None]

        edges = Topology.Edges(wire)
        return_list = [round(x, mantissa) for x in getRep(edges, normalize=normalize, rotate=rotate, tolerance=tolerance)]
        return return_list
    
    @staticmethod
    def Reverse(wire, transferDictionaries = False, tolerance: float = 0.0001):
        """
        Creates a wire that has the reverse direction of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        transferDictionaries : bool , optional
            If set to True the dictionaries of the input wire are transferred to the new wire. Othwerwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The reversed wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Reverse - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.Reverse - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        
        original_vertices = Topology.Vertices(wire)
        edges = Topology.Edges(wire)
        if transferDictionaries:
            edge_selectors = []
            for i, e_s in enumerate(edges):
                s = Topology.Centroid(e_s)
                d = Topology.Dictionary(e_s)
                s = Topology.SetDictionary(s, d)
                edge_selectors.append(s)
        vertices = Topology.Vertices(wire)
        vertices.reverse()
        return_wire = Wire.ByVertices(vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        if transferDictionaries:
            return_wire = Topology.TransferDictionariesBySelectors(return_wire, selectors=edge_selectors, tranEdges=True)
            return_wire = Topology.TransferDictionariesBySelectors(return_wire, selectors=original_vertices, tranVertices=True)
        return_wire = Topology.SetDictionary(return_wire, Topology.Dictionary(wire), silent=True)
        return return_wire

    @staticmethod
    def Roof(face, angle: float = 45, boundary: bool = True, tolerance: float = 0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        boundary : bool , optional
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created roof. This method returns the roof as a set of edges. No faces are created.

        """
        from topologicpy import Polyskel
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import topologic_core as topologic
        import math

        def subtrees_to_edges(subtrees, polygon, slope):
            polygon_z = {}
            for x, y, z in polygon:
                polygon_z[(x, y)] = z

            edges = []
            for subtree in subtrees:
                source = subtree.source
                height = subtree.height
                z = slope * height
                source_vertex = Vertex.ByCoordinates(source.X(), source.Y(), z)

                for sink in subtree.sinks:
                    if (sink.X(), sink.Y()) in polygon_z:
                        z = 0
                    else:
                        z = None
                        for st in subtrees:
                            if st.source.X() == sink.X() and st.source.Y() == sink.Y():
                                z = slope * st.height
                                break
                            for sk in st.sinks:
                                if sk.X() == sink.X() and sk.Y() == sink.Y():
                                    z = slope * st.height
                                    break
                        if z is None:
                            height = subtree.height
                            z = slope * height
                    sink_vertex = Vertex.ByCoordinates(sink.X(), sink.Y(), z)
                    if (source.X(), source.Y()) == (sink.X(), sink.Y()):
                        continue
                    e = Edge.ByStartVertexEndVertex(source_vertex, sink_vertex, tolerance=tolerance, silent=True)
                    if e not in edges and e != None:
                        edges.append(e)
            return edges
        
        def face_to_skeleton(face, angle=0, boundary=True):
            normal = Face.Normal(face)
            eb_wire = Face.ExternalBoundary(face)
            ib_wires = Face.InternalBoundaries(face)
            eb_vertices = Topology.Vertices(eb_wire)
            if normal[2] > 0:
                eb_vertices = list(reversed(eb_vertices))
            eb_polygon_coordinates = [(Vertex.X(v), Vertex.Y(v), Vertex.Z(v)) for v in eb_vertices]
            eb_polygonxy = [(x[0], x[1]) for x in eb_polygon_coordinates]

            ib_polygonsxy = []
            zero_coordinates = eb_polygon_coordinates
            for ib_wire in ib_wires:
                ib_vertices = Topology.Vertices(ib_wire)
                if normal[2] > 0:
                    ib_vertices = list(reversed(ib_vertices))
                ib_polygon_coordinates = [(Vertex.X(v), Vertex.Y(v), Vertex.Z(v)) for v in ib_vertices]
                ib_polygonxy = [(x[0], x[1]) for x in ib_polygon_coordinates]
                ib_polygonsxy.append(ib_polygonxy)
                zero_coordinates += ib_polygon_coordinates
            skeleton = Polyskel.skeletonize(eb_polygonxy, ib_polygonsxy)
            if len(skeleton) == 0:
                print("Wire.Roof - Error: The Polyskel.skeletonize 3rd party software failed to create a skeleton. Returning None.")
                return None
            slope = math.tan(math.radians(angle))
            roofEdges = subtrees_to_edges(skeleton, zero_coordinates, slope)
            if boundary == True:
                roofEdges = Helper.Flatten(roofEdges)+Topology.Edges(face)
            else:
                roofEdges = Helper.Flatten(roofEdges)
            roofTopology = Topology.SelfMerge(Cluster.ByTopologies(roofEdges), tolerance=tolerance)
            return roofTopology
        
        if not Topology.IsInstance(face, "Face"):
            return None
        angle = abs(angle)
        if angle >= 90-tolerance:
            return None
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        d = Topology.Dictionary(flat_face)
        roof = face_to_skeleton(flat_face, angle=angle, boundary=boundary)
        if not roof:
            return None
        roof = Topology.Unflatten(roof, origin=origin, direction=normal)
        return roof
    
    @staticmethod
    def Simplify(wire, method='douglas-peucker', tolerance=0.0001, silent=False):
        """
        Simplifies the input wire edges based on the selected algorithm: Douglas-Peucker or Visvalingam–Whyatt.
        
        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        method : str, optional
            The simplification method to use: 'douglas-peucker' or 'visvalingam-whyatt' or 'reumann-witkam'.
            The default is 'douglas-peucker'.
        tolerance : float , optional
            The desired tolerance.
            If using the douglas-peucker method, edge lengths shorter than this amount will be removed.
            If using the visvalingam-whyatt method, triangulare areas less than is amount will be removed.
            If using the Reumann-Witkam method, the tolerance specifies the maximum perpendicular distance allowed
            between any point and the current line segment; points falling within this distance are discarded.
            The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
            
        Returns
        -------
        topologic_core.Wire
            The simplified wire.
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def perpendicular_distance(point, line_start, line_end):
            # Calculate the perpendicular distance from a point to a line segment
            x0 = Vertex.X(point)
            y0 = Vertex.Y(point)
            x1 = Vertex.X(line_start)
            y1 = Vertex.Y(line_start)
            x2 = Vertex.X(line_end)
            y2 = Vertex.Y(line_end)

            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = Vertex.Distance(line_start, line_end)

            return numerator / denominator

        def douglas_peucker(wire, tolerance=0.0001):
            if isinstance(wire, list):
                points = wire
            else:
                points = Topology.Vertices(wire)
            if len(points) <= 2:
                return points

            start_point = points[0]
            end_point = points[-1]

            max_distance = 0
            max_index = 0

            for i in range(1, len(points) - 1):
                d = perpendicular_distance(points[i], start_point, end_point)
                if d > max_distance:
                    max_distance = d
                    max_index = i

            if max_distance <= tolerance:
                return [start_point, end_point]

            first_segment = douglas_peucker(points[:max_index + 1], tolerance=tolerance)
            second_segment = douglas_peucker(points[max_index:], tolerance=tolerance)

            return first_segment[:-1] + second_segment

        def visvalingam_whyatt(wire, tolerance=0.0001):
            if isinstance(wire, list):
                points = wire
            else:
                points = Topology.Vertices(wire)

            if len(points) <= 2:
                return points

            # Calculate the effective area for each point except the first and last
            def effective_area(p1, p2, p3):
                # Triangle area formed by p1, p2, and p3
                return 0.5 * abs(Vertex.X(p1) * (Vertex.Y(p2) - Vertex.Y(p3)) + Vertex.X(p2) * (Vertex.Y(p3) - Vertex.Y(p1)) + Vertex.X(p3) * (Vertex.Y(p1) - Vertex.Y(p2)))

            # Keep track of effective areas
            areas = [None]  # First point has no area
            for i in range(1, len(points) - 1):
                area = effective_area(points[i - 1], points[i], points[i + 1])
                areas.append((area, i))
            areas.append(None)  # Last point has no area

            # Sort points by area in ascending order
            sorted_areas = sorted([(area, idx) for area, idx in areas[1:-1] if area is not None])

            # Remove points with area below the tolerance threshold
            remove_indices = {idx for area, idx in sorted_areas if area <= tolerance}

            # Construct the simplified list of points
            simplified_points = [point for i, point in enumerate(points) if i not in remove_indices]

            return simplified_points

        def reumann_witkam(wire, tolerance=0.0001):
            if isinstance(wire, list):
                points = wire
            else:
                points = Topology.Vertices(wire)
            
            if len(points) <= 2:
                return points

            simplified_points = [points[0]]
            start_point = points[0]
            i = 1

            while i < len(points) - 1:
                end_point = points[i]
                next_point = points[i + 1]
                dist = perpendicular_distance(next_point, start_point, end_point)

                # If the next point is outside the tolerance corridor, add the current end_point
                if dist > tolerance:
                    simplified_points.append(end_point)
                    start_point = end_point

                i += 1

            # Always add the last point
            simplified_points.append(points[-1])

            return simplified_points

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Simplify = Error: The input wire parameter is not a Wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            wires = Wire.Split(wire)
            new_wires = []
            for w in wires:
                if Topology.IsInstance(w, "Edge"):
                    if Edge.Length(w) > tolerance:
                        new_wires.append(w)
                elif Topology.IsInstance(w, "Wire"):
                    new_wires.append(Wire.Simplify(w, method=method, tolerance=tolerance, silent=silent))
            return_wire = Topology.SelfMerge(Cluster.ByTopologies(new_wires))
            return return_wire

        new_vertices = []
        if 'douglas' in method.lower(): #douglas-peucker
            new_vertices = douglas_peucker(wire, tolerance=tolerance)
        elif 'vis' in method.lower(): # 'visvalingam-whyatt'
            new_vertices = visvalingam_whyatt(wire, tolerance=tolerance)
        elif 'reu' in method.lower(): # 'reumann-witkam'
            new_vertices = reumann_witkam(wire, tolerance=tolerance)
        else:
            if not silent:
                print(f"Wire.Simplify - Warning: Unknown method ({method}). Please use 'douglas-peucker' or 'visvalingam-whyatt' or 'reumann-witkam'. Defaulting to 'douglas-peucker'.")
            new_vertices = douglas_peucker(wire, tolerance=tolerance)
        
        if len(new_vertices) < 2:
            if not silent:
                print("Wire.Simplify - Warning: Could not generate enough vertices for a simplified wire. Returning the original wire.")
            wire
        new_wire = Wire.ByVertices(new_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        if not Topology.IsInstance(new_wire, "wire"):
            if not silent:
                print("Wire.Simplify - Warning: Could not generate a simplified wire. Returning the original wire.")
            return wire
        return new_wire

    @staticmethod
    def Skeleton(face, boundary: bool = True, tolerance: float = 0.001):
        """
        Creates a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
        This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        boundary : bool , optional
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created straight skeleton.

        """
        if not Topology.IsInstance(face, "Face"):
            return None
        return Wire.Roof(face, angle=0, boundary=boundary, tolerance=tolerance)
    
    @staticmethod
    def Spiral(origin = None, radiusA : float = 0.05, radiusB : float = 0.5, height : float = 1, turns : int = 10, sides : int = 36, clockwise : bool = False, reverse : bool = False, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a spiral.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the spiral. The default is None which results in the spiral being placed at (0, 0, 0).
        radiusA : float , optional
            The initial radius of the spiral. The default is 0.05.
        radiusB : float , optional
            The final radius of the spiral. The default is 0.5.
        height : float , optional
            The height of the spiral. The default is 1.
        turns : int , optional
            The number of turns of the spiral. The default is 10.
        sides : int , optional
            The number of sides of one full turn in the spiral. The default is 36.
        clockwise : bool , optional
            If set to True, the spiral will be oriented in a clockwise fashion. Otherwise, it will be oriented in an anti-clockwise fashion. The default is False.
        reverse : bool , optional
            If set to True, the spiral will increase in height from the center to the circumference. Otherwise, it will increase in height from the conference to the center. The default is False.
        direction : list , optional
            The vector representing the up direction of the spiral. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the spiral. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        topologic_core.Wire
            The created spiral.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Spiral - Error: the input origin is not a valid topologic Vertex. Returning None.")
            return None
        if radiusA <= 0:
            print("Wire.Spiral - Error: the input radiusA cannot be less than or equal to zero. Returning None.")
            return None
        if radiusB <= 0:
            print("Wire.Spiral - Error: the input radiusB cannot be less than or equal to zero. Returning None.")
            return None
        if radiusA == radiusB:
            print("Wire.Spiral - Error: the inputs radiusA and radiusB cannot be equal. Returning None.")
            return None
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if turns <= 0:
            print("Wire.Spiral - Error: the input turns cannot be less than or equal to zero. Returning None.")
            return None
        if sides < 3:
            print("Wire.Spiral - Error: the input sides cannot be less than three. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Spiral - Error: the input placement string is not one of center, lowerleft, upperleft, lowerright, or upperright. Returning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            print("Wire.Spiral - Error: the input direction vector is not a valid direction. Returning None.")
            return None
        
        vertices = []
        xList = []
        yList = []
        zList = []
        if clockwise:
            cw = -1
        else:
            cw = 1
        n_vertices = sides*turns + 1
        zOffset = height/float(n_vertices)
        if reverse == True:
            z = height
        else:
            z = 0
        ang = 0
        angOffset = float(360/float(sides))
        b = (radiusB - radiusA)/(2*math.pi*turns)
        while ang <= 360*turns:
            rad = math.radians(ang)
            x = (radiusA + b*rad)*math.cos(rad)*cw
            xList.append(x)
            y = (radiusA + b*rad)*math.sin(rad)
            yList.append(y)
            zList.append(z)
            if reverse == True:
                z = z - zOffset
            else:
                z = z + zOffset
            vertices.append(Vertex.ByCoordinates(x, y, z))
            ang = ang + angOffset
        
        x_min = min(xList)
        maxX = max(xList)
        y_min = min(yList)
        maxY = max(yList)
        radius = radiusA + radiusB*turns*0.5
        baseWire = Wire.ByVertices(vertices, close=False, tolerance=tolerance)
        if placement.lower() == "center":
            baseWire = Topology.Translate(baseWire, 0, 0, -height*0.5)
        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, -x_min, -y_min, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, -x_min, -maxY, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -maxX, -y_min, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -maxX, -maxY, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def Split(wire) -> list:
        """
        Splits the input wire into segments at its intersections (i.e. at any vertex where more than two edges meet).

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of split wire segments.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def vertexDegree(v, wire):
            edges = []
            _ = v.Edges(wire, edges)
            return len(edges)
        
        def vertexOtherEdge(vertex, edge, wire):
            edges = []
            _ = vertex.Edges(wire, edges)
            if Topology.IsSame(edges[0], edge):
                return edges[-1]
            else:
                return edges[0]
        
        def edgeOtherVertex(edge, vertex):
            vertices = Topology.Vertices(edge)
            if Topology.IsSame(vertex, vertices[0]):
                return vertices[-1]
            else:
                return vertices[0]
        
        def edgeInList(edge, edgeList):
            for anEdge in edgeList:
                if Topology.IsSame(anEdge, edge):
                    return True
            return False
        
        vertices = Topology.Vertices(wire)
        hubs = []
        for aVertex in vertices:
            if vertexDegree(aVertex, wire) > 2:
                hubs.append(aVertex)
        wires = []
        global_edges = []
        for aVertex in hubs:
            hub_edges = []
            _ = aVertex.Edges(wire, hub_edges)
            wire_edges = []
            for hub_edge in hub_edges:
                if not edgeInList(hub_edge, global_edges):
                    current_edge = hub_edge
                    oe = edgeOtherVertex(current_edge, aVertex)
                    while vertexDegree(oe, wire) == 2:
                        if not edgeInList(current_edge, global_edges):
                            global_edges.append(current_edge)
                            wire_edges.append(current_edge)
                        current_edge = vertexOtherEdge(oe, current_edge, wire)
                        oe = edgeOtherVertex(current_edge, oe)
                    if not edgeInList(current_edge, global_edges):
                        global_edges.append(current_edge)
                        wire_edges.append(current_edge)
                    if len(wire_edges) > 1:
                        wires.append(Cluster.ByTopologies(wire_edges).SelfMerge())
                    else:
                        wires.append(wire_edges[0])
                    wire_edges = []
        if len(wires) < 1:
            return [wire]
        return wires
    
    @staticmethod
    def Square(origin= None, size: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. The default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created square.

        """
        return Wire.Rectangle(origin=origin, width=size, length=size, direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Squircle(origin = None, radius: float = 0.5, sides: int = 121, a: float = 2.0, b: float = 2.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a Squircle which is a hybrid between a circle and a square. See https://en.wikipedia.org/wiki/Squircle

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the squircle. The default is None which results in the squircle being placed at (0, 0, 0).
        radius : float , optional
            The desired radius of the squircle. The default is 0.5.
        sides : int , optional
            The desired number of sides of the squircle. The default is 121.
        a : float , optional
            The "a" factor affects the x position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        b : float , optional
            The "b" factor affects the y position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created squircle.
        
        """
        def get_squircle(a=1, b=1, radius=0.5, sides=100):
            import numpy as np
            t = np.linspace(0, 2*np.pi, sides)
            x = (np.abs(np.cos(t))**(1/a)) * np.sign(np.cos(t))
            y = (np.abs(np.sin(t))**(1/b)) * np.sign(np.sin(t))
            return x*radius, y*radius
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Squircle - Error: The input origin parameter is not a valid Vertex. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Squircle - Error: The input placement parameter is not a recognized string. Returning None.")
            return None
        radius = abs(radius)
        if radius <= tolerance:
            return None
        
        if a <= 0:
            print("Wire.Squircle - Error: The a input parameter must be a positive number. Returning None.")
            return None
        if b <= 0:
            print("Wire.Squircle - Error: The b input parameter must be a positive number. Returning None.")
            return None
        if a == 1 and b == 1:
            return Wire.Circle(radius=radius, sides=sides, direction=direction, placement=placement, tolerance=tolerance)
        x_list, y_list = get_squircle(a=a, b=b, radius=radius, sides=sides)
        vertices = []
        for i, x in enumerate(x_list):
            v = Vertex.ByCoordinates(x, y_list[i], 0)
            vertices.append(v)
        baseWire = Wire.ByVertices(vertices, close=True, tolerance=tolerance)
        baseWire = Topology.RemoveCollinearEdges(baseWire, angTolerance=angTolerance, tolerance=tolerance)
        baseWire = Wire.Simplify(baseWire, tolerance=tolerance)
        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, radius, radius, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -radius, radius, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def Star(origin= None, radiusA: float = 0.5, radiusB: float = 0.2, rays: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the star. The default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 8.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created star.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusA <= tolerance or radiusB <= tolerance:
            return None
        rays = abs(rays)
        if rays < 3:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        sides = rays*2 # Sides is double the number of rays
        baseV = []

        xList = []
        yList = []
        for i in range(sides):
            if i%2 == 0:
                radius = radiusA
            else:
                radius = radiusB
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + Vertex.X(origin)
            y = math.cos(angle)*radius + Vertex.Y(origin)
            z = Vertex.Z(origin)
            xList.append(x)
            yList.append(y)
            baseV.append([x, y])

        if placement.lower() == "lowerleft":
            xmin = min(xList)
            ymin = min(yList)
            xOffset = Vertex.X(origin) - xmin
            yOffset = Vertex.Y(origin) - ymin
        elif placement.lower() == "upperleft":
            xmin = min(xList)
            ymax = max(yList)
            xOffset = Vertex.X(origin) - xmin
            yOffset = Vertex.Y(origin) - ymax
        elif placement.lower() == "lowerright":
            xmax = max(xList)
            ymin = min(yList)
            xOffset = Vertex.X(origin) - xmax
            yOffset = Vertex.Y(origin) - ymin
        elif placement.lower() == "upperright":
            xmax = max(xList)
            ymax = max(yList)
            xOffset = Vertex.X(origin) - xmax
            yOffset = Vertex.Y(origin) - ymax
        else:
            xOffset = 0
            yOffset = 0
        tranBase = []
        for coord in baseV:
            tranBase.append(Vertex.ByCoordinates(coord[0]+xOffset, coord[1]+yOffset, Vertex.Z(origin)))
        
        baseWire = Wire.ByVertices(tranBase, close=True, tolerance=tolerance)
        baseWire = Wire.Reverse(baseWire)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def StartEndVertices(wire) -> list:
        """
        Returns the start and end vertices of the input wire. The wire must be manifold and open.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Wire.IsManifold(wire):
            print("Wire.StartEndVertices - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        vertices = Topology.Vertices(wire)
        if Wire.IsClosed(wire):
            return [vertices[0], vertices[0]] # If the wire is closed, the start and end vertices are the same vertex
        endPoints = [v for v in vertices if (Vertex.Degree(v, wire) == 1)]
        if len(endPoints) < 2:
            print("Wire.StartEndVertices - Error: Could not find the end vertices if the input wire parameter. Returning None.")
            return None
        edge1 = Topology.SuperTopologies(endPoints[0], wire, topologyType="edge")[0]
        sv = Edge.StartVertex(edge1)
        if (Topology.IsSame(endPoints[0], sv)):
            wireStartVertex = endPoints[0]
            wireEndVertex = endPoints[1]
        else:
            wireStartVertex = endPoints[1]
            wireEndVertex = endPoints[0]
        return [wireStartVertex, wireEndVertex]
    
    @staticmethod
    def StartVertex(wire):
        """
        Returns the start vertex of the input wire. The wire must be manifold and open.

        """
        sv, ev = Wire.StartEndVertices(wire)
        return sv
    
    @staticmethod
    def Trapezoid(origin= None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0, 0, 0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. The default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. The default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. The default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. The default is 0.0.
        length : float , optional
            The length of the trapezoid. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created trapezoid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        widthA = abs(widthA)
        widthB = abs(widthB)
        length = abs(length)
        if widthA <= tolerance or widthB <= tolerance or length <= tolerance:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "center":
            xOffset = -((-widthA*0.5 + offsetA) + (-widthB*0.5 + offsetB) + (widthA*0.5 + offsetA) + (widthB*0.5 + offsetB))/4.0
            yOffset = 0
        elif placement.lower() == "lowerleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = -length*0.5

        vb1 = Vertex.ByCoordinates(Vertex.X(origin)-widthA*0.5+offsetA+xOffset,Vertex.Y(origin)-length*0.5+yOffset,Vertex.Z(origin))
        vb2 = Vertex.ByCoordinates(Vertex.X(origin)+widthA*0.5+offsetA+xOffset,Vertex.Y(origin)-length*0.5+yOffset,Vertex.Z(origin))
        vb3 = Vertex.ByCoordinates(Vertex.X(origin)+widthB*0.5+offsetB+xOffset,Vertex.Y(origin)+length*0.5+yOffset,Vertex.Z(origin))
        vb4 = Vertex.ByCoordinates(Vertex.X(origin)-widthB*0.5++offsetB+xOffset,Vertex.Y(origin)+length*0.5+yOffset,Vertex.Z(origin))

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True, tolerance=tolerance)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def TShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a T-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. The default is None which results in the T-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the T-shape. The default is 1.0.
        length : float , optional
            The overall length of the T-shape. The default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the T-shape. The default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the T-shape. The default is 0.25.
        direction : list , optional
            The vector representing the up direction of the T-shape. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the T-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Wire
            The created T-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Wire.LShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Wire.LShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Wire.LShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Wire.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Wire.LShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance*2):
            if not silent:
                print("Wire.LShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance*2):
            if not silent:
                print("Wire.LShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.LShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.LShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.LShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        # Define the vertices of the T-shape (counterclockwise)
        v1 = Vertex.ByCoordinates(width/2-a/2, 0)
        v2 = Vertex.ByCoordinates(width/2+a/2, 0)
        v3 = Vertex.ByCoordinates(width/2+a/2, length-b)
        v4 = Vertex.ByCoordinates(width, length-b)
        v5 = Vertex.ByCoordinates(width, length)
        v6 = Vertex.ByCoordinates(0, length)
        v7 = Vertex.ByCoordinates(0, length-b)
        v8 = Vertex.ByCoordinates(width/2-a/2, length-b)  # Top of vertical arm

        # Create the T-shaped wire
        t_shape = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8], close=True, tolerance=tolerance)
        t_shape = Topology.Translate(t_shape, -width/2, -length/2, 0)
        t_shape = Topology.Translate(t_shape, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))
        reverse = False
        if flipHorizontal == True:
            xScale = -1
            reverse = not reverse
        else:
            xScale = 1
        if flipVertical == True:
            yScale = -1
            reverse = not reverse
        else:
            yScale = 1
        if xScale == -1 or yScale == -1:
            t_shape = Topology.Scale(t_shape, origin=origin, x=xScale, y=yScale, z=1)
            if reverse == True:
                t_shape = Wire.Reverse(t_shape)
        if placement.lower() == "lowerleft":
            t_shape = Topology.Translate(t_shape, width/2, length/2, 0)
        elif placement.lower() == "upperright":
            t_shape = Topology.Translate(t_shape, -width/2, -length/2, 0)
        elif placement.lower() == "upperleft":
            t_shape = Topology.Translate(t_shape, width/2, -length/2, 0)
        elif placement.lower() == "lowerright":
            t_shape = Topology.Translate(t_shape, -width/2, length/2, 0)
        
        if direction != [0, 0, 1]:
            t_shape = Topology.Orient(t_shape, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return t_shape

    @staticmethod
    def VertexDistance(wire, vertex, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the distance, computed along the input wire of the input vertex from the input origin vertex.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            The input vertex
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input wire. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        float
            The distance of the input vertex from the input origin along the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Wire.VertexDistance - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        wire_length = Wire.Length(wire)
        if wire_length <= tolerance:
            print("Wire.VertexDistance: The input wire parameter is a degenerate topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.VertexDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(vertex, wire, tolerance=tolerance):
            print("Wire.VertexDistance: The input vertex parameter is not internal to the input wire parameter. Returning None.")
            return None
        
        def distance_from_start(wire, v):
            total_distance = 0.0
            found = False
            # Iterate over the edges of the wire
            for edge in Wire.Edges(wire):
                if Vertex.IsInternal(v, edge, tolerance=tolerance):
                    total_distance += Vertex.Distance(Edge.StartVertex(edge), v)
                    found = True
                    break
                total_distance += Edge.Length(edge)
            if found == False:
                return None
            return total_distance
        
        d1 = distance_from_start(wire, vertex)
        d2 = distance_from_start(wire, origin)
        if d1 == None:
            print("Wire.VertexDistance - Error: The input vertex parameter is not internal to the input wire parameter. Returning None.")
            return None
        if d2 == None:
            print("Wire.VertexDistance - Error: The input origin parameter is not internal to the input wire parameter. Returning None.")
            return None
        return round(abs(d2-d1), mantissa)

    @staticmethod
    def VertexByDistance(wire, distance: float = 0.0, origin= None, tolerance = 0.0001):
        """
        Creates a vertex along the input wire offset by the input distance from the input origin.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        distance : float , optional
            The offset distance. The default is 0.
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input edge. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Vertex import Vertex

        def compute_u(u):
            def count_decimal_places(number):
                try:
                    # Convert the number to a string to analyze decimal places
                    num_str = str(number)
                    # Split the number into integer and decimal parts
                    integer_part, decimal_part = num_str.split('.')
                    # Return the length of the decimal part
                    return len(decimal_part)
                except ValueError:
                    # If there's no decimal part, return 0
                    return 0
            dp = count_decimal_places(u)
            u = -(int(u) - u)
            return round(u,dp)

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexByDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        wire_length = Wire.Length(wire)
        if wire_length <= tolerance:
            print("Wire.VertexByDistance: The input wire parameter is a degenerate topologic wire. Returning None.")
            return None
        if abs(distance) <= tolerance:
            return Wire.StartVertex(wire)
        if abs(distance - wire_length) <= tolerance:
            return Wire.EndVertex(wire)
        if not Wire.IsManifold(wire):
            print("Wire.VertexAtParameter - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.VertexByDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(origin, wire, tolerance=tolerance):
            print("Wire.VertexByDistance - Error: The input origin parameter is not internal to the input wire parameter. Returning None.")
            return None
        if Vertex.Distance(Wire.StartVertex(wire), origin) <= tolerance:
            u = distance/wire_length
        elif Vertex.Distance(Wire.EndVertex(wire), origin) <= tolerance:
            u = 1 - distance/wire_length
        else:
            d = Wire.VertexDistance(wire, origin) + distance
            u = d/wire_length

        return Wire.VertexByParameter(wire, u=compute_u(u))
    
    @staticmethod
    def VertexByParameter(wire, u: float = 0):
        """
        Creates a vertex along the input wire offset by the input *u* parameter. The wire must be manifold.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        u : float , optional
            The *u* parameter along the input topologic Wire. A parameter of 0 returns the start vertex. A parameter of 1 returns the end vertex. The default is 0.

        Returns
        -------
        topologic_core.Vertex
            The vertex at the input u parameter

        """
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexByParameter - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if u < 0 or u > 1:
            print("Wire.VertexByParameter - Error: The input u parameter is not within the valid range of [0, 1]. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.VertexByParameter - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        
        if u == 0:
            return Wire.StartVertex(wire)
        if u == 1:
            return Wire.EndVertex(wire)
        
        edges = Wire.Edges(wire)
        total_length = 0.0
        edge_lengths = []
        
        # Compute the total length of the wire
        for edge in edges:
            e_length = Edge.Length(edge)
            edge_lengths.append(e_length)
            total_length += e_length

        # Initialize variables for tracking the current edge and accumulated length
        current_edge = None
        accumulated_length = 0.0

        # Iterate over the lines to find the appropriate segment
        for i, edge in enumerate(edges):
            edge_length = edge_lengths[i]

            # Check if the desired point is on this line
            if u * total_length <= accumulated_length + edge_length:
                current_edge = edge
                break
            else:
                accumulated_length += edge_length

        # Calculate the residual u value for the current line
        residual_u = (u * total_length - accumulated_length) / Edge.Length(current_edge)

        # Compute the point at the parameter on the current line
        vertex = Edge.VertexByParameter(current_edge, residual_u)

        return vertex

    @staticmethod
    def Vertices(wire) -> list:
        """
        Returns the list of vertices of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of vertices.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None
        vertices = []
        _ = wire.Vertices(None, vertices) # Hook to Core
        return vertices

