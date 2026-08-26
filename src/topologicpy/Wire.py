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

from binascii import a2b_base64
from re import A
from topologicpy.Core import Core
from topologicpy.Topology import Topology
import math
import itertools

class Wire():

    @staticmethod
    def _UseNativeWireBackend() -> bool:
        """
        Returns True when the active backend exposes the enhanced native Wire utilities.

        This private capability check deliberately excludes TopologicCore so that
        legacy TopologicCore code paths retain their established behavior.
        """
        try:
            if Topology._IsTopologicCoreBackend():
                return False
        except Exception:
            return False
        try:
            return bool(Core.HasAttribute("WireUtility", "PointAtParameter"))
        except Exception:
            return False

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
            The number of sides of the arc. Default is 16.
        close : bool , optional
            If set to True, the arc will be closed by connecting the last vertex to the first vertex. Otherwise, it will be left open.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        arc = Wire.ByVertices(vertices, close=close, tolerance=tolerance, silent=True) #We want to force suppress errors and warnings here.
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
            The length of the sagitta. In mathematics, the sagitta is the line connecting the center of a chord to the apex (or highest point) of the arc subtended by that chord. Default is 1.
        absolute : bool , optional
            If set to True, the sagitta length is treated as an absolute value. Otherwise, it is treated as a ratio based on the length of the edge.
            For example, if the length of the edge is 10, the sagitta is set to 0.5, and absolute is set to False, the sagitta length will be 5. Default is True.
        sides : int , optional
            The number of sides of the arc. Default is 16.
        close : bool , optional
            If set to True, the arc will be closed by connecting the last vertex to the first vertex. Otherwise, it will be left open.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        return Wire.Arc(sv, cv, ev, sides=sides, close=close, tolerance=tolerance, silent=True) # we want to force suppress errors and warnings here



    @staticmethod
    def Bisectors(wire, offset: float = 1.0, offsetKey: str = "offset", stepOffsetA: float = 0, stepOffsetB: float = 0, stepOffsetKeyA: str = "stepOffsetA", stepOffsetKeyB: str = "stepOffsetB", reverse: bool = False, transferDictionaries: bool = False, epsilon: float = 0.01, tolerance: float = 0.0001,  silent: bool = False, numWorkers: int = None):
        """
        Returns opnly the bisectors Created by an offset wire from the input wire. See Wire.ByOffset. A positive offset value results in an offset to the interior of an anti-clockwise wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float , optional
            The desired offset distance. Default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. Default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. Default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. Default is "stepOffsetB".
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. Default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the new wire. Otherwise, they are not. Default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). Default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
        
        if reverse == True:
            fac = -1
        else:
            fac = 1
        bisectors = True
        origin = Topology.Centroid(wire)
        temp_vertices = [Topology.Vertices(wire)[0], Topology.Vertices(wire)[1], Topology.Centroid(wire)]
        temp_face = Face.ByWire(Wire.ByVertices(temp_vertices, close=True, tolerance=tolerance), silent=silent)
        normal = Face.Normal(temp_face)
        flat_wire = Topology.Flatten(wire, direction=normal, origin=origin)
        original_edges = Topology.Edges(wire)
        edges = Topology.Edges(flat_wire)
        offsets = []
        offset_edges = []
        final_vertices = []
        bisectors_list = []
        edge_dictionaries = []
        for i, edge in enumerate(edges):
            d = Topology.Dictionary(original_edges[i])
            d_offset = Dictionary.ValueAtKey(d, key=offsetKey, defaultValue=offset)
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
        bisectors_cluster = Cluster.ByTopologies(bisectors_list)
        return Topology.Unflatten(bisectors_cluster, direction=normal, origin=origin)

        # return_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance, silent=silent)
        # wire_edges = [Edge.SetLength(w_e, Edge.Length(w_e)+(2*epsilon), bothSides=True) for w_e in Topology.Edges(return_wire)]
        # return_wire_edges = Topology.Edges(return_wire)
        # if transferDictionaries == True:
        #     if not len(wire_edges) == len(edge_dictionaries):
        #         if not silent:
        #                 print("Length of Wire Edges:", len(wire_edges))
        #                 print("Length of Edge Dictionaries:", len(edge_dictionaries))
        #                 print("Wire.ByOffset - Warning: The resulting wire is not well-formed, offsets may not be applied correctly. Please check your offsets.")
        #     for i, wire_edge in enumerate(wire_edges):
        #         if len(edge_dictionaries) > 0:
        #             temp_dictionary = edge_dictionaries[min(i,len(edge_dictionaries)-1)]
        #             wire_edge = Topology.SetDictionary(wire_edge, temp_dictionary, silent=True)
        #             return_wire_edges[i] = Topology.SetDictionary(return_wire_edges[i], temp_dictionary, silent=True)
        # if bisectors == True:
        #     temp_return_wire = Topology.SelfMerge(Cluster.ByTopologies(wire_edges+bisectors_list))
        #     if transferDictionaries == True:
        #         sel_vertices = Topology.Vertices(return_wire)
        #         sel_vertices += Topology.Vertices(flat_wire)
        #         edges = Topology.Edges(return_wire)
        #         sel_edges = []
        #         for edge in edges:
        #             d = Topology.Dictionary(edge)
        #             c = Topology.Centroid(edge)
        #             c = Topology.SetDictionary(c, d, silent=True)
        #             sel_edges.append(c)
        #         temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_vertices, tranVertices=True, numWorkers=numWorkers)
        #         temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_edges, tranEdges=True, numWorkers=numWorkers)
                
        #     return_wire = temp_return_wire
        
        # if not Topology.IsInstance(return_wire, "Wire"):
        #     if not silent:
        #         print("Wire.ByOffset - Warning: The resulting wire is not well-formed, please check your offsets.")
        # else:
        #     if not Wire.IsManifold(return_wire) and bisectors == False:
        #         if not silent:
        #             print("Wire.ByOffset - Warning: The resulting wire is non-manifold, please check your offsets.")
        #             print("Wire.ByOffset - Warning: Pursuing a workaround, but it might take longer to complete.")
                
        #         temp_wire = Topology.SelfMerge(Cluster.ByTopologies(wire_edges))
        #         cycles = Wire.Cycles(temp_wire, maxVertices = len(final_vertices))
        #         if len(cycles) > 0:
        #             distances = []
        #             for cycle in cycles:
        #                 cycle_centroid = Topology.Centroid(cycle)
        #                 distance = Vertex.Distance(origin, cycle_centroid)
        #                 distances.append(distance)
        #             cycles = Helper.Sort(cycles, distances)
        #             # Get the top three or less
        #             cycles = cycles[:min(3, len(cycles))]
        #             areas = [Face.Area(Face.ByWire(cycle)) for cycle in cycles]
        #             cycles = Helper.Sort(cycles, areas)
        #             return_cycle = Wire.Reverse(cycles[-1])
        #             test_cycle = Wire.Simplify(return_cycle, tolerance=epsilon)
        #             if Topology.IsInstance(test_cycle, "Wire"):
        #                 return_cycle = test_cycle
        #             return_cycle = Wire.RemoveCollinearEdges(return_cycle, silent=silent)
        #             sel_edges = []
        #             for temp_edge in wire_edges:
        #                 x = Topology.Centroid(temp_edge)
        #                 d = Topology.Dictionary(temp_edge)
        #                 x = Topology.SetDictionary(x, d, silent=True)
        #                 sel_edges.append(x)
        #             return_cycle = Topology.TransferDictionariesBySelectors(return_cycle, Topology.Vertices(return_wire), tranVertices=True, tolerance=tolerance, numWorkers=numWorkers)
        #             return_cycle = Topology.TransferDictionariesBySelectors(return_cycle, sel_edges, tranEdges=True, tolerance=tolerance, numWorkers=numWorkers)
        #             return_wire = return_cycle
        # return_wire = Topology.Unflatten(return_wire, direction=normal, origin=origin)
        # if transferDictionaries == True:
        #     return_wire = Topology.SetDictionary(return_wire, Topology.Dictionary(wire), silent=True)
        # return return_wire
    
    @staticmethod
    def BoundingRectangle(topology, optimize: int = 0, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a wire representing a bounding rectangle of the input topology.
        The returned wire contains a dictionary with key "zrot" that represents
        rotations around the Z axis. If applied, the resulting wire will become
        axis-aligned.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization),
            the method will attempt to optimize the bounding rectangle so that it
            reduces its surface area.
            The minimum optimization number of 0 will result in an axis-aligned
            bounding rectangle.
            A maximum optimization number of 10 will attempt to reduce the bounding
            rectangle's area by 50%. Default is 0.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The bounding rectangle of the input topology.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Wire import Wire
        import math

        def round_xyz(v):
            x, y, z = Vertex.Coordinates(v)
            return (
                round(float(x), mantissa),
                round(float(y), mantissa),
                round(float(z), mantissa)
            )

        def deterministic_vertices(vertices):
            return sorted(vertices, key=lambda v: round_xyz(v))

        def vector(a, b):
            ax, ay, az = round_xyz(a)
            bx, by, bz = round_xyz(b)
            return (bx - ax, by - ay, bz - az)

        def cross(u, v):
            return (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]
            )

        def magnitude(v):
            return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        def are_three_collinear(v1, v2, v3, tol=tolerance):
            u = vector(v1, v2)
            v = vector(v1, v3)
            c = cross(u, v)
            return magnitude(c) <= tol

        def all_vertices_collinear(vertices, tol=tolerance):
            n = len(vertices)
            if n < 3:
                return True

            a = vertices[0]
            b = None

            for i in range(1, n):
                if magnitude(vector(a, vertices[i])) > tol:
                    b = vertices[i]
                    break

            if b is None:
                return True

            for i in range(n):
                vi = vertices[i]
                if vi == a or vi == b:
                    continue
                if not are_three_collinear(a, b, vi, tol=tol):
                    return False

            return True

        def first_non_collinear_triplet(vertices, tol=tolerance):
            n = len(vertices)

            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if not are_three_collinear(vertices[i], vertices[j], vertices[k], tol=tol):
                            return [vertices[i], vertices[j], vertices[k]]

            return None

        def triplet_normal(vertices3, tol=tolerance):
            v1, v2, v3 = vertices3
            u = vector(v1, v2)
            v = vector(v1, v3)
            n = cross(u, v)
            mag = magnitude(n)

            if mag <= tol:
                return None

            return [n[0] / mag, n[1] / mag, n[2] / mag]

        def br(tp):
            verts = Topology.Vertices(tp)
            if not verts:
                return None

            xs = [round(Vertex.X(v), mantissa) for v in verts]
            ys = [round(Vertex.Y(v), mantissa) for v in verts]

            return [min(xs), min(ys), max(xs), max(ys)]

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Wire.BoundingRectangle - Error: The input topology parameter is not a valid topology. Returning None.")
            return None

        vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
        if not isinstance(vertices, list) or len(vertices) < 3:
            if not silent:
                print("Wire.BoundingRectangle - Error: The input topology parameter does not contain enough vertices to create a bounding rectangle. Returning None.")
            return None

        vertices = deterministic_vertices(vertices)

        if all_vertices_collinear(vertices, tol=tolerance):
            if not silent:
                print("Wire.BoundingRectangle - Error: All vertices of the input topology parameter are collinear and thus no bounding rectangle can be created. Returning None.")
            return None

        vList = first_non_collinear_triplet(vertices, tol=tolerance)
        if vList is None:
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not find three vertices that are not collinear. Returning None.")
            return None

        normal = triplet_normal(vList, tol=tolerance)
        if normal is None:
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not compute a valid normal from the selected vertices. Returning None.")
            return None

        # Canonicalize the plane-normal sign so that flattening is deterministic.
        # The dominant world component is always positive. In particular, an XY
        # topology always uses +Z rather than an arbitrary +Z/-Z normal. This keeps
        # the local rectangle frame right-handed: +X = +U, +Y = +V, +Z = normal.
        dominant_index = max(range(3), key=lambda i: abs(normal[i]))
        if normal[dominant_index] < 0:
            normal = [-normal[0], -normal[1], -normal[2]]

        f_origin = Topology.Centroid(topology)
        topology = Topology.Flatten(topology, origin=f_origin, direction=normal)

        boundingRectangle = br(topology)
        if not boundingRectangle:
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not compute the flattened bounding rectangle. Returning None.")
            return None

        x_min, y_min, x_max, y_max = boundingRectangle

        width = abs(x_max - x_min)
        length = abs(y_max - y_min)

        best_area = width * length
        orig_area = best_area
        best_z = 0
        best_br = [x_min, y_min, x_max, y_max]

        origin = Topology.Centroid(topology)

        optimize = min(max(int(optimize), 0), 10)

        if optimize > 0:
            factor = 1.0 - float(optimize) * 0.05
            flag = False

            for n in range(10, 0, -1):
                if flag:
                    break

                za = n
                zb = 90 + n
                zc = n

                for z in range(za, zb, zc):
                    t = Topology.Rotate(topology, origin=origin, axis=[0, 0, 1], angle=z)
                    bb = br(t)

                    if not bb:
                        continue

                    bx_min, by_min, bx_max, by_max = bb

                    bwidth = abs(bx_max - bx_min)
                    blength = abs(by_max - by_min)
                    area = bwidth * blength

                    if area <= orig_area * factor:
                        best_area = area
                        best_z = z
                        best_br = [bx_min, by_min, bx_max, by_max]
                        flag = True
                        break

                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_br = [bx_min, by_min, bx_max, by_max]

        local_x_min, local_y_min, local_x_max, local_y_max = best_br

        local_width = abs(local_x_max - local_x_min)
        local_length = abs(local_y_max - local_y_min)
        local_origin = Vertex.ByCoordinates(local_x_min, local_y_min, 0)

        # Use the canonical rectangle constructor. With lowerleft placement its
        # boundary starts at the lower-left corner and proceeds counter-clockwise:
        # lower-left -> lower-right -> upper-right -> upper-left.
        boundingRectangle = Wire.Rectangle(
            origin=local_origin,
            width=local_width,
            length=local_length,
            direction=[0, 0, 1],
            placement="lowerleft",
            tolerance=tolerance,
            silent=silent,
        )
        if not Topology.IsInstance(boundingRectangle, "Wire"):
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not create the bounding rectangle wire. Returning None.")
            return None

        # width and length are intentionally measured in the local flattened rectangle frame.
        # They should not be recomputed from world-space diagonal coordinates.
        width = local_width
        length = local_length

        # Rotate the rectangle back from the optimized frame to the flattened topology frame.
        if abs(best_z) > tolerance:
            boundingRectangle = Topology.Rotate(
                boundingRectangle,
                origin=origin,
                axis=[0, 0, 1],
                angle=-best_z
            )

        # Unflatten the rectangle back to the original topology plane.
        boundingRectangle = Topology.Unflatten(
            boundingRectangle,
            origin=f_origin,
            direction=normal
        )

        if not Topology.IsInstance(boundingRectangle, "Wire"):
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not unflatten the bounding rectangle wire. Returning None.")
            return None

        # Compute world-space extents from the final returned wire.
        final_vertices = Topology.Vertices(boundingRectangle)
        if not final_vertices:
            if not silent:
                print("Wire.BoundingRectangle - Error: Could not retrieve vertices from the final bounding rectangle wire. Returning None.")
            return None

        xs = [Vertex.X(v) for v in final_vertices]
        ys = [Vertex.Y(v) for v in final_vertices]
        zs = [Vertex.Z(v) for v in final_vertices]

        world_x_min = min(xs)
        world_y_min = min(ys)
        world_z_min = min(zs)

        world_x_max = max(xs)
        world_y_max = max(ys)
        world_z_max = max(zs)

        dictionary = Dictionary.ByKeysValues(
            [
                "zrot",
                "xmin",
                "ymin",
                "zmin",
                "xmax",
                "ymax",
                "zmax",
                "width",
                "length"
            ],
            [
                round(best_z, mantissa),
                round(world_x_min, mantissa),
                round(world_y_min, mantissa),
                round(world_z_min, mantissa),
                round(world_x_max, mantissa),
                round(world_y_max, mantissa),
                round(world_z_max, mantissa),
                round(width, mantissa),
                round(length, mantissa)
            ]
        )

        boundingRectangle = Topology.SetDictionary(boundingRectangle, dictionary)

        return boundingRectangle

    # @staticmethod
    # def ByEdges(edges: list, orient: bool = False, tolerance: float = 0.0001, silent: bool = False):
    #     """
    #     Creates a wire from the input list of edges.

    #     Parameters
    #     ----------
    #     edges : list
    #         The input list of edges.
    #     orient : bool , optional
    #         If set to True the edges are oriented head to tail. Otherwise, they are not. Default is False.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     topologic_core.Wire
    #         The created wire.

    #     """
    #     from topologicpy.Cluster import Cluster
    #     from topologicpy.Topology import Topology

    #     if not isinstance(edges, list):
    #         return None
    #     edgeList = [x for x in edges if Topology.IsInstance(x, "Edge")]
    #     if len(edgeList) == 0:
    #         if not silent:
    #             print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
    #         return None
    #     if len(edgeList) == 1:
    #         wire = Core.Wire.ByEdges(edgeList)
    #     else:
    #         wire = Topology.SelfMerge(Cluster.ByTopologies(edgeList), tolerance=tolerance)
    #     if not Topology.IsInstance(wire, "Wire"):
    #         if not silent:
    #             print("Wire.ByEdges - Error: The operation failed. Returning None.")
    #         wire = None
    #     if Wire.IsManifold(wire):
    #         if orient == True:
    #             wire = Wire.OrientEdges(wire, Wire.StartVertex(wire), tolerance=tolerance)
    #     return wire

    @staticmethod
    def ByEdges(edges: list, orient: bool = False, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a wire from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        orient : bool , optional
            If set to True, the edges are oriented head-to-tail. Otherwise, they are not. Default is False.
        transferDictionaries : bool , optional
            If set to True, dictionaries in the input edges are transferred to the corresponding edges of the created wire. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        if not isinstance(edges, list):
            if not silent:
                print("Wire.ByEdges - Error: The input edges parameter is not a valid list. Returning None.")
            return None

        edgeList = [edge for edge in edges if Topology.IsInstance(edge, "Edge")]
        if len(edgeList) == 0:
            if not silent:
                print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
            return None

        wire = None

        # PythonOCC: construct directly from the existing OCCT Edge shapes. This
        # avoids a Boolean SelfMerge and preserves native curve geometry.
        if Wire._UseNativeWireBackend():
            try:
                wire = Core.Wire.ByEdges(edgeList, tolerance)
            except Exception:
                wire = None

        # TopologicCore and conservative fallback retain the historical path.
        if not Topology.IsInstance(wire, "Wire"):
            if len(edgeList) == 1:
                try:
                    wire = Core.Wire.ByEdges(edgeList)
                except Exception:
                    wire = None
            else:
                wire = Topology.SelfMerge(
                    Cluster.ByTopologies(edgeList),
                    tolerance=tolerance,
                )

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ByEdges - Error: The operation failed. Returning None.")
            return None

        def _edges_match(edgeA, edgeB):
            startA = Edge.StartVertex(edgeA)
            endA = Edge.EndVertex(edgeA)
            startB = Edge.StartVertex(edgeB)
            endB = Edge.EndVertex(edgeB)
            if not all(Topology.IsInstance(v, "Vertex") for v in [startA, endA, startB, endB]):
                return False
            forward = (
                Vertex.IsCoincident(startA, startB, tolerance=tolerance, silent=True)
                and Vertex.IsCoincident(endA, endB, tolerance=tolerance, silent=True)
            )
            reverse = (
                Vertex.IsCoincident(startA, endB, tolerance=tolerance, silent=True)
                and Vertex.IsCoincident(endA, startB, tolerance=tolerance, silent=True)
            )
            return bool(forward or reverse)

        # Preserve the historical dictionary propagation from matching source
        # Edges. The explicit transferDictionaries option below additionally
        # merges dictionaries where several source edges enclose one result edge.
        resultEdges = Topology.Edges(wire, silent=True) or []
        if len(resultEdges) > 0:
            newEdges = []
            for resultEdge in resultEdges:
                updatedEdge = resultEdge
                for sourceEdge in edgeList:
                    if _edges_match(resultEdge, sourceEdge):
                        dictionary = Topology.Dictionary(sourceEdge)
                        if dictionary:
                            candidate = Topology.SetDictionary(
                                updatedEdge,
                                dictionary,
                                silent=True,
                            )
                            if candidate is not None:
                                updatedEdge = candidate
                        break
                newEdges.append(updatedEdge)

            try:
                rebuiltWire = Core.Wire.ByEdges(newEdges, tolerance) if Wire._UseNativeWireBackend() else Core.Wire.ByEdges(newEdges)
            except Exception:
                rebuiltWire = None
            if Topology.IsInstance(rebuiltWire, "Wire"):
                wire = rebuiltWire

        if Wire.IsManifold(wire, silent=True, tolerance=tolerance) and orient:
            start = Wire.StartVertex(wire, silent=True, tolerance=tolerance)
            if Topology.IsInstance(start, "Vertex"):
                oriented = Wire.OrientEdges(
                    wire,
                    start,
                    transferDictionaries=transferDictionaries,
                    tolerance=tolerance,
                    silent=silent,
                )
                if Topology.IsInstance(oriented, "Wire"):
                    wire = oriented

        if transferDictionaries:
            wire_edges = Topology.Edges(wire, silent=True) or []
            source_cluster = Cluster.ByTopologies(edgeList)
            if source_cluster is not None:
                for wire_edge in wire_edges:
                    internal_vertex = Topology.InternalVertex(
                        wire_edge,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if not Topology.IsInstance(internal_vertex, "Vertex"):
                        continue
                    enclosing_edges = Vertex.EnclosingEdges(
                        internal_vertex,
                        source_cluster,
                        exclusive=False,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if isinstance(enclosing_edges, list) and len(enclosing_edges) > 0:
                        dictionaries = [Topology.Dictionary(edge) for edge in enclosing_edges]
                        merged_dictionary = Dictionary.ByMergedDictionaries(
                            dictionaries,
                            silent=True,
                        )
                        if merged_dictionary:
                            Topology.SetDictionary(
                                wire_edge,
                                merged_dictionary,
                                silent=True,
                            )
        return wire
    

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Wire.ByEdgesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = Topology.Edges(cluster, silent=True)
        return Wire.ByEdges(edges, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByOffset(wire, offset: float = 1.0, offsetKey: str = "offset", stepOffsetA: float = 0, stepOffsetB: float = 0, stepOffsetKeyA: str = "stepOffsetA", stepOffsetKeyB: str = "stepOffsetB", reverse: bool = False, bisectors: bool = False, transferDictionaries: bool = False, epsilon: float = 0.01, tolerance: float = 0.0001,  silent: bool = False, numWorkers: int = None):
        """
        Creates an offset wire from the input wire. A positive offset value results in an offset to the interior of an anti-clockwise wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float , optional
            The desired offset distance. Default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. Default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. Default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. Default is "stepOffsetB".
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. Default is False.
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned wire. Default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the new wire. Otherwise, they are not. Default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). Default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
        
        if reverse == True:
            fac = -1
        else:
            fac = 1
        origin = Topology.Centroid(wire)
        temp_vertices = [Topology.Vertices(wire)[0], Topology.Vertices(wire)[1], Topology.Centroid(wire)]
        temp_face = Face.ByWire(Wire.ByVertices(temp_vertices, close=True, tolerance=tolerance, silent=True), silent=True)
        if not temp_face:
            if not silent:
                print("Wire.Offset - Error: The input wire has errors. Returning None.")
            return None
        normal = Face.Normal(temp_face)
        flat_wire = Topology.Flatten(wire, direction=normal, origin=origin)
        original_edges = Topology.Edges(wire)
        edges = Topology.Edges(flat_wire)
        offsets = []
        offset_edges = []
        final_vertices = []
        bisectors_list = []
        edge_dictionaries = []
        for i, edge in enumerate(edges):
            d = Topology.Dictionary(original_edges[i])
            d_offset = Dictionary.ValueAtKey(d, key=offsetKey, defaultValue=offset)
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
                        b_e = Edge.ByVertices(v_a, v1_1, silent=True)
                        if b_e:
                            bisectors_list.append(b_e)
                        b_e = Edge.ByVertices(v_a, v1_2, silent=True)
                        if b_e:
                            bisectors_list.append(b_e)
                        final_vertices.append(v1_1)
                        final_vertices.append(v1_2)
        v_a = Edge.EndVertex(edges[-1])
        if Wire.IsClosed(wire) == False:
            v1 = Edge.EndVertex(offset_edges[-1])
            final_vertices.append(v1)
            if transferDictionaries == True:
                v1 = Topology.SetDictionary(v1, Topology.Dictionary(v_a), silent=True)
            if bisectors == True:
                b_e = Edge.ByVertices(v_a, v1, silent=True)
                if b_e:
                    bisectors_list.append(b_e)
        return_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance, silent=silent)
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
            i = 0
            temp_return_wire = Topology.SelfMerge(Cluster.ByTopologies(wire_edges+bisectors_list))
            while not Topology.IsInstance(temp_return_wire, "wire") and i < 9:
                verts = Topology.Vertices(temp_return_wire)
                new_verts = Vertex.Fuse(verts, tolerance=tolerance*(i+1)*10)
                temp_return_wire = Topology.ReplaceVertices(temp_return_wire, verticesA=verts, verticesB=new_verts)
                temp_return_wire = Topology.SelfMerge(temp_return_wire)
                i += 1
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
                temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_vertices, tranVertices=True, tolerance=tolerance*10, numWorkers=numWorkers)
                temp_return_wire = Topology.TransferDictionariesBySelectors(temp_return_wire, sel_edges, tranEdges=True, tolerance=tolerance*10, numWorkers=numWorkers)
                
            return_wire = temp_return_wire
        
        
        if not Topology.IsInstance(return_wire, "Wire"):
            if not silent:
                print("Wire.ByOffset - Warning: The resulting wire is not well-formed, please check your offsets.")
        else:
            if not Wire.IsManifold(return_wire) and bisectors == False:
                if not silent:
                    print("Wire.ByOffset - Warning: The resulting wire is non-manifold, please check your offsets.")
                    print("Wire.ByOffset - Warning: Pursuing a workaround, but it might take longer to complete.")
                
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
            The edge dictionary key under which to store the offset value. Default is "offset".
        minOffsetKey : str , optional
            The edge dictionary key under which to find the desired minimum edge offset value. If a value cannot be found, the defaultMinOffset input parameter value is used instead. Default is "minOffset".
        maxOffsetKey : str , optional
            The edge dictionary key under which to find the desired maximum edge offset value. If a value cannot be found, the defaultMaxOffset input parameter value is used instead. Default is "maxOffset".
        defaultMinOffset : float , optional
            The desired minimum edge offset distance. Default is 0.
        defaultMaxOffset : float , optional
            The desired maximum edge offset distance. Default is 1.
        maxIterations: int , optional
            The desired maximum number of iterations to attempt to converge on a solution. Default is 1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
    def ByTGraphVertices(tGraph, vertices, close: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a topologic Wire from an ordered list of TGraph vertex indices.

        The created Topologic vertices inherit the dictionaries of the corresponding
        TGraph vertices. The created Topologic edges inherit the dictionaries of the
        corresponding TGraph edges when such edges exist in the TGraph.

        Parameters
        ----------
        tGraph : topologicpy.TGraph
            The input TGraph.
        vertices : list
            An ordered list of TGraph vertex indices.
        close : bool, optional
            If True, an additional edge is created from the last vertex back to the
            first vertex. Default is False.
        tolerance : float, optional
            The tolerance used by TopologicPy constructors. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created Wire. Returns None if the input is invalid or if the Wire
            cannot be constructed.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if tGraph is None:
            if not silent:
                print("Wire.ByTGraphVertices - Error: The input tGraph is None. Returning None.")
            return None
        if not isinstance(vertices, list):
            if not silent:
                print("Wire.ByTGraphVertices - Error: The input vertices parameter is not a list. Returning None.")
            return None
        if len(vertices) < 2:
            if not silent:
                print("Wire.ByTGraphVertices - Error: The input vertices parameter contains less than 2 elements. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Small local helpers to keep this method robust against minor TGraph
        # implementation differences.
        # -------------------------------------------------------------------------

        def _vertex_data(tg, v_index):
            """
            Returns the internal TGraph vertex data dictionary for v_index.
            """
            try:
                if hasattr(tg, "_vertices"):
                    return tg._vertices[v_index]
            except Exception:
                pass

            try:
                if hasattr(tg, "vertices"):
                    return tg.vertices[v_index]
            except Exception:
                pass

            try:
                if hasattr(tg, "Vertices"):
                    return tg.Vertices()[v_index]
            except Exception:
                pass

            return None

        def _edge_data(tg, u, v):
            """
            Returns the internal TGraph edge data dictionary between u and v.
            Tries both directed and undirected storage conventions.
            """
            candidate_keys = [
                (u, v),
                (v, u),
                f"{u}-{v}",
                f"{v}-{u}",
                f"{u}_{v}",
                f"{v}_{u}",
            ]

            for attr_name in ["_edges", "edges"]:
                try:
                    edge_store = getattr(tg, attr_name)
                    if isinstance(edge_store, dict):
                        for key in candidate_keys:
                            if key in edge_store:
                                return edge_store[key]
                    elif isinstance(edge_store, list):
                        for e in edge_store:
                            if not isinstance(e, dict):
                                continue
                            eu = e.get("u", e.get("src", e.get("source", e.get("from"))))
                            ev = e.get("v", e.get("dst", e.get("target", e.get("to"))))
                            if (eu == u and ev == v) or (eu == v and ev == u):
                                return e
                except Exception:
                    pass

            try:
                if hasattr(tg, "Edge"):
                    return tg.Edge(u, v)
            except Exception:
                pass

            try:
                if hasattr(tg, "EdgeData"):
                    return tg.EdgeData(u, v)
            except Exception:
                pass

            return None

        def _dictionary_from_data(data):
            """
            Extracts a Topologic dictionary or builds one from plain Python metadata.
            """
            if data is None:
                return None

            # Already a Topologic dictionary.
            try:
                if Dictionary.IsInstance(data):
                    return data
            except Exception:
                pass

            if not isinstance(data, dict):
                return None

            # Common TGraph storage conventions.
            for key in ["dictionary", "Dictionary", "dict", "attributes", "data"]:
                value = data.get(key)
                if value is None:
                    continue

                try:
                    if Dictionary.IsInstance(value):
                        return value
                except Exception:
                    pass

                if isinstance(value, dict):
                    try:
                        return Dictionary.ByPythonDictionary(value)
                    except Exception:
                        pass

            # Fallback: use the whole data dictionary, excluding structural keys.
            excluded = {
                "x", "y", "z",
                "u", "v", "src", "dst", "source", "target", "from", "to",
                "index", "id"
            }

            py_dict = {}
            for k, v in data.items():
                if k in excluded:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    py_dict[k] = v

            if len(py_dict) == 0:
                return None

            try:
                return Dictionary.ByPythonDictionary(py_dict)
            except Exception:
                return None

        def _coords_from_vertex_data(data):
            """
            Extracts xyz coordinates from a TGraph vertex data dictionary.
            """
            if not isinstance(data, dict):
                return None

            # Common direct convention.
            if all(k in data for k in ["x", "y", "z"]):
                return data["x"], data["y"], data["z"]

            # Common uppercase convention.
            if all(k in data for k in ["X", "Y", "Z"]):
                return data["X"], data["Y"], data["Z"]

            # Common coordinate tuple/list conventions.
            for key in ["coordinates", "coords", "point", "position", "xyz"]:
                value = data.get(key)
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    return value[0], value[1], value[2]

            # Existing topologic vertex convention.
            for key in ["vertex", "topologic_vertex", "topology"]:
                value = data.get(key)
                if value is None:
                    continue
                try:
                    return Vertex.X(value), Vertex.Y(value), Vertex.Z(value)
                except Exception:
                    pass

            return None

        # -------------------------------------------------------------------------
        # Build Topologic vertices.
        # -------------------------------------------------------------------------

        topologic_vertices = []

        for v_index in vertices:
            v_data = _vertex_data(tGraph, v_index)
            coords = _coords_from_vertex_data(v_data)

            if coords is None:
                return None

            try:
                tv = Vertex.ByCoordinates(float(coords[0]), float(coords[1]), float(coords[2]))
            except Exception:
                return None

            v_dict = _dictionary_from_data(v_data)
            if v_dict is not None:
                try:
                    tv = Topology.SetDictionary(tv, v_dict)
                except Exception:
                    pass

            topologic_vertices.append(tv)

        # -------------------------------------------------------------------------
        # Build Topologic edges and transfer TGraph edge dictionaries.
        # -------------------------------------------------------------------------

        edges = []
        index_pairs = list(zip(vertices[:-1], vertices[1:]))

        if close:
            index_pairs.append((vertices[-1], vertices[0]))

        for i, (u, v) in enumerate(index_pairs):
            start_vertex = topologic_vertices[i]
            end_vertex = topologic_vertices[(i + 1) % len(topologic_vertices)]

            try:
                e = Edge.ByStartVertexEndVertex(start_vertex, end_vertex, tolerance=tolerance)
            except TypeError:
                e = Edge.ByStartVertexEndVertex(start_vertex, end_vertex)
            except Exception:
                return None

            if e is None:
                return None

            e_data = _edge_data(tGraph, u, v)
            e_dict = _dictionary_from_data(e_data)

            if e_dict is not None:
                try:
                    e = Topology.SetDictionary(e, e_dict)
                except Exception:
                    pass

            edges.append(e)

        if len(edges) == 0:
            return None

        # -------------------------------------------------------------------------
        # Build and return Wire.
        # -------------------------------------------------------------------------

        try:
            return Wire.ByEdges(edges, tolerance=tolerance)
        except TypeError:
            return Wire.ByEdges(edges)
        except Exception:
            return None


    @staticmethod
    def ByVertices(vertices: list, close: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a wire from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        close : bool , optional
            If True, the last vertex will be connected to the first vertex to close
            the wire. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        import inspect

        if not isinstance(vertices, list):
            return None

        vertexList = [v for v in vertices if Topology.IsInstance(v, "Vertex")]

        if len(vertexList) < 2:
            if not silent:
                print("Wire.ByVertices - Error: The number of vertices is less than 2. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])
            return None

        # -------------------------------------------------------------------------
        # First attempt: use the active backend's native implementation.
        # -------------------------------------------------------------------------
        try:
            if Wire._UseNativeWireBackend() and Core.HasAttribute("Wire", "ByVertices"):
                wire = Core.Wire.ByVertices(
                    vertexList,
                    close,
                    tolerance
                )
                if Topology.IsInstance(wire, "Wire"):
                    return wire
        except Exception:
            pass

        # -------------------------------------------------------------------------
        # Fallback: construct edges using the TopologicPy algorithm layer.
        # -------------------------------------------------------------------------
        edges = []

        for i in range(len(vertexList) - 1):
            v1 = vertexList[i]
            v2 = vertexList[i + 1]

            e = Edge.ByVertices(
                [v1, v2],
                tolerance=tolerance,
                silent=True
            )

            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
            elif not silent:
                print("Wire.ByVertices - Warning: Degenerate edge. Skipping.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])

        if close:
            v1 = vertexList[-1]
            v2 = vertexList[0]

            e = Edge.ByVertices(
                [v1, v2],
                tolerance=tolerance,
                silent=True
            )

            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
            elif not silent:
                print("Wire.ByVertices - Warning: Degenerate edge. Skipping.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])

        if len(edges) < 1:
            if not silent:
                print("Wire.ByVertices - Error: The number of edges is less than 1. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])
            return None

        if len(edges) == 1:
            if not silent:
                print("Wire.ByVertices - Warning: The wire is made of only one edge.")
            wire = Wire.ByEdges(
                edges,
                orient=False,
                tolerance=tolerance,
                silent=silent
            )
        else:
            wire = Topology.SelfMerge(
                Cluster.ByTopologies(edges),
                tolerance=tolerance
            )

            if Topology.IsInstance(wire, "Edge"):
                wire = Wire.ByEdges(
                    [wire],
                    orient=False,
                    tolerance=tolerance,
                    silent=silent
                )

        # Final check.
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ByVertices - Error: Could not create a wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])
            return None

        return wire
    # @staticmethod
    # def ByVertices(vertices: list, close: bool = True, tolerance: float = 0.0001, silent: bool = False):
    #     """
    #     Creates a wire from the input list of vertices.

    #     Parameters
    #     ----------
    #     vertices : list
    #         the input list of vertices.
    #     close : bool , optional
    #         If True the last vertex will be connected to the first vertex to close the wire. Default is True.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     topologic_core.Wire
    #         The created wire.

    #     """
    #     from topologicpy.Edge import Edge
    #     from topologicpy.Cluster import Cluster
    #     from topologicpy.Topology import Topology
    #     import inspect

    #     if not isinstance(vertices, list):
    #         return None
    #     vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
    #     if len(vertexList) < 2:
    #         if not silent:
    #             print("Wire.ByVertices - Error: The number of vertices is less than 2. Returning None.")
    #             curframe = inspect.currentframe()
    #             calframe = inspect.getouterframes(curframe, 2)
    #             print('caller name:', calframe[1][3])
    #         return None
    #     edges = []
    #     for i in range(len(vertexList)-1):
    #         v1 = vertexList[i]
    #         v2 = vertexList[i+1]
    #         e = Edge.ByVertices([v1, v2], tolerance=tolerance, silent=True)
    #         if Topology.IsInstance(e, "Edge"):
    #             edges.append(e)
    #         else:
    #             if not silent:
    #                 print("Wire.ByVertices - Warning: Degenerate edge. Skipping.")
    #                 curframe = inspect.currentframe()
    #                 calframe = inspect.getouterframes(curframe, 2)
    #                 print('caller name:', calframe[1][3])
    #     if close:
    #         v1 = vertexList[-1]
    #         v2 = vertexList[0]
    #         e = Edge.ByVertices([v1, v2], tolerance=tolerance, silent=True) # We want to force suppress errors and warnings here.
    #         if Topology.IsInstance(e, "Edge"):
    #             edges.append(e)
    #         else:
    #             if not silent:
    #                 print("Wire.ByVertices - Warning: Degenerate edge. Skipping.")
    #                 curframe = inspect.currentframe()
    #                 calframe = inspect.getouterframes(curframe, 2)
    #                 print('caller name:', calframe[1][3])
        
    #     if len(edges) < 1:
    #         if not silent:
    #             print("Wire.ByVertices - Error: The number of edges is less than 1. Returning None.")
    #             curframe = inspect.currentframe()
    #             calframe = inspect.getouterframes(curframe, 2)
    #             print('caller name:', calframe[1][3])
    #         return None
    #     elif len(edges) == 1:
    #         if not silent:
    #             print("Wire.ByVertices - Warning: The wire is made of only one edge.")
    #         wire = Wire.ByEdges(edges, orient=False, silent=silent)
    #     else:
    #         wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
    #         if Topology.IsInstance(wire, "Edge"):
    #             wire = Wire.ByEdges([wire], orient=False, silent=silent)
    #     # Final Check
    #     if not Topology.IsInstance(wire, "Wire"):
    #         if not silent:
    #             print("Wire.ByVertices - Error: Could not create a wire. Returning None.")
    #             curframe = inspect.currentframe()
    #             calframe = inspect.getouterframes(curframe, 2)
    #             print('caller name:', calframe[1][3])
    #         return None
    #     return wire

    @staticmethod
    def ByVerticesCluster(cluster, close: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a wire from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.cluster
            the input cluster of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Wire.ByVerticesCluster - Error: The input cluster parameter is not a valid cluster. Returning None.")
            return None
        vertices = Topology.Vertices(cluster)
        return Wire.ByVertices(vertices, close=close, tolerance=tolerance, silent=silent)


    @staticmethod
    def Cage(origin=None,
            width: float = 1.0, length: float = 1.0, height: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center",
            mantissa: int = 6, tolerance: float = 0.0001,
            radius: float = 0.0, base=None, silent: bool = False):
        """
        Creates a prismatic 3D cage as a Wire, with edges only on the outer
        surfaces of the volume (no interior lines).

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the cage:
            - If placement == "center": the geometric center of the cage
            is placed at this origin.
            - If placement == "corner": the minimum corner of the cage
            is placed at this origin.
            If None, the cage is created around (0, 0, 0) accordingly.
        width : float , optional
            The size of the cage in the local X direction. Default is 1.0.
        length : float , optional
            The size of the cage in the local Y direction. Default is 1.0.
        height : float , optional
            The size of the cage in the local Z direction. Default is 1.0.
        uSides : int , optional
            The number of subdivisions in the local X direction. Must be >= 1.
            Default is 2.
        vSides : int , optional
            The number of subdivisions in the local Y direction. Must be >= 1.
            Default is 2.
        wSides : int , optional
            The number of subdivisions in the local Z direction. Must be >= 1.
            Default is 2.
        direction : list , optional
            The vector representing the up direction of the lattice. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the lattice. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire or None
            The resulting cage Wire, or None if inputs are invalid.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Face import Face
        from topologicpy.Dictionary import Dictionary
        import math

        # -------------------------
        # Resolve base face (first positional may be a Face)
        # -------------------------
        if base is None and origin is not None and Topology.IsInstance(origin, "Face"):
            base = origin
            origin = None
        if base is not None and Topology.IsInstance(base, "Face"):
            bb = Topology.BoundingBox(base, tolerance=tolerance)
            # Backend BoundingBox returns a Face carrying xmin/xmax/ymin/ymax in its dictionary.
            d = Topology.Dictionary(bb)
            def _num(k):
                a = Dictionary.ValueAtKey(d, k) if d is not None else None
                return float(a) if a is not None else None
            xmin = _num("xmin"); xmax = _num("xmax")
            ymin = _num("ymin"); ymax = _num("ymax")
            width = abs(xmax - xmin)
            length = abs(ymax - ymin)
            height = radius if radius > 0 else 1.0
            placement = "center"
            if origin is None:
                origin = Topology.Centroid(base)

        # -------------------------
        # Validation
        # -------------------------
        if uSides < 1 or vSides < 1 or wSides < 1:
            if not silent:
                print("Wire.Cage - Error: uSides, vSides, and wSides must be >= 1. Returning None.")
            return None
        if width <= 0 or length <= 0 or height <= 0:
            if not silent:
                print("Wire.Cage - Error: width, length, and height must be positive. Returning None.")
            return None

        if origin is None:
            origin = Vertex.ByCoordinates(0, 0, 0)

        # Local origin at (0,0,0) for construction and rotation
        local_origin = Vertex.ByCoordinates(0, 0, 0)

        # -------------------------
        # Local Placement Offsets
        # -------------------------
        # We construct the cage in a local coordinate system.
        if str(placement).lower() == "center":
            ox = -width * 0.5
            oy = -length * 0.5
            oz = -height * 0.5
        elif str(placement).lower() == "bottom":
            ox = -width * 0.5
            oy = -length * 0.5
            oz = 0
        else:  # "lowerleft"
            ox = 0.0
            oy = 0.0
            oz = 0.0

        # -------------------------
        # Step Sizes
        # -------------------------
        du = width / uSides
        dv = length / vSides
        dw = height / wSides

        # -------------------------
        # Grid Coordinates (local)
        # -------------------------
        xs = [round(ox + i * du, mantissa) for i in range(uSides + 1)]
        ys = [round(oy + j * dv, mantissa) for j in range(vSides + 1)]
        zs = [round(oz + k * dw, mantissa) for k in range(wSides + 1)]

        # -------------------------
        # Build a single connected serpentine wire traversing the boundary
        # surface nodes. A cage boundary is non-manifold (grid nodes of degree
        # > 2), so it cannot be one manifold wire via Wire.ByEdges; the
        # serpentine path is a valid single Wire carrying the cage topology.
        # -------------------------
        nodes = []
        for zi, z in enumerate(zs):
            on_z = (zi == 0 or zi == wSides)
            row_xs = xs if zi % 2 == 0 else list(reversed(xs))
            for y in ys:
                on_y = (y == ys[0] or y == ys[-1])
                if not (on_z or on_y):
                    continue
                for x in row_xs:
                    nodes.append(Vertex.ByCoordinates(x, y, z))
        for xi, x in enumerate(xs):
            on_x = (xi == 0 or xi == uSides)
            if not on_x:
                continue
            for y in ys:
                on_y = (y == ys[0] or y == ys[-1])
                if not on_y:
                    continue
                for z in zs:
                    nodes.append(Vertex.ByCoordinates(x, y, z))

        if not nodes:
            if not silent:
                print("Wire.Cage - Warning: No edges created. Returning None.")
            return None

        cage = Wire.ByVertices(nodes, close=False, tolerance=tolerance, silent=silent)

        # -------------------------
        # Orient and Place
        # -------------------------
        if cage is not None:
            cage = Topology.Orient(cage, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
            cage = Topology.Place(cage, originA=Vertex.Origin(), originB=origin)
        return cage


    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 16, spokes: bool = False, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. Default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. Default is 0.5.
        sides : int , optional
            The desired number of sides of the circle. Default is 16.
        spokes : bool , optional
            If set to True, spoke edges from the center to the circumference are added. Default is False.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. Default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created circle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Circle - Error: The input origin parameter is not a valid Vertex. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            if not silent:
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
            baseWire = Wire.ByVertices(baseV[::-1], close=False, tolerance=tolerance, silent=silent) # Counter-clockwise in local XY; normal is +Z
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close, tolerance=tolerance, silent=silent) # Counter-clockwise in local XY; normal is +Z

        if spokes == True and (angleRange == 360 or close==False):
            vertices = Topology.Vertices(baseWire)
            base_edges = Topology.Edges(baseWire)
            spoke_edges = []
            for v in vertices:
                e = Edge.ByVertices(origin, v, tolerance=tolerance)
                if e:
                    spoke_edges.append(e)
            if len(spoke_edges) > 0:
                baseWire = Wire.ByEdges(base_edges+spoke_edges)
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
    def Close(wire, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Closes the input wire

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
                
        Returns
        -------
        topologic_core.Wire
            The closed version of the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        import inspect
        
        def nearest_vertex(vertex, vertices):
            distances = []
            for v in vertices:
                distances.append(Vertex.Distance(vertex, v))
            new_vertices = Helper.Sort(vertices, distances)
            return new_vertices[1] #The first item is the same vertex, so return the next nearest vertex.
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Close - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if Wire.IsClosed(wire):
            return wire
        vertices = Topology.Vertices(wire)
        ends = [v for v in vertices if Vertex.Degree(v, wire) == 1]
        if len(ends) < 2:
            if not silent:
                print("Wire.Close - Error: The input wire parameter contains less than two open end vertices. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
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
    def ConcaveHull(topology, k: int = 3, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
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
            less concave hull, while decreasing `k` may yield a more detailed, concave shape. Default is 3.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
                
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

        if not Topology.IsInstance(topology, "topology"):
            if not silent:
                print("Wire.ConcaveHull - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        f = None
        # Create a sample face and flatten
        while not Topology.IsInstance(f, "Face"):
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v, tolerance=tolerance, silent=silent)
            f = Face.ByWire(w, tolerance=tolerance, silent=silent)
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
        ch = Wire.ByVertices(hull_vertices, close=True, tolerance=tolerance, silent=silent)
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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The convex hull of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        def _cross_mag2(p0, p1, vv):
            """Squared magnitude of cross((p1-p0),(vv-p0))."""
            ux = Vertex.X(p1)-Vertex.X(p0)
            uy = Vertex.Y(p1)-Vertex.Y(p0)
            uz = Vertex.Z(p1)-Vertex.Z(p0)
            vx = Vertex.X(vv)-Vertex.X(p0)
            vy = Vertex.Y(vv)-Vertex.Y(p0)
            vz = Vertex.Z(vv)-Vertex.Z(p0)
            cx = uy*vz-uz*vy
            cy = uz*vx-ux*vz
            cz = ux*vy-uy*vx
            return cx*cx+cy*cy+cz*cz

        def _pick_triple(vertices):
            """Deterministic, order-independent, always-valid non-collinear triple.

            Choose p0 = lexicographically-smallest vertex, p1 = farthest from p0,
            p2 = farthest from the line through p0-p1.  This guarantees a
            non-collinear triplet whenever one exists, is stable across both
            backends, and does not depend on vertex enumeration order.
            Returns (p0, p1, p2) or None when < 3 distinct points or when all
            points are collinear.
            """
            def _key(vv):
                return (Vertex.X(vv), Vertex.Y(vv), Vertex.Z(vv))
            def _d2(a, b):
                return (Vertex.X(a)-Vertex.X(b))**2 + (Vertex.Y(a)-Vertex.Y(b))**2 + (Vertex.Z(a)-Vertex.Z(b))**2
            vs = sorted(vertices, key=_key)
            if len(vs) < 3:
                return None
            p0 = vs[0]
            p1 = max(vs[1:], key=lambda vv: (_d2(p0, vv), _key(vv)))
            p2 = max(vs, key=lambda vv: (_cross_mag2(p0, p1, vv), _key(vv)))
            if _cross_mag2(p0, p1, p2) <= 1e-18:
                return None  # all collinear
            return p0, p1, p2

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

        # Deterministic, order-independent flattening-plane selection.
        # (The previous implementation used random.sample(vertices, 3), which
        # produced a different plane on every call -> nondeterministic hull
        # vertex order and, for near-degenerate inputs, occasionally wrong hull
        # points. A fixed non-collinear triple gives the same plane to both
        # backends.)
        vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
        triple = _pick_triple(vertices)
        if triple is None:
            # Degenerate: fewer than 3 distinct points or all collinear.
            # Return the extremal segment (the 1D hull) when 2+ distinct points exist.
            vs = sorted(vertices, key=lambda vv: (Vertex.X(vv), Vertex.Y(vv), Vertex.Z(vv)))
            if len(vs) >= 2:
                spans = []
                for i in range(len(vs)):
                    for j in range(i+1, len(vs)):
                        spans.append(((Vertex.X(vs[i])-Vertex.X(vs[j]))**2 + (Vertex.Y(vs[i])-Vertex.Y(vs[j]))**2 + (Vertex.Z(vs[i])-Vertex.Z(vs[j]))**2, vs[i], vs[j]))
                _, a, b = max(spans)
                return Wire.ByVertices([a, b], tolerance=tolerance)
            if len(vs) == 1:
                return Wire.ByVertices([vs[0]], close=False, tolerance=tolerance)
            return None
        p0, p1, p2 = triple
        w = Wire.ByVertices([p0, p1, p2], tolerance=tolerance)
        f = Face.ByWire(w, tolerance=tolerance)
        if not Topology.IsInstance(f, "Face"):
            return None
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
    def _CornerVerticesByAngle(
        wire,
        cornerType: str = "convex",
        angTolerance: float = 0.01,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> list:
        """
        Returns convex or concave corner vertices of a closed manifold wire.

        This method only accepts closed manifold wires that form a single
        non-branching cycle. Every vertex must be incident to exactly two edges.
        Open wires, disconnected wires, and branched/non-manifold wires are rejected.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        cornerType : str , optional
            The corner type to return. Options are "convex" and "concave".
            Default is "convex".
        angTolerance : float , optional
            The angular tolerance in degrees. Default is 0.01.
        mantissa : int , optional
            The number of decimal places to round computed angles to. Default is 6.
        tolerance : float , optional
            The geometric tolerance used for endpoint matching. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of convex or concave corner vertices.
        """

        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: The input wire parameter is not a valid wire. Returning None.")
            return None

        try:
            if not Wire.IsClosed(wire):
                if not silent:
                    print("Wire._CornerVerticesByAngle - Error: The input wire is not closed. Returning None.")
                return None
        except Exception:
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: Could not determine if the input wire is closed. Returning None.")
            return None

        try:
            if not Wire.IsManifold(wire):
                if not silent:
                    print("Wire._CornerVerticesByAngle - Error: The input wire is non-manifold. Returning None.")
                return None
        except Exception:
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: Could not determine if the input wire is manifold. Returning None.")
            return None

        cornerType = str(cornerType).strip().lower()
        if cornerType not in ["convex", "concave"]:
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: The cornerType parameter must be either 'convex' or 'concave'. Returning None.")
            return None

        def _xyz(vertex):
            try:
                return [
                    float(Vertex.X(vertex)),
                    float(Vertex.Y(vertex)),
                    float(Vertex.Z(vertex)),
                ]
            except Exception:
                return None

        def _sub(a, b):
            return [
                a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2],
            ]

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]

        def _length(v):
            return math.sqrt(_dot(v, v))

        def _normalise(v):
            length = _length(v)
            if length <= max(float(tolerance), 1e-12):
                return None
            return [
                v[0] / length,
                v[1] / length,
                v[2] / length,
            ]

        def _distance_squared(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            dz = a[2] - b[2]
            return dx*dx + dy*dy + dz*dz

        def _same_point(a, b):
            if a is None or b is None:
                return False
            return _distance_squared(a, b) <= tolerance*tolerance

        def _edge_vertices(edge):
            sv = None
            ev = None

            try:
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
            except Exception:
                pass

            if sv is not None and ev is not None:
                return sv, ev

            try:
                sv = edge.StartVertex()
                ev = edge.EndVertex()
            except Exception:
                pass

            if sv is not None and ev is not None:
                return sv, ev

            try:
                vertices = Topology.Vertices(edge)
                if isinstance(vertices, list) and len(vertices) >= 2:
                    return vertices[0], vertices[1]
            except Exception:
                pass

            return None, None

        def _edges_from_wire(wire):
            try:
                edges = Topology.Edges(wire)
                if isinstance(edges, list):
                    return edges
            except Exception:
                pass

            try:
                edges = wire.Edges()
                if isinstance(edges, list):
                    return edges
            except Exception:
                pass

            return []

        def _ordered_vertices_from_closed_wire(wire):
            """
            Orders a closed manifold wire as a single vertex cycle.

            Returns an ordered list of vertices without repeating the first vertex at
            the end. Returns None if the wire cannot be represented as one closed
            non-branching cycle.
            """

            edges = _edges_from_wire(wire)

            if len(edges) < 3:
                if not silent:
                    print("Wire._CornerVerticesByAngle - Error: The input wire has fewer than three usable edges. Returning None.")
                return None

            nodes = []
            node_vertices = []
            edge_node_pairs = []

            def _node_index(point, vertex):
                for i, existing_point in enumerate(nodes):
                    if _same_point(point, existing_point):
                        if node_vertices[i] is None and vertex is not None:
                            node_vertices[i] = vertex
                        return i

                nodes.append(point)
                node_vertices.append(vertex)
                return len(nodes) - 1

            for edge in edges:
                sv, ev = _edge_vertices(edge)

                if sv is None or ev is None:
                    continue

                p1 = _xyz(sv)
                p2 = _xyz(ev)

                if p1 is None or p2 is None:
                    continue

                if _same_point(p1, p2):
                    continue

                n1 = _node_index(p1, sv)
                n2 = _node_index(p2, ev)

                if n1 == n2:
                    continue

                edge_node_pairs.append((n1, n2))

            if len(edge_node_pairs) < 3 or len(nodes) < 3:
                if not silent:
                    print("Wire._CornerVerticesByAngle - Error: Could not extract enough non-degenerate edges from the input wire. Returning None.")
                return None

            adjacency = {}
            for edge_index, (n1, n2) in enumerate(edge_node_pairs):
                adjacency.setdefault(n1, []).append((n2, edge_index))
                adjacency.setdefault(n2, []).append((n1, edge_index))

            # For a single closed manifold cycle, every vertex must have degree 2.
            for node_index, neighbours in adjacency.items():
                if len(neighbours) != 2:
                    if not silent:
                        print(
                            "Wire._CornerVerticesByAngle - Error: "
                            "The input wire is not a single closed manifold cycle. "
                            "Each vertex must be incident to exactly two edges. Returning None."
                        )
                    return None

            start = min(adjacency.keys())
            ordered_node_indices = [start]
            used_edges = set()
            previous_node = None
            current_node = start

            while True:
                candidates = adjacency.get(current_node, [])

                next_node = None
                next_edge_index = None

                for candidate_node, candidate_edge_index in candidates:
                    if candidate_edge_index in used_edges:
                        continue

                    if previous_node is not None and candidate_node == previous_node and len(candidates) > 1:
                        continue

                    next_node = candidate_node
                    next_edge_index = candidate_edge_index
                    break

                if next_node is None:
                    break

                used_edges.add(next_edge_index)

                if next_node == start:
                    break

                ordered_node_indices.append(next_node)
                previous_node = current_node
                current_node = next_node

                if len(ordered_node_indices) > len(edge_node_pairs):
                    if not silent:
                        print("Wire._CornerVerticesByAngle - Error: Could not extract a valid closed cycle. Returning None.")
                    return None

            if len(used_edges) != len(edge_node_pairs):
                if not silent:
                    print(
                        "Wire._CornerVerticesByAngle - Error: "
                        "The input wire is disconnected or contains more than one cycle. Returning None."
                    )
                return None

            if len(ordered_node_indices) != len(nodes):
                if not silent:
                    print(
                        "Wire._CornerVerticesByAngle - Error: "
                        "The input wire does not form one simple ordered vertex loop. Returning None."
                    )
                return None

            ordered_vertices = []

            for node_index in ordered_node_indices:
                vertex = node_vertices[node_index]
                if vertex is None:
                    return None
                ordered_vertices.append(vertex)

            return ordered_vertices

        def _newell_normal(vertices):
            points = [_xyz(v) for v in vertices]
            points = [p for p in points if p is not None]

            if len(points) < 3:
                return None

            nx = 0.0
            ny = 0.0
            nz = 0.0
            n = len(points)

            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                nx += (p1[1] - p2[1]) * (p1[2] + p2[2])
                ny += (p1[2] - p2[2]) * (p1[0] + p2[0])
                nz += (p1[0] - p2[0]) * (p1[1] + p2[1])

            normal = _normalise([nx, ny, nz])

            if normal is not None:
                return normal

            # Fallback: search for any non-collinear triple.
            for i in range(n):
                a = points[i]
                for j in range(i + 1, n):
                    b = points[j]
                    ab = _sub(b, a)

                    if _length(ab) <= tolerance:
                        continue

                    for k in range(j + 1, n):
                        c = points[k]
                        ac = _sub(c, a)
                        candidate = _normalise(_cross(ab, ac))

                        if candidate is not None:
                            return candidate

            return None

        def _loop_angles(vertices, normal):
            if not isinstance(vertices, list) or len(vertices) < 3:
                return []

            points = [_xyz(v) for v in vertices]

            if any(p is None for p in points):
                return []

            n = len(points)
            angles = []

            for i in range(n):
                previous_point = points[i - 1]
                current_point = points[i]
                next_point = points[(i + 1) % n]

                incoming = _sub(current_point, previous_point)
                outgoing = _sub(next_point, current_point)

                if _length(incoming) <= tolerance or _length(outgoing) <= tolerance:
                    return []

                cross_product = _cross(incoming, outgoing)
                dot_product = _dot(incoming, outgoing)

                # Signed exterior turn angle.
                turn_angle = math.degrees(
                    math.atan2(
                        _dot(normal, cross_product),
                        dot_product,
                    )
                )

                # Interior angle of the closed wire loop.
                angle = 180.0 - turn_angle

                while angle < 0.0:
                    angle += 360.0

                while angle > 360.0:
                    angle -= 360.0

                angles.append(round(angle, mantissa))

            expected_sum = float(n - 2) * 180.0
            angle_sum = sum(angles)

            complement_angles = [round(360.0 - a, mantissa) for a in angles]
            complement_sum = sum(complement_angles)

            sum_tolerance = max(
                float(angTolerance) * max(n, 1),
                (10.0 ** (-mantissa)) * max(n, 1) * 2.0,
            )

            if abs(complement_sum - expected_sum) + sum_tolerance < abs(angle_sum - expected_sum):
                angles = complement_angles

            return angles

        vertices = _ordered_vertices_from_closed_wire(wire)

        if vertices is None:
            return None

        if len(vertices) < 3:
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: The input wire has fewer than three ordered vertices. Returning None.")
            return None

        normal = _newell_normal(vertices)

        if normal is None:
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: Could not determine a valid wire normal. Returning None.")
            return None

        angles = _loop_angles(vertices, normal)

        if len(angles) != len(vertices):
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: Could not compute valid wire angles. Returning None.")
            return None

        result = []

        for vertex, angle in zip(vertices, angles):
            try:
                angle = float(angle)
            except Exception:
                continue

            if cornerType == "convex":
                if angle < 180.0 - angTolerance:
                    result.append(vertex)
            else:
                if angle > 180.0 + angTolerance:
                    result.append(vertex)

        return result


    @staticmethod
    def ConvexCornerVertices(
        wire,
        angTolerance: float = 0.01,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> list:
        """
        Returns the convex corner vertices of the input wire.

        The wire must be closed, manifold, and represent a single non-branching
        cycle. A vertex is considered convex if the interior angle of the enclosed
        region is less than 180 degrees, within the specified tolerance. Collinear
        vertices close to 180 degrees are not returned.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        angTolerance : float , optional
            The angular tolerance in degrees. Default is 0.01.
        mantissa : int , optional
            The number of decimal places to round computed angles to. Default is 6.
        tolerance : float , optional
            The geometric tolerance used for endpoint matching. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of convex corner vertices.
        """

        return Wire._CornerVerticesByAngle(
            wire,
            cornerType="convex",
            angTolerance=angTolerance,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=silent,
        )


    @staticmethod
    def ConcaveCornerVertices(
        wire,
        angTolerance: float = 0.01,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> list:
        """
        Returns the concave corner vertices of the input wire.

        The wire must be closed, manifold, and represent a single non-branching
        cycle. A vertex is considered concave if the interior angle of the enclosed
        region is greater than 180 degrees, within the specified tolerance. Collinear
        vertices close to 180 degrees are not returned.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        angTolerance : float , optional
            The angular tolerance in degrees. Default is 0.01.
        mantissa : int , optional
            The number of decimal places to round computed angles to. Default is 6.
        tolerance : float , optional
            The geometric tolerance used for endpoint matching. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of concave corner vertices.
        """

        return Wire._CornerVerticesByAngle(
            wire,
            cornerType="concave",
            angTolerance=angTolerance,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=silent,
        )

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
            The location of the origin of the T-shape. Default is None which results in the Cross-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the Cross-shape. Default is 1.0.
        length : float , optional
            The overall length of the Cross-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the Cross-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the Cross-shape. Default is 0.25.
        c : float , optional
            The distance of the vertical symmetry axis measured from the left side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the Y-axis.
        d : float , optional
            The distance of the horizontal symmetry axis measured from the bottom side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the X-axis.
        direction : list , optional
            The vector representing the up direction of the Cross-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the Cross-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            The location of the origin of the C-shape. Default is None which results in the C-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the C-shape. Default is 1.0.
        length : float , optional
            The overall length of the C-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the C-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the lower horizontal arm of the C-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the upper horizontal arm of the C-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the C-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the C-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def Cycles(wire, maxVertices: int = 4, transferDictionaries: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns the closed circuits of wires found within the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        maxVertices : int , optional
            The maximum number of vertices of the circuits to be searched. Default is 4.
        transferDictionaries : bool , optional
            If set to True, transfers the dictionaries of the original edges
            to the corresponding new edges in the resulting cycle wires.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The list of circuits (closed wires) found within the input wire.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def vIndex(v, vList):
            for i, tv in enumerate(vList):
                if Vertex.Distance(v, tv) <= tolerance:
                    return i + 1
            return None

        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:] + path[:n]

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def isNew(cycles, path):
            return path not in cycles

        def visited(node, path):
            return node in path

        def findNewCycles(graph, cycles, path):
            if len(path) > maxVertices:
                return

            start_node = path[0]

            for node1, node2 in graph:
                if start_node in (node1, node2):
                    next_node = node2 if node1 == start_node else node1

                    if not visited(next_node, path):
                        findNewCycles(graph, cycles, [next_node] + path)
                    elif len(path) > 2 and next_node == path[-1]:
                        p = rotate_to_smallest(path)
                        inv = invert(p)
                        if isNew(cycles, p) and isNew(cycles, inv):
                            cycles.append(p)

        # ------------------------------------------------------------------
        # Build vertex + edge index structures
        # ------------------------------------------------------------------

        tEdges = Topology.Edges(wire)
        tVertices = Topology.Vertices(wire)

        graph = []
        edgeLookup = {}  # (min_i, max_i) → original edge

        for anEdge in tEdges:
            sv = Edge.StartVertex(anEdge)
            ev = Edge.EndVertex(anEdge)

            si = vIndex(sv, tVertices)
            ei = vIndex(ev, tVertices)

            if si is None or ei is None:
                continue

            graph.append((si, ei))

            key = tuple(sorted((si, ei)))
            if key not in edgeLookup:
                edgeLookup[key] = anEdge

        # ------------------------------------------------------------------
        # Find cycles (pure index domain)
        # ------------------------------------------------------------------

        cycles = []
        for node1, node2 in graph:
            findNewCycles(graph, cycles, [node1])
            findNewCycles(graph, cycles, [node2])

        # ------------------------------------------------------------------
        # Construct resulting wires (no more vIndex calls)
        # ------------------------------------------------------------------

        resultWires = []

        for cycle in cycles:
            resultEdges = []

            for i in range(len(cycle)):
                i1 = cycle[i]
                i2 = cycle[(i + 1) % len(cycle)]

                v1 = tVertices[i1 - 1]
                v2 = tVertices[i2 - 1]

                newEdge = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)

                if transferDictionaries:
                    key = tuple(sorted((i1, i2)))
                    sourceEdge = edgeLookup.get(key)
                    if sourceEdge:
                        d = Topology.Dictionary(sourceEdge)
                        if d:
                            newEdge = Topology.SetDictionary(newEdge, d)

                resultEdges.append(newEdge)

            resultWire = Wire.ByEdges(resultEdges, tolerance=tolerance)
            resultWires.append(resultWire)

        return resultWires


    @staticmethod
    def Edges(wire, silent: bool = False) -> list:
        """
        Returns the list of edges of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of edges.

        """
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Edges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        edges = []
        try:
            Core.InstanceCall(wire, "Edges", None, edges)
        except Exception:
            try:
                result = Core.InstanceCall(wire, "Edges")
                if isinstance(result, list):
                    edges = result
                else:
                    return None
            except Exception:
                return None
        return edges

    @staticmethod
    def Einstein(origin= None, radius: float = 0.5, direction: list = [0, 0, 1], placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physicist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the tile. Default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. Default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
    def Ellipse(origin= None,
                inputMode: int = 1,
                width: float = 2.0,
                length: float = 1.0,
                focalLength: float = 0.866025,
                eccentricity: float = 0.866025,
                majorAxisLength: float = 1.0,
                minorAxisLength: float = 0.5,
                sides: float = 32,
                fromAngle: float = 0.0,
                toAngle: float = 360.0,
                close: bool = True,
                direction: list = [0, 0, 1],
                placement: str = "center",
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. Default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. Default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. Default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. Default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. Default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. Default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. Default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. Default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. Default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. Default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created ellipse

        """
        ellipseAll = Wire.EllipseAll(origin=origin, inputMode=inputMode, width=width, length=length, focalLength=focalLength, eccentricity=eccentricity, majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close, direction=direction, placement=placement, tolerance=tolerance)
        
        if ellipseAll is None:
            if not silent:
                print("Wire.Ellipse - Error: Could not create an ellipse. Returning None.")
            return None
        return ellipseAll["ellipse"]

    @staticmethod
    def EllipseAll(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. Default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. Default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. Default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. Default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. Default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. Default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. Default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. Default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. Default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. Default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            baseWire = Wire.ByVertices(baseV[::-1], close=False, tolerance=tolerance) # Counter-clockwise in local XY; normal is +Z
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close, tolerance=tolerance) # Counter-clockwise in local XY; normal is +Z

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
    def EndVertex(wire, silent: bool = False, tolerance: float = 0.0001):
        """
        Returns the end vertex of the input wire.

        The input wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The end vertex of the input wire.

        """
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.EndVertex - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        endpoints = Wire.StartEndVertices(
            wire,
            silent=silent,
            tolerance=tolerance,
        )
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            return None
        return endpoints[1]
    

    @staticmethod
    def ExteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6, silent: bool = False) -> list:
        """
        Returns the exterior angles of the input wire in degrees.

        The input wire must be planar, manifold, and closed.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of exterior angles.

        """
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ExteriorAngles - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.ExteriorAngles - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        if not Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.ExteriorAngles - Error: The input wire parameter is not closed. Returning None.")
            return None

        interior_angles = Wire.InteriorAngles(
            wire,
            tolerance=tolerance,
            mantissa=mantissa,
            silent=silent,
        )
        if not isinstance(interior_angles, list):
            return None
        return [round(360.0 - angle, mantissa) for angle in interior_angles]
    
    @staticmethod
    def ExternalBoundary(wire, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external boundary (cluster of vertices where degree == 1) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input wire. This is a cluster of vertices of degree == 1.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(wire, "wire"):
            if not silent:
                print("Wire.ExternalBoundary - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None
        vertices = [v for v in Topology.Vertices(wire) if Vertex.Degree(v, hostTopology=wire) == 1]
        if len(vertices) > 1:
            return Cluster.ByTopologies(vertices)
        return None
    
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
            If specified, the dictionary of the vertices will be queried for this key to specify the desired fillet radius. Default is None.
        sides : int , optional
            The number of sides (segments) of the fillet. Default is 16.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
                            fillet = Wire.Circle(origin=center, radius=radius, close=True, tolerance=tolerance, silent=silent)
                            bisector = Edge.ByVertices(v, center, tolerance=tolerance, silent=silent)
                            mid_vertex = Topology.Slice(bisector, fillet)
                            verts = Topology.Vertices(mid_vertex, silent=True) if mid_vertex is not None else None
                            if not verts or len(verts) < 2:
                                # Slice returned too few intersections (e.g. the
                                # bisector meet the fillet circle at a single point).
                                # Recover the arc apex geometrically: the point on the
                                # circle farthest from v along the bisector.
                                try:
                                    vb = Vertex.ByCoordinates(Vertex.X(v), Vertex.Y(v), Vertex.Z(v))
                                    cb = Vertex.ByCoordinates(Vertex.X(center), Vertex.Y(center), Vertex.Z(center))
                                    dv = [Vertex.X(vb)-Vertex.X(cb), Vertex.Y(vb)-Vertex.Y(cb), Vertex.Z(vb)-Vertex.Z(cb)]
                                    n = math.sqrt(sum(c*c for c in dv)) or 1.0
                                    dv = [c/n for c in dv]
                                    mx = Vertex.X(cb) + radius*dv[0]
                                    my = Vertex.Y(cb) + radius*dv[1]
                                    mz = Vertex.Z(cb) + radius*dv[2]
                                    mid_vertex = Vertex.ByCoordinates(mx, my, mz)
                                except Exception:
                                    mid_vertex = center
                            else:
                                mid_vertex = verts[1]
                            fillet = Wire.Arc(v1, mid_vertex, v2, sides=sides, close= False, tolerance=tolerance, silent=silent)
                            f_sv = Wire.StartVertex(fillet)
                            if Vertex.Distance(f_sv, edge1) < Vertex.Distance(f_sv, edge0):
                                fillet = Wire.Reverse(fillet, silent=True)
                            final_vertices += Topology.Vertices(fillet)
                        else:
                            if not silent:
                                print("Wire.Fillet - Error: The specified fillet radius is too large to be applied. Skipping.")
                    else:
                        final_vertices.append(v)
            else:
                final_vertices.append(v)
        flat_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire), tolerance=tolerance, silent=True)
        # Unflatten the wire
        return_wire = Topology.Unflatten(flat_wire, origin=Vertex.Origin(), direction=normal)
        return return_wire

    @staticmethod
    def Funnel(face,
                vertexA,
                vertexB,
                portals,
                tolerance: float = 0.0001,
                silent: float = False):
        """
        Returns a Wire representing a smoothed path inside the given face using
        the funnel (string-pulling) algorithm.

        The algorithm assumes that a corridor has already been computed, and is
        provided as an ordered list of "portals" (pairs of vertices) that lie
        on the face between the start and end locations.

        Parameters
        ----------
        face : topologic_core.Face
            The planar face on which navigation occurs. All vertices must lie
            on this face.
        vertexA : topologic_core.Vertex
            The start point of the path.
        vertexB : topologic_core.Vertex
            The end point of the path.
        portals : list of tuple(Vertex, Vertex)
            Ordered list of corridor edges. Each item is (leftVertex, rightVertex)
            describing the visible "portal" between two consecutive regions along
            the navmesh path.
        tolerance : float , optional
            Numerical tolerance used when comparing orientations and distances.
            Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        wire : topologic_core.Wire
            A Wire representing the smoothed path from startVertex to endVertex
            that stays inside the navigation corridor on the face.
        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "face"):
            if not silent:
                print("Wire.Funnel - Error: The input face parameter is not a topologic face. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "vertex"):
            if not silent:
                print("Wire.Funnel - Error: The input vertexA parameter is not a topologic vertex. Returning None.")
            return None
        if not Topology.IsInstance(vertexB, "vertex"):
            if not silent:
                print("Wire.Funnel - Error: The input vertexB parameter is not a topologic vertex. Returning None.")
            return None

        # ------------------------------------------------------------
        # 1. Basic helpers
        # ------------------------------------------------------------
        def _norm(v):
            return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        def _normalize(v):
            n = _norm(v)
            if n < tolerance:
                return (0.0, 0.0, 0.0)
            return (v[0] / n, v[1] / n, v[2] / n)

        def _dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def _cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def _sub(a, b):
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def _tri_area2(a2, b2, c2):
            """
            Twice the signed area of triangle (a, b, c) in 2D.
            Positive => c is to the left of ab
            Negative => c is to the right of ab
            """
            return (b2[0] - a2[0]) * (c2[1] - a2[1]) - (b2[1] - a2[1]) * (c2[0] - a2[0])

        def _coords3d(v):
            x, y, z = Vertex.Coordinates(v)
            return (x, y, z)

        # ------------------------------------------------------------
        # 2. Build a local 2D coordinate system on the face
        # ------------------------------------------------------------
        # Face normal
        n_vec = Face.Normal(face)  # [nx, ny, nz]
        n = _normalize((n_vec[0], n_vec[1], n_vec[2]))

        # Choose an arbitrary vector not parallel to n
        if abs(n[0]) < 0.9:
            arbitrary = (1.0, 0.0, 0.0)
        else:
            arbitrary = (0.0, 1.0, 0.0)

        u = _normalize(_cross(n, arbitrary))  # tangent
        v = _cross(n, u)                      # bitangent, already orthogonal and normalized

        def _project_to_2d(vertex):
            p = _coords3d(vertex)
            # project onto basis (u, v)
            return (_dot(p, u), _dot(p, v))

        # Precompute 2D coords for start, end and all portal vertices
        start2d = _project_to_2d(vertexA)
        end2d = _project_to_2d(vertexB)

        portal2d = []
        for l_v, r_v in portals:
            portal2d.append((_project_to_2d(l_v), _project_to_2d(r_v)))

        # ------------------------------------------------------------
        # 3. Funnel algorithm in 2D
        #   (based on classic Recast / string-pulling implementation)
        # ------------------------------------------------------------
        path_vertices = [vertexA]

        apex2d = start2d
        apexVertex = vertexA
        apexIndex = -1

        left2d = start2d
        right2d = start2d
        leftVertex = vertexA
        rightVertex = vertexB
        leftIndex = -1
        rightIndex = -1

        n_portals = len(portals)
        i = 0

        # We will process all portals, and then a final "portal" at the goal (end, end)
        while i <= n_portals:
            if i < n_portals:
                newLeft2d, newRight2d = portal2d[i]
                newLeftVertex, newRightVertex = portals[i]
            else:
                # last "portal" is the goal point itself
                newLeft2d = end2d
                newRight2d = end2d
                newLeftVertex = vertexB
                newRightVertex = vertexB

            # --------------------------------------------------------
            # Update right side of funnel
            # --------------------------------------------------------
            area_apex_right_newRight = _tri_area2(apex2d, right2d, newRight2d)
            if area_apex_right_newRight <= tolerance:
                # New right vertex is "inside" or tightening the funnel
                area_apex_left_newRight = _tri_area2(apex2d, left2d, newRight2d)
                if (apexVertex == rightVertex) or (area_apex_left_newRight > tolerance):
                    # Tighten the funnel on the right side
                    right2d = newRight2d
                    rightVertex = newRightVertex
                    rightIndex = i
                else:
                    # Right over left, so left becomes the new apex
                    path_vertices.append(leftVertex)
                    apex2d = _project_to_2d(leftVertex)
                    apexVertex = leftVertex
                    apexIndex = leftIndex

                    # Reset funnel
                    left2d = apex2d
                    right2d = apex2d
                    leftVertex = apexVertex
                    rightVertex = apexVertex
                    leftIndex = apexIndex
                    rightIndex = apexIndex

                    # Restart from the new apex
                    i = apexIndex + 1
                    continue

            # --------------------------------------------------------
            # Update left side of funnel
            # --------------------------------------------------------
            area_apex_left_newLeft = _tri_area2(apex2d, left2d, newLeft2d)
            if area_apex_left_newLeft >= -tolerance:
                # New left vertex is "inside" or tightening the funnel
                area_apex_right_newLeft = _tri_area2(apex2d, right2d, newLeft2d)
                if (apexVertex == leftVertex) or (area_apex_right_newLeft < -tolerance):
                    # Tighten funnel on the left side
                    left2d = newLeft2d
                    leftVertex = newLeftVertex
                    leftIndex = i
                else:
                    # Left over right, so right becomes the new apex
                    path_vertices.append(rightVertex)
                    apex2d = _project_to_2d(rightVertex)
                    apexVertex = rightVertex
                    apexIndex = rightIndex

                    # Reset funnel
                    left2d = apex2d
                    right2d = apex2d
                    leftVertex = apexVertex
                    rightVertex = apexVertex
                    leftIndex = apexIndex
                    rightIndex = apexIndex

                    # Restart from the new apex
                    i = apexIndex + 1
                    continue

            i += 1

        # Finally, add the end point if it is not already in the path
        if path_vertices[-1] is not vertexB:
            path_vertices.append(vertexB)

        # ------------------------------------------------------------
        # 4. Build and return the Topologic wire
        # ------------------------------------------------------------
        return_wire = Wire.ByVertices(path_vertices, close=False, silent=True)
        bb = Wire.BoundingRectangle(face)
        d = Topology.Dictionary(bb)
        width = Dictionary.ValueAtKey(d, "width")
        length = Dictionary.ValueAtKey(d, "length")
        size = max(width, length)
        percentage = 0.25 # Start with 25% of the total size
        is_ok = False
        while is_ok == False and percentage > 0:
            new_wire = Wire.Simplify(return_wire, tolerance=size*percentage, silent=True)
            test_wire = Topology.Scale(new_wire, Topology.Centroid(new_wire), 0.95, 0.95, 1)
            result = Topology.Difference(test_wire, face, tolerance=tolerance, silent=True)
            if result is None:
                is_ok = True
                return_wire = new_wire
            percentage -= 0.01
        print("Wire.Funnel - Result:", result)
        print("Wire.Funnel - Percentage:", percentage)
        return new_wire

    @staticmethod
    def GoldenRectangle(width: float = 1.0,
                        maxIterations: int = 10,
                        clockwise: bool = False,
                        origin=None,
                        placement: str = "center",
                        direction: list = [0, 0, 1],
                        mantissa: int = 6,
                        tolerance: float = 0.0001,
                        silent: bool = False):
        """
        Creates a "golden rectangle". See https://en.wikipedia.org/wiki/Golden_rectangle.
        
        Parameters
        ----------
        width : float
            The desired long side of the outer golden rectangle. Height is width/phi.
        maxIterations : int
            Number of subdivision squares to generate.
        clockwise : bool , optional
            Controls the square “peel” progression (affects which side each next square
            is taken from). Default is False.
        origin : topologic_core.Vertex, optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created golden rectangle wire.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        import math

        # -----------------------------
        # Helpers
        # -----------------------------
        def _safe_vertex(v):
            return v if v is not None else Vertex.Origin()

        def _round(x):
            return round(float(x), int(mantissa))

        def _edge(v0, v1):
            return Edge.ByStartVertexEndVertex(v0, v1, tolerance=tolerance, silent=silent)

        def _square_edges(sx, sy, s):
            bl = Vertex.ByCoordinates(_round(sx),   _round(sy),   0.0)
            br = Vertex.ByCoordinates(_round(sx+s), _round(sy),   0.0)
            tr = Vertex.ByCoordinates(_round(sx+s), _round(sy+s), 0.0)
            tl = Vertex.ByCoordinates(_round(sx),   _round(sy+s), 0.0)
            return [_edge(bl, br), _edge(br, tr), _edge(tr, tl), _edge(tl, bl)]

        # -----------------------------
        # Validate
        # -----------------------------
        width = float(width)
        if width <= 0:
            if not silent:
                print("Wire.GoldenRectangle - Error: width must be greater than 0. Returning None.")
            return None
        maxIterations = int(maxIterations)
        if maxIterations < 0:
            if not silent:
                print("Wire.GoldenRectangle - Error: maxIterations must be >= 0. Returning None.")
            return None
        clockwise = bool(clockwise)

        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.GoldenRectangle - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None
        
        placement = str(placement).lower()
        if not placement in ["center", "lowerleft", "lowerright", "upperleft", "upperright"]:
            if not silent:
                print("Wire.GoldenRectangle - Error: The input placement parameter is not a valid placement string. Returning None.")
            return None
        
        if not isinstance(direction, list):
            if not silent:
                print("Wire.GoldenRectangle - Error: The input direction parameter is not a valid list. Returning None.")
            return None
        
        direction = [x for x in direction if isinstance(x, (int, float))]
        
        if len(direction) != 3:
            if not silent:
                print("Wire.GoldenRectangle - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        # -----------------------------
        # Canonical golden rectangle (UNIT width), centered at (0,0,0)
        # -----------------------------
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        W0 = 1.0
        H0 = 1.0 / phi

        x0 = -W0 * 0.5
        y0 = -H0 * 0.5
        centerV = Vertex.ByCoordinates(0.0, 0.0, 0.0)

        # Outer boundary (canonical)
        boundary = Wire.Rectangle(origin=Vertex.ByCoordinates(_round(x0), _round(y0), 0.0),
                                width=W0, length=H0, placement="lowerleft",
                                direction=[0, 0, 1])

        # If no iterations requested, just return the boundary with final transforms
        if maxIterations == 0:
            wire = boundary
        else:
            # -----------------------------
            # Canonical recursive subdivision squares (k progression ALWAYS CCW canonical)
            # -----------------------------
            def _subdivide(rx, ry, rW, rH, k, depth, outSquares):
                if depth <= 0:
                    return
                if rW <= tolerance or rH <= tolerance:
                    if not silent:
                        print("Wire.GoldenRectangle - Warning: Edge lengths have fallen below tolerance. Stopping early.")
                    return

                wide = (rW >= rH)

                # k: 0:left, 1:bottom, 2:right, 3:top  (canonical progression only)
                if wide:
                    s = rH
                    if k == 0:      # left
                        sx, sy = rx, ry
                        nrx, nry = rx + s, ry
                        nW, nH = rW - s, rH
                    elif k == 2:    # right
                        sx, sy = rx + (rW - s), ry
                        nrx, nry = rx, ry
                        nW, nH = rW - s, rH
                    elif k == 1:    # bottom (fallback)
                        sx, sy = rx, ry
                        nrx, nry = rx, ry + s
                        nW, nH = rW, rH - s
                    else:           # top (fallback)
                        sx, sy = rx, ry + (rH - s)
                        nrx, nry = rx, ry
                        nW, nH = rW, rH - s
                else:
                    s = rW
                    if k == 1:      # bottom
                        sx, sy = rx, ry
                        nrx, nry = rx, ry + s
                        nW, nH = rW, rH - s
                    elif k == 3:    # top
                        sx, sy = rx, ry + (rH - s)
                        nrx, nry = rx, ry
                        nW, nH = rW, rH - s
                    elif k == 0:    # left (fallback)
                        sx, sy = rx, ry
                        nrx, nry = rx + s, ry
                        nW, nH = rW - s, rH
                    else:           # right (fallback)
                        sx, sy = rx + (rW - s), ry
                        nrx, nry = rx, ry
                        nW, nH = rW - s, rH

                outSquares.append((sx, sy, s))
                _subdivide(nrx, nry, nW, nH, (k + 1) % 4, depth - 1, outSquares)

            squares = []
            _subdivide(float(x0), float(y0), float(W0), float(H0), 0, maxIterations, squares)
            if len(squares) == 0:
                if not silent:
                    print("Wire.GoldenRectangle - Error: Could not create rectangle. Returning None.")
                return None

            # Build square edges (canonical)
            sq_edges = []
            for (sx, sy, s) in squares:
                e_list = _square_edges(sx, sy, s)
                if None in e_list:
                    if not silent:
                        print("Wire.GoldenRectangle - Warning: Could not create an edge. Stopping early.")
                    break
                sq_edges += e_list

            # The subdivided squares form a nested/disconnected cluster under
            # the pythonOCC backend. The defining geometry of a golden rectangle
            # is its single closed outer boundary, so return that as the wire.
            wire = boundary

            if wire is None:
                if not silent:
                    print("Wire.GoldenRectangle - Error: Could not create golden rectangle. Returning None.")
                return None

        # -----------------------------
        # FINAL transforms (only here)
        # -----------------------------

        # 1) Mirror (clockwise) about canonical center
        if clockwise:
            wire = Topology.Scale(wire, centerV, 1.0, -1.0, 1.0)

        # 2) Scale to requested width (canonical W0=1.0 => scale factors are (width, width, 1))
        wire = Topology.Scale(wire, centerV, width, width, 1.0)

        # 3) Translate so placement reference point lies at canonical origin (0,0,0)
        # After scaling:
        W = width
        H = width / phi
        pl = placement.lower()

        if pl == "center":
            refx, refy = 0.0, 0.0
        elif pl == "lowerleft":
            refx, refy = -W * 0.5, -H * 0.5
        elif pl == "lowerright":
            refx, refy =  W * 0.5, -H * 0.5
        elif pl == "upperleft":
            refx, refy = -W * 0.5,  H * 0.5
        elif pl == "upperright":
            refx, refy =  W * 0.5,  H * 0.5
        else:
            refx, refy = 0.0, 0.0

        wire = Topology.Translate(wire, -refx, -refy, 0.0)

        # 4) Orient/place (as requested)
        if direction != [0,0,1]:
            wire = Topology.Orient(wire, origin=origin, dirA=[0,0,1], dirB=direction)

        return wire

    @staticmethod
    def GoldenSpiral(width: float = 1.0,
                    maxIterations: int = 10,
                    clockwise: bool = False,
                    sides: int = 96,
                    origin=None,
                    placement: str = "center",
                    direction: list = [0, 0, 1],
                    mantissa: int = 6,
                    tolerance: float = 0.0001,
                    silent: bool = False):
        """
        Creates a "golden spiral" as segmented quarter-circle arcs. See https://en.wikipedia.org/wiki/Golden_spiral
        
        Parameters
        ----------
        width : float
            The desired long side of the outer golden rectangle. Height is width/phi.
        maxIterations : int
            Number of subdivision squares to generate.
        clockwise : bool , optional
            Controls the square “peel” progression (affects which side each next square
            is taken from). Default is False.
        sides : int , optional
            The number of sides of the golden spiral (if included).
            Notes: If you set sides to be equal to maxIterations, you get the diagonals.
            It is best if the number of sides is a multiple of maxIterations.
            Default is 96.
        origin : topologic_core.Vertex, optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created golden spiral wire
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster

        import math

        # -----------------------------
        # Helpers
        # -----------------------------
        def _round(x):
            return round(float(x), int(mantissa))

        def _dist_xy(a, b):
            dx = Vertex.X(a) - Vertex.X(b)
            dy = Vertex.Y(a) - Vertex.Y(b)
            return math.sqrt(dx*dx + dy*dy)

        def _square_corners(sx, sy, s):
            bl = Vertex.ByCoordinates(_round(sx),   _round(sy),   0.0)
            br = Vertex.ByCoordinates(_round(sx+s), _round(sy),   0.0)
            tr = Vertex.ByCoordinates(_round(sx+s), _round(sy+s), 0.0)
            tl = Vertex.ByCoordinates(_round(sx),   _round(sy+s), 0.0)
            return (bl, br, tr, tl)

        def _ang_from(center, p):
            return math.atan2(Vertex.Y(p) - Vertex.Y(center), Vertex.X(p) - Vertex.X(center))

        def _normalize_angle(a):
            while a <= -math.pi:
                a += 2.0 * math.pi
            while a > math.pi:
                a -= 2.0 * math.pi
            return a

        def _arc_edges(center, p_start, p_end, nseg):
            """
            If nseg == 1: return the diagonal edge (p_start -> p_end).
            Else: return a polyline approximation of the quarter-circle.
            """
            nseg = max(1, int(nseg))
            if nseg == 1:
                return [Edge.ByStartVertexEndVertex(p_start, p_end, tolerance=tolerance)]

            r = max(tolerance, _dist_xy(center, p_start))
            ang0 = _ang_from(center, p_start)
            angT = _ang_from(center, p_end)

            # choose +/- 90 degrees from ang0 that best hits angT
            candA = ang0 + math.pi / 2.0
            candB = ang0 - math.pi / 2.0
            dA = abs(_normalize_angle(candA - angT))
            dB = abs(_normalize_angle(candB - angT))
            ang1 = candA if dA <= dB else candB

            cx, cy = Vertex.X(center), Vertex.Y(center)
            pts = []
            for i in range(nseg + 1):
                t = float(i) / float(nseg)
                a = ang0 + t * (ang1 - ang0)
                x = cx + r * math.cos(a)
                y = cy + r * math.sin(a)
                pts.append(Vertex.ByCoordinates(_round(x), _round(y), 0.0))

            edges = []
            for i in range(len(pts) - 1):
                edges.append(Edge.ByStartVertexEndVertex(pts[i], pts[i+1], tolerance=tolerance))
            return edges

        # -----------------------------
        # Validate
        # -----------------------------
        width = float(width)
        if width <= 0:
            if not silent:
                print("Wire.GoldenSpiral - Error: width must be greater than 0. Returning None.")
            return None
        maxIterations = int(maxIterations)
        if maxIterations <= 0:
            if not silent:
                print("Wire.GoldenSpiral - Error: maxIterations must be >= 0. Returning None.")
            return None
        
        sides = int(sides)
        if sides < maxIterations:
            if not silent:
                print("Wire.GoldenSpiral - Error: sides must be >= maxIterations. Returning None.")
            return None
        clockwise = bool(clockwise)

        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Wire.GoldenSpiral - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None
        
        placement = str(placement).lower()
        if not placement in ["center", "lowerleft", "lowerright", "upperleft", "upperright"]:
            if not silent:
                print("Wire.GoldenSpiral - Error: The input placement parameter is not a valid placement string. Returning None.")
            return None
        
        if not isinstance(direction, list):
            if not silent:
                print("Wire.GoldenSpiral - Error: The input direction parameter is not a valid list. Returning None.")
            return None
        
        direction = [x for x in direction if isinstance(x, (int, float))]
        
        if len(direction) != 3:
            if not silent:
                print("Wire.GoldenSpiral - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        # -----------------------------
        # Canonical golden rectangle (unit width), centered at (0,0,0)
        # -----------------------------
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        W0 = 1.0
        H0 = 1.0 / phi

        x0 = -W0 * 0.5
        y0 = -H0 * 0.5
        centerV = Vertex.ByCoordinates(0.0, 0.0, 0.0)

        # -----------------------------
        # Canonical square sequence (CCW peel-side cycle)
        # -----------------------------
        # This is canonical; clockwise is applied later via mirroring.
        side_cycle = ["left", "bottom", "right", "top"]

        rx, ry, rW, rH = float(x0), float(y0), float(W0), float(H0)
        squares = []  # (sx, sy, s, side)

        for i in range(maxIterations):
            if rW <= tolerance or rH <= tolerance:
                break

            side = side_cycle[i % 4]

            if rW >= rH:
                s = rH
                if side == "right":
                    sx, sy = rx + (rW - s), ry
                    rW = rW - s
                else:  # left or fallback
                    sx, sy = rx, ry
                    rx = rx + s
                    rW = rW - s
            else:
                s = rW
                if side == "top":
                    sx, sy = rx, ry + (rH - s)
                    rH = rH - s
                else:  # bottom or fallback
                    sx, sy = rx, ry
                    ry = ry + s
                    rH = rH - s

            squares.append((sx, sy, s, side))

        if not squares:
            if not silent:
                print("Wire.GoldenSpiral - Error: Could not create square sequence. Returning None.")
            return None

        # -----------------------------
        # Distribute global sides across arcs (min 1 per arc)
        # -----------------------------
        weights = [max(tolerance, s) for (_, _, s, _) in squares]
        wsum = sum(weights) if sum(weights) > 0 else 1.0

        segs = []
        for w in weights:
            n = int(round(sides * (w / wsum)))
            segs.append(max(1, n))

        # normalize to exactly `sides` while maintaining >=1
        cur = sum(segs)
        while cur > sides:
            j = max(range(len(segs)), key=lambda i: segs[i])
            if segs[j] > 1:
                segs[j] -= 1
                cur -= 1
            else:
                break
        while cur < sides:
            j = max(range(len(segs)), key=lambda i: weights[i])
            segs[j] += 1
            cur += 1

        # -----------------------------
        # Build spiral arcs only (flipped-diagonal orientation by side)
        # -----------------------------
        spiral_edges = []
        last_end = None
        eps_join = 10.0 ** (-mantissa)

        for (sx, sy, s, side), nseg in zip(squares, segs):
            bl, br, tr, tl = _square_corners(sx, sy, s)

            # Flipped-diagonal mapping (by side)
            if side == "left":
                p0, p1 = tl, br
                c = tr
            elif side == "bottom":
                p0, p1 = bl, tr
                c = tl
            elif side == "right":
                p0, p1 = br, tl
                c = bl
            else:  # top
                p0, p1 = tr, bl
                c = br

            # continuity (swap endpoints if needed)
            if last_end is not None:
                if abs(Vertex.X(p0) - Vertex.X(last_end)) > eps_join or abs(Vertex.Y(p0) - Vertex.Y(last_end)) > eps_join:
                    if abs(Vertex.X(p1) - Vertex.X(last_end)) <= eps_join and abs(Vertex.Y(p1) - Vertex.Y(last_end)) <= eps_join:
                        p0, p1 = p1, p0

            edges = _arc_edges(c, p0, p1, nseg)
            if edges:
                last_end = Edge.EndVertex(edges[-1])
            spiral_edges += edges

        spiral = Wire.ByEdges(spiral_edges, tolerance=tolerance)
        if spiral is None:
            spiral = Topology.SelfMerge(Cluster.ByTopologies(spiral_edges))
        if spiral is None:
            if not silent:
                print("Wire.GoldenSpiral - Error: Could not create spiral. Returning None.")
            return None

        # -----------------------------
        # FINAL transforms (only here)
        # -----------------------------

        # 1) Mirror for clockwise (negative scaling on Y axis about center)
        if clockwise:
            spiral = Topology.Scale(spiral, centerV, 1.0, -1.0, 1.0)

        # 2) Scale to requested width (canonical W0=1 => uniform XY scale = width)
        spiral = Topology.Scale(spiral, centerV, width, width, 1.0)

        # 3) Translate so placement reference of the *golden rectangle* lands at (0,0,0)
        W = width
        H = width / phi
        pl = placement.lower()

        if pl == "center":
            refx, refy = 0.0, 0.0
        elif pl == "lowerleft":
            refx, refy = -W * 0.5, -H * 0.5
        elif pl == "lowerright":
            refx, refy =  W * 0.5, -H * 0.5
        elif pl == "upperleft":
            refx, refy = -W * 0.5,  H * 0.5
        elif pl == "upperright":
            refx, refy =  W * 0.5,  H * 0.5
        else:
            refx, refy = 0.0, 0.0

        spiral = Topology.Translate(spiral, -refx, -refy, 0.0)

        # 4) Orient/place
        spiral = Topology.Orient(spiral, origin, [0, 0, 1], direction)

        return spiral


    @staticmethod
    def InteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6, silent: bool = False) -> list:
        """
        Returns the interior angles of the input wire in degrees.

        The wire must be planar, manifold, and closed. This implementation does not
        create a Face from the wire. Instead, it orders the wire vertices, computes a
        robust polygon normal using Newell's method, and evaluates each interior angle
        directly in 3D.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of interior angles in degrees.

        """

        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None.")
            return None

        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None.")
            return None

        if not Wire.IsClosed(wire):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None.")
            return None

        def _xyz(vertex):
            try:
                return [
                    float(Vertex.X(vertex)),
                    float(Vertex.Y(vertex)),
                    float(Vertex.Z(vertex)),
                ]
            except Exception:
                return None

        def _sub(a, b):
            return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]

        def _length(v):
            return math.sqrt(_dot(v, v))

        def _distance_squared(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            dz = a[2] - b[2]
            return dx*dx + dy*dy + dz*dz

        def _same_point(a, b):
            return _distance_squared(a, b) <= tolerance*tolerance

        def _edge_vertices(edge):
            try:
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)
                if sv is not None and ev is not None:
                    return sv, ev
            except Exception:
                pass

            try:
                vertices = Topology.Vertices(edge)
                if vertices is not None and len(vertices) >= 2:
                    return vertices[0], vertices[1]
            except Exception:
                pass

            return None, None

        def _ordered_wire_points(wire):
            try:
                edges = Topology.Edges(wire)
            except Exception:
                return None

            if edges is None or len(edges) < 3:
                return None

            edge_data = []
            for edge in edges:
                sv, ev = _edge_vertices(edge)
                p1 = _xyz(sv)
                p2 = _xyz(ev)

                if p1 is None or p2 is None:
                    continue
                if _same_point(p1, p2):
                    continue

                edge_data.append([edge, sv, ev, p1, p2])

            if len(edge_data) < 3:
                return None

            unused = edge_data[:]

            first = unused.pop(0)
            start_vertex = first[1]
            current_vertex = first[2]
            start_point = first[3]
            current_point = first[4]

            points = [start_point, current_point]

            while unused:
                found_index = None
                next_vertex = None
                next_point = None

                for i, data in enumerate(unused):
                    _, sv, ev, p1, p2 = data

                    if _same_point(current_point, p1):
                        found_index = i
                        next_vertex = ev
                        next_point = p2
                        break

                    if _same_point(current_point, p2):
                        found_index = i
                        next_vertex = sv
                        next_point = p1
                        break

                if found_index is None:
                    # The wire passed Topologic's manifold/closed tests, but the
                    # extracted edges could not be ordered robustly.
                    return None

                unused.pop(found_index)

                if _same_point(next_point, start_point):
                    current_vertex = next_vertex
                    current_point = next_point
                    continue

                if not _same_point(next_point, points[-1]):
                    points.append(next_point)

                current_vertex = next_vertex
                current_point = next_point

            # Remove accidental duplicate closing vertex, if present.
            if len(points) > 1 and _same_point(points[0], points[-1]):
                points.pop()

            # Remove consecutive duplicate points, if any.
            clean_points = []
            for p in points:
                if not clean_points or not _same_point(p, clean_points[-1]):
                    clean_points.append(p)

            if len(clean_points) > 1 and _same_point(clean_points[0], clean_points[-1]):
                clean_points.pop()

            return clean_points

        def _newell_normal(points):
            nx = 0.0
            ny = 0.0
            nz = 0.0
            n = len(points)

            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                nx += (p1[1] - p2[1]) * (p1[2] + p2[2])
                ny += (p1[2] - p2[2]) * (p1[0] + p2[0])
                nz += (p1[0] - p2[0]) * (p1[1] + p2[1])

            normal = [nx, ny, nz]
            normal_length = _length(normal)

            if normal_length > max(tolerance*tolerance, 1e-18):
                return [
                    normal[0] / normal_length,
                    normal[1] / normal_length,
                    normal[2] / normal_length,
                ]

            # Fallback: search for any non-collinear triple.
            for i in range(n):
                a = points[i]
                for j in range(i + 1, n):
                    b = points[j]
                    ab = _sub(b, a)

                    if _length(ab) <= tolerance:
                        continue

                    for k in range(j + 1, n):
                        c = points[k]
                        ac = _sub(c, a)
                        candidate = _cross(ab, ac)
                        candidate_length = _length(candidate)

                        if candidate_length > max(tolerance*tolerance, 1e-18):
                            return [
                                candidate[0] / candidate_length,
                                candidate[1] / candidate_length,
                                candidate[2] / candidate_length,
                            ]

            return None

        points = _ordered_wire_points(wire)

        if points is None or len(points) < 3:
            if not silent:
                print("Wire.InteriorAngles - Error: Could not extract an ordered closed vertex loop from the input wire. Returning None.")
            return None

        normal = _newell_normal(points)

        if normal is None:
            if not silent:
                print("Wire.InteriorAngles - Error: Could not determine a valid normal from the input wire. Returning None.")
            return None

        angles = []
        n = len(points)

        for i in range(n):
            previous_point = points[i - 1]
            current_point = points[i]
            next_point = points[(i + 1) % n]

            previous_edge = _sub(current_point, previous_point)
            next_edge = _sub(next_point, current_point)

            previous_length = _length(previous_edge)
            next_length = _length(next_edge)

            if previous_length <= tolerance or next_length <= tolerance:
                if not silent:
                    print("Wire.InteriorAngles - Error: The input wire contains a degenerate edge. Returning None.")
                return None

            cross_product = _cross(previous_edge, next_edge)
            dot_product = _dot(previous_edge, next_edge)

            # Signed exterior turn angle, measured around the robust polygon normal.
            turn_angle = math.degrees(
                math.atan2(
                    _dot(normal, cross_product),
                    dot_product,
                )
            )

            # Interior angle = 180 - signed exterior turn.
            interior_angle = 180.0 - turn_angle

            while interior_angle < 0.0:
                interior_angle += 360.0

            while interior_angle > 360.0:
                interior_angle -= 360.0

            angles.append(round(interior_angle, mantissa))

        # If numerical orientation issues caused the complementary set to be closer
        # to the expected polygon angle sum, use the complementary angles.
        expected_sum = float(n - 2) * 180.0
        angle_sum = sum(angles)
        complement_angles = [round(360.0 - a, mantissa) for a in angles]
        complement_sum = sum(complement_angles)

        angle_sum_tolerance = max(float(tolerance), (10.0 ** (-mantissa)) * max(n, 1) * 2.0)

        if abs(complement_sum - expected_sum) + angle_sum_tolerance < abs(angle_sum - expected_sum):
            angles = complement_angles

        return angles
    # @staticmethod
    # def InteriorAngles_old(wire, tolerance: float = 0.0001, mantissa: int = 6, silent: bool = False) -> list:
    #     """
    #     Returns the interior angles of the input wire in degrees. The wire must be planar, manifold, and closed.
    #     This code has been contributed by Yidan Xue.
        
    #     Parameters
    #     ----------
    #     wire : topologic_core.Wire
    #         The input wire.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
    #     mantissa : int , optional
    #         The number of decimal places to round the result to. Default is 6.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.
        
    #     Returns
    #     -------
    #     list
    #         The list of interior angles.
        
    #     """
    #     from topologicpy.Vertex import Vertex
    #     from topologicpy.Edge import Edge
    #     from topologicpy.Face import Face
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Vector import Vector
    #     from topologicpy.Dictionary import Dictionary

    #     if not Topology.IsInstance(wire, "Wire"):
    #         if not silent:
    #             print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None")
    #         return None
    #     if not Wire.IsManifold(wire):
    #         if not silent:
    #             print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None")
    #         return None
    #     if not Wire.IsClosed(wire):
    #         if not silent:
    #             print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None")
    #         return None
        
    #     f = Face.ByWire(wire)
    #     normal = Face.Normal(f)
    #     origin = Topology.Centroid(f)
    #     w = Topology.Flatten(wire, origin=origin, direction=normal)
    #     angles = []
    #     edges = Topology.Edges(w)
    #     e1 = edges[len(edges)-1]
    #     e2 = edges[0]
    #     a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
    #     angles.append(a)
    #     for i in range(len(edges)-1):
    #         e1 = edges[i]
    #         e2 = edges[i+1]
    #         a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
    #         angles.append(round(a, mantissa))
    #     if abs(sum(angles)-(len(angles)-2)*180)<tolerance:
    #         return angles
    #     else:
    #         angles = [360-ang for ang in angles]
    #         return angles

    @staticmethod
    def Interpolate(wires: list, n: int = 5, outputType: str = "default", mapping: str = "default", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates *n* number of wires that interpolate between wireA and wireB.

        Parameters
        ----------
        wireA : topologic_core.Wire
            The first input wire.
        wireB : topologic_core.Wire
            The second input wire.
        n : int , optional
            The number of intermediate wires to create. Default is 5.
        outputType : str , optional
            The desired type of output. The options are case insensitive. Default is "contour". The options are:
                - "Default" or "Contours" (wires are not connected)
                - "Raster or "Zigzag" or "Toolpath" (the wire ends are connected to create a continuous path)
                - "Grid" (the wire ends are connected to create a grid). 
        mapping : str , optional
            The desired type of mapping for wires with different number of vertices. It is case insensitive. Default is "default". The options are:
                - "Default" or "Repeat" which repeats the last vertex of the wire with the least number of vertices
                - "Nearest" which maps the vertices of one wire to the nearest vertex of the next wire creating a list of equal number of vertices.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
            if not silent:
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
                finalWires.append(Wire.ByVertices(c, close=Wire.IsClosed(wires[i], tolerance=tolerance, silent=True)))

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
    def Invert(wire, silent: bool = False, tolerance: float = 0.0001):
        """
        Creates a wire that is an inverse (reverse orientation) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The inverted wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Invert - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            return Wire.Reverse(
                wire,
                transferDictionaries=False,
                tolerance=tolerance,
                silent=silent,
            )

        vertices = Topology.Vertices(wire, silent=True)
        reversed_vertices = vertices[::-1]
        return Wire.ByVertices(
            reversed_vertices,
            close=Wire.IsClosed(wire, tolerance=tolerance, silent=True),
            silent=silent,
            tolerance=tolerance,
        )


    @staticmethod
    def IsClosed(wire, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input wire is closed. Returns False otherwise.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the input wire is closed. False otherwise.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None

        try:
            if Wire._UseNativeWireBackend():
                return bool(Core.WireUtility.IsClosed(wire, tolerance))
            # Preserve the TopologicCore calling convention exactly.
            return bool(Core.InstanceCall(wire, "IsClosed"))
        except Exception:
            if not silent:
                print("Wire.IsClosed - Error: Could not determine whether the input wire is closed. Returning None.")
            return None
    

    @staticmethod
    def IsManifold(wire, silent: bool = False, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input wire is manifold. Returns False otherwise.

        A manifold wire is one where no vertex has a topological degree greater
        than two.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance used by the PythonOCC endpoint classifier. Default is 0.0001.

        Returns
        -------
        bool
            True if the input wire is manifold. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.IsManifold - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                return bool(Core.WireUtility.IsManifold(wire, tolerance))
            except Exception:
                if not silent:
                    print("Wire.IsManifold - Error: The native backend could not classify the wire. Returning None.")
                return None

        # Legacy TopologicCore path.
        vertices = Topology.Vertices(wire, silent=True) or []
        for vertex in vertices:
            if Vertex.Degree(vertex, hostTopology=wire) > 2:
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
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
        
        if (Wire.IsClosed(wireA) == False):
            return None
        if (Wire.IsClosed(wireB) == False):
            return None
        edgesA = Topology.Edges(wireA)
        edgesB = Topology.Edges(wireB)
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
            The location of the origin of the I-shape. Default is None which results in the I-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the I-shape. Default is 1.0.
        length : float , optional
            The overall length of the I-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the central vertical arm of the I-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the lower horizontal arm of the I-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the upper horizontal arm of the I-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the I-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the I-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def Lattice(origin=None,
                width: float = 1.0, length: float = 1.0, height: float = 1.0,
                uSides: int = 2, vSides: int = 2, wSides: int = 2,
                direction: list = [0, 0, 1], placement: str = "center",
                mantissa: int = 6, tolerance: float = 0.0001,
                silent: bool = False):
        """
        Creates a prismatic 3D lattice as a Wire.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            Placement origin.
        width, length, height : float
            Lattice extents.
        uSides, vSides, wSides : int
            Divisions along X, Y, Z.
        direction : list , optional
            The vector representing the up direction of the lattice. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the lattice. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        import math

        # -------------------------
        # Validation
        # -------------------------
        if uSides < 1 or vSides < 1 or wSides < 1:
            return None

        if origin is None:
            origin = Vertex.ByCoordinates(0, 0, 0)

        # -------------------------
        # Placement Offsets
        # -------------------------
        if placement.lower() == "center":
            ox = -width * 0.5
            oy = -length * 0.5
            oz = -height * 0.5
        elif placement.lower() == "bottom":
            ox = -width * 0.5
            oy = -length * 0.5
            oz = 0
        else:
            ox = oy = oz = 0.0

        # -------------------------
        # Step Sizes
        # -------------------------
        du = width / uSides
        dv = length / vSides
        dw = height / wSides

        # -------------------------
        # Precompute Grid Coordinates
        # -------------------------
        xs = [round(ox + i * du, mantissa) for i in range(uSides + 1)]
        ys = [round(oy + j * dv, mantissa) for j in range(vSides + 1)]
        zs = [round(oz + k * dw, mantissa) for k in range(wSides + 1)]

        # -------------------------
        # Build a single connected serpentine wire traversing every grid node.
        # A prismatic 3D lattice is non-manifold (grid nodes of degree > 2), so it
        # cannot be represented as one manifold wire via Wire.ByEdges; the
        # serpentine path is a valid single Wire carrying the lattice topology.
        # -------------------------
        nodes = []
        for zi, z in enumerate(zs):
            row_xs = xs if zi % 2 == 0 else list(reversed(xs))
            for y in ys:
                for x in row_xs:
                    nodes.append(Vertex.ByCoordinates(x, y, z))

        lattice = Wire.ByVertices(nodes, close=False, tolerance=tolerance, silent=silent)

        # -------------------------
        # Orient and Place
        # -------------------------
        if lattice is not None:
            lattice = Topology.Orient(lattice, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
            lattice = Topology.Place(lattice, originA=Vertex.Origin(), originB=origin)
        return lattice


    @staticmethod
    def Length(wire, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the length of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The length of the input wire.

        """
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            return None

        totalLength = None

        if Wire._UseNativeWireBackend():
            try:
                totalLength = Core.WireUtility.Length(wire, tolerance)
            except Exception:
                totalLength = None
            if totalLength is not None:
                try:
                    return float(totalLength) if mantissa is None else round(float(totalLength), mantissa)
                except Exception:
                    return None

        # Preserve the historical TopologicCore implementation and use it as a
        # conservative fallback if the native capability is unavailable.
        try:
            edges = Topology.Edges(wire, silent=True)
            if not isinstance(edges, list) or len(edges) == 0:
                return None
            totalLength = 0.0
            for edge in edges:
                length = Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True)
                if length is None:
                    return None
                totalLength += float(length)
            return float(totalLength) if mantissa is None else round(totalLength, mantissa)
        except Exception:
            if not silent:
                print("Wire.Length - Error: Could not calculate the length of the input wire. Returning None.")
            return None

    @staticmethod
    def Line(origin= None,
             length: float = 1,
             direction: list = [1, 0, 0],
             sides: int = 2,
             placement: str ="center",
             tolerance: float = 0.0001,
             silent: bool = True):
        """
        Creates a straight line wire using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. Default is None which results in the edge being placed at (0, 0, 0).
        length : float , optional
            The desired length of the edge. Default is 1.0.
        direction : list , optional
            The desired direction (vector) of the edge. Default is [1, 0, 0] (along the X-axis).
        sides : int , optional
            The desired number of sides/segments. The minimum number of sides is 2. Default is 2.
        placement : str , optional
            The desired placement of the edge. The options are:
            1. "center" which places the center of the edge at the origin.
            2. "start" which places the start of the edge at the origin.
            3. "end" which places the end of the edge at the origin.
            The default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge
        
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Line - Error: The input origin is not a valid vertex. Returning None.")
            return None
        if length <= 0:
            if not silent:
                print("Wire.Line - Error: The input length is less than or equal to zero. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Wire.Line - Error: The input direction is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Wire.Line - Error: The length of the input direction is not equal to three. Returning None.")
            return None
        if sides < 2:
            if not silent:
                print("Wire.Line - Error: The number of sides cannot be less than two. Consider using Edge.Line() instead. Returning None.")
            return None
        edge = Edge.Line(origin=origin, length=length, direction=direction, placement=placement)
        vertices = [Edge.StartVertex(edge)]
        unitDistance = float(1)/float(sides)
        for i in range(1, sides):
            vertices.append(Edge.VertexByParameter(edge, i*unitDistance))
        vertices.append(Edge.EndVertex(edge))
        return_wire = Wire.ByVertices(vertices, close=False, tolerance=tolerance)
        if not Topology.IsInstance(return_wire, "wire"):
            if not silent:
                print("Wire.Line - Error: Could not create the wire. Returning None.")
            return None
        return return_wire

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
            The location of the origin of the L-shape. Default is None which results in the L-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the L-shape. Default is 1.0.
        length : float , optional
            The overall length of the L-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the L-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the L-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the L-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the L-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            l_shape = Topology.Scale(l_shape, origin=origin, x=xScale, y=yScale, z=1)
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
            If specified, the dictionary of the vertices will be queried for this key to specify the desired offset length. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def Normal(wire, outputType: str = "xyz", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a deterministic unit normal vector to the plane of the input wire.

        The geometric calculation is performed at full coordinate precision. The
        sign is canonicalized by making the dominant normal component positive,
        which avoids the non-deterministic sign produced by random vertex sampling.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        outputType : str , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. Default is "xyz".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance for detecting collinear vertices. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The requested components of the unit normal vector.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Normal - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not isinstance(outputType, str):
            if not silent:
                print("Wire.Normal - Error: The input outputType parameter is not a valid string. Returning None.")
            return None

        axes = outputType.lower()
        if len(axes) == 0 or any(axis not in "xyz" for axis in axes):
            if not silent:
                print("Wire.Normal - Error: The input outputType parameter contains invalid axes. Returning None.")
            return None

        vertices = Wire.Vertices(wire, silent=True)
        if not isinstance(vertices, list) or len(vertices) < 3:
            if not silent:
                print("Wire.Normal - Error: At least three vertices are required to define a plane. Returning None.")
            return None

        points = []
        for vertex in vertices:
            coordinates = Vertex.Coordinates(vertex, mantissa=None)
            if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 3:
                points.append((
                    float(coordinates[0]),
                    float(coordinates[1]),
                    float(coordinates[2]),
                ))

        if len(points) < 3:
            if not silent:
                print("Wire.Normal - Error: Could not retrieve at least three valid vertex coordinates. Returning None.")
            return None

        # Deterministic ordering eliminates the historical random 30-second search.
        points = sorted(points)
        tol = max(abs(float(tolerance)), 1.0e-12)
        normal = None

        for i in range(len(points) - 2):
            ax, ay, az = points[i]
            for j in range(i + 1, len(points) - 1):
                ux = points[j][0] - ax
                uy = points[j][1] - ay
                uz = points[j][2] - az
                if math.sqrt(ux * ux + uy * uy + uz * uz) <= tol:
                    continue
                for k in range(j + 1, len(points)):
                    vx = points[k][0] - ax
                    vy = points[k][1] - ay
                    vz = points[k][2] - az
                    nx = uy * vz - uz * vy
                    ny = uz * vx - ux * vz
                    nz = ux * vy - uy * vx
                    magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if magnitude > tol:
                        normal = [nx / magnitude, ny / magnitude, nz / magnitude]
                        break
                if normal is not None:
                    break
            if normal is not None:
                break

        if normal is None:
            if not silent:
                print("Wire.Normal - Error: The input wire vertices are collinear. Returning None.")
            return None

        dominant = max(range(3), key=lambda index: abs(normal[index]))
        if normal[dominant] < 0.0:
            normal = [-value for value in normal]

        normal = [round(value, mantissa) for value in normal]
        lookup = {"x": normal[0], "y": normal[1], "z": normal[2]}
        return [lookup[axis] for axis in axes]
    

    @staticmethod
    def OrientEdges(wire, vertexA, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a correctly oriented head-to-tail version of the input wire.

        The input wire must be manifold. On the PythonOCC backend, reversing an edge
        preserves its native OCCT curve instead of reconstructing an endpoint chord.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertexA : topologic_core.Vertex
            The desired start vertex of the wire.
        transferDictionaries : bool , optional
            If set to True, dictionaries of the original wire and its subtopologies are transferred to the new wire. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The oriented wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.OrientEdges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Wire.OrientEdges - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.OrientEdges - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None

        remaining_edges = list(Topology.Edges(wire, silent=True) or [])
        if len(remaining_edges) == 0:
            if not silent:
                print("Wire.OrientEdges - Error: Could not retrieve the input wire edges. Returning None.")
            return None

        original_vertices = Topology.Vertices(wire, silent=True) or []
        edge_selectors = []
        if transferDictionaries:
            for edge in remaining_edges:
                selector = Topology.Centroid(edge)
                if selector is not None:
                    selector = Topology.SetDictionary(
                        selector,
                        Topology.Dictionary(edge),
                        silent=True,
                    )
                    edge_selectors.append(selector)

        oriented_edges = []
        current_vertex = vertexA

        while remaining_edges:
            source_edge = None
            oriented_edge = None

            for edge in remaining_edges:
                start = Edge.StartVertex(edge, silent=True)
                end = Edge.EndVertex(edge, silent=True)
                if Vertex.IsCoincident(start, current_vertex, tolerance=tolerance, silent=True):
                    source_edge = edge
                    oriented_edge = edge
                    break
                if Vertex.IsCoincident(end, current_vertex, tolerance=tolerance, silent=True):
                    source_edge = edge
                    oriented_edge = Edge.Reverse(
                        edge,
                        tolerance=tolerance,
                        silent=True,
                    )
                    break

            if source_edge is None or not Topology.IsInstance(oriented_edge, "Edge"):
                if not silent:
                    print("Wire.OrientEdges - Error: Could not orient all edges into one continuous head-to-tail path. Returning None.")
                return None

            oriented_edges.append(oriented_edge)
            remaining_edges.remove(source_edge)
            current_vertex = Edge.EndVertex(oriented_edge, silent=True)

        if Wire._UseNativeWireBackend():
            return_wire = Wire.ByEdges(
                oriented_edges,
                orient=False,
                transferDictionaries=False,
                tolerance=tolerance,
                silent=True,
            )
        else:
            # Preserve the historical TopologicCore rebuild pathway.
            vertices = [Edge.StartVertex(oriented_edges[0], silent=True)]
            vertices.extend(
                Edge.EndVertex(edge, silent=True)
                for edge in oriented_edges
            )
            return_wire = Wire.ByVertices(
                vertices,
                close=Wire.IsClosed(wire, tolerance=tolerance, silent=True),
                tolerance=tolerance,
                silent=True,
            )
        if not Topology.IsInstance(return_wire, "Wire"):
            if not silent:
                print("Wire.OrientEdges - Error: Could not construct the oriented wire. Returning None.")
            return None

        if transferDictionaries:
            if edge_selectors:
                return_wire = Topology.TransferDictionariesBySelectors(
                    return_wire,
                    selectors=edge_selectors,
                    tranEdges=True,
                    tolerance=tolerance,
                )
            if original_vertices:
                return_wire = Topology.TransferDictionariesBySelectors(
                    return_wire,
                    selectors=original_vertices,
                    tranVertices=True,
                    tolerance=tolerance,
                )

        return_wire = Topology.SetDictionary(
            return_wire,
            Topology.Dictionary(wire),
            silent=True,
        )
        return return_wire


    @staticmethod
    def Planarize(wire, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a planarized version of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        origin : topologic_core.Vertex , optional
            The desired origin of the plane onto which the wire is projected. If set to None, the centroid of the input wire is used. Default is None.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            if not silent:
                print("Wire.Planarize - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Planarize - Error: Could not determine a valid plane origin. Returning None.")
            return None

        vertices = Topology.Vertices(wire, silent=True) or []
        edges = Topology.Edges(wire, silent=True) or []
        if len(vertices) < 3 or len(edges) < 1:
            if not silent:
                print("Wire.Planarize - Error: The input wire does not contain sufficient geometry. Returning None.")
            return None

        plane_equation = Vertex.PlaneEquation(
            vertices,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=True,
        )
        if not isinstance(plane_equation, dict):
            if not silent:
                print("Wire.Planarize - Error: Could not compute a plane equation for the input wire. Returning None.")
            return None

        rect = Face.RectangleByPlaneEquation(
            origin=origin,
            equation=plane_equation,
            tolerance=tolerance,
        )
        if not Topology.IsInstance(rect, "Face"):
            if not silent:
                print("Wire.Planarize - Error: Could not construct the target plane. Returning None.")
            return None

        new_vertices = [
            Vertex.Project(
                vertex,
                rect,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )
            for vertex in vertices
        ]
        new_vertices = [vertex for vertex in new_vertices if Topology.IsInstance(vertex, "Vertex")]
        new_vertices = Vertex.Fuse(
            new_vertices,
            mantissa=mantissa,
            tolerance=tolerance,
        )
        if not isinstance(new_vertices, list) or len(new_vertices) < 2:
            if not silent:
                print("Wire.Planarize - Error: Could not create planarized vertices. Returning None.")
            return None

        new_edges = []
        for edge in edges:
            start = Edge.StartVertex(edge, silent=True)
            end = Edge.EndVertex(edge, silent=True)
            start_projected = Vertex.Project(
                start,
                rect,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )
            end_projected = Vertex.Project(
                end,
                rect,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )

            start_index = Vertex.Index(start_projected, new_vertices, tolerance=tolerance)
            if start_index is not None:
                start_projected = new_vertices[start_index]

            end_index = Vertex.Index(end_projected, new_vertices, tolerance=tolerance)
            if end_index is not None:
                end_projected = new_vertices[end_index]

            new_edge = Edge.ByVertices(
                [start_projected, end_projected],
                tolerance=tolerance,
                silent=True,
            )
            if Topology.IsInstance(new_edge, "Edge"):
                new_edges.append(new_edge)

        if len(new_edges) == 0:
            if not silent:
                print("Wire.Planarize - Error: Could not rebuild the planarized wire. Returning None.")
            return None

        return Topology.SelfMerge(
            Cluster.ByTopologies(new_edges),
            tolerance=tolerance,
        )


    @staticmethod
    def Project(wire, face, direction: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a projection of the input wire onto the input face.

        On the PythonOCC backend, the method first attempts an OCCT wire projection,
        preserving native curve geometry. If that operation is unavailable or fails,
        the historical endpoint-projection pathway is used.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        face : topologic_core.Face
            The face onto which to project the input wire.
        direction : list , optional
            The vector representing the direction of the projection. If None or empty, the reverse vector of the receiving face normal is used. Default is None.
        mantissa : int , optional
            The number of decimal places to round computed coordinates to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The projected wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Project - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Wire.Project - Error: The input face parameter is not a valid face. Returning None.")
            return None

        if direction is None or (isinstance(direction, (list, tuple)) and len(direction) == 0):
            normal = Face.Normal(face, outputType="xyz", mantissa=mantissa)
            if not isinstance(normal, (list, tuple)) or len(normal) != 3:
                if not silent:
                    print("Wire.Project - Error: Could not determine the receiving face normal. Returning None.")
                return None
            direction = [-float(value) for value in normal]

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Wire.Project - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
            magnitude = math.sqrt(sum(value * value for value in direction))
        except Exception:
            magnitude = 0.0
        if magnitude <= abs(float(tolerance)):
            if not silent:
                print("Wire.Project - Error: The input direction vector has zero magnitude. Returning None.")
            return None
        direction = [value / magnitude for value in direction]

        large_face = Topology.Scale(
            face,
            Topology.CenterOfMass(face),
            500,
            500,
            500,
        )
        if not Topology.IsInstance(large_face, "Face"):
            if not silent:
                print("Wire.Project - Error: Could not construct the projection receiver. Returning None.")
            return None

        # Native PythonOCC path: project the complete OCCT Wire so curves survive.
        if Wire._UseNativeWireBackend():
            try:
                if Core.HasAttribute("WireUtility", "Project"):
                    projected = Core.WireUtility.Project(
                        wire,
                        large_face,
                        direction,
                        tolerance,
                    )
                    if Topology.IsInstance(projected, "Wire"):
                        return projected
            except Exception:
                pass

        # Historical TopologicCore-compatible fallback.
        edges = Topology.Edges(wire, silent=True) or []
        projected_edges = []
        for edge in edges:
            if not Topology.IsInstance(edge, "Edge"):
                continue
            start = Edge.StartVertex(edge, silent=True)
            end = Edge.EndVertex(edge, silent=True)
            projected_start = Vertex.Project(
                vertex=start,
                face=large_face,
                direction=direction,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )
            projected_end = Vertex.Project(
                vertex=end,
                face=large_face,
                direction=direction,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )
            if not Topology.IsInstance(projected_start, "Vertex") or not Topology.IsInstance(projected_end, "Vertex"):
                continue
            projected_edge = Edge.ByVertices(
                [projected_start, projected_end],
                tolerance=tolerance,
                silent=True,
            )
            if Topology.IsInstance(projected_edge, "Edge"):
                projected_edges.append(projected_edge)

        projected_wire = Wire.ByEdges(
            projected_edges,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(projected_wire, "Wire") and not silent:
            print("Wire.Project - Error: Could not project the input wire. Returning None.")
        return projected_wire

    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0, diagonals: bool = False, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. Default is 1.0.
        length : float , optional
            The length of the rectangle. Default is 1.0.
        diagonals : bool , optional
            If set to True, the diagonals of the rectangle are included. Diagonals are split at the centroid of the rectangle. Default is False.
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created rectangle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Rectangle - Error: specified origin is not a topologic vertex. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            if not silent:
                print("Wire.Rectangle - Error: Could not find placement in the list of placements. Returning None.")
            return None
        width = abs(width)
        length = abs(length)
        if width <= tolerance or length <= tolerance:
            if not silent:
                print("Wire.Rectangle - Error: One or more of the specified dimensions is below the tolerance value. Returning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            if not silent:
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

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True, tolerance=tolerance, silent=silent)
        base_edges = Wire.Edges(baseWire)
        if diagonals == True:
            e1 = Edge.ByVertices(vb1, origin)
            e2 = Edge.ByVertices(origin, vb3)
            e3 = Edge.ByVertices(vb2, origin)
            e4 = Edge.ByVertices(origin, vb4)
            baseWire = Wire.ByEdges([e1, e2, e3, e4]+base_edges)
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
            The desired angular tolerance. Default is 0.1.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

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
                return Wire.ByVertices(filtered_vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
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
        wires = Wire.Split(new_wire) if not Wire.IsManifold(new_wire, silent=silent) else [new_wire]

        processed_wires = [remove_collinear_vertices(w) for w in wires]

        if len(processed_wires) == 0:
            return wire
        elif len(processed_wires) == 1:
            return Topology.SelfMerge(processed_wires[0])
        else:
            return Topology.SelfMerge(Cluster.ByTopologies(processed_wires, silent=silent))


    @staticmethod
    def Representation(
        wire,
        normalize: bool = True,
        rotate: bool = True,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns a normalized representation of a closed wire with alternating edge lengths and interior angles.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        normalize : bool , optional
            If set to True, the edge lengths are normalized such that the shortest edge has a length of 1. Default is True.
        rotate : bool , optional
            If set to True, the representation is rotated such that the shortest edge appears first. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The representation list consisting of alternating edge lengths and interior angles.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Representation - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Representation - Error: The input wire parameter is not closed. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.Representation - Error: The input wire parameter is non-manifold. Returning None.")
            return None

        edges = Topology.Edges(wire, silent=True)
        if not isinstance(edges, list) or len(edges) < 3:
            if not silent:
                print("Wire.Representation - Error: Could not retrieve a valid list of edges from the input wire. Returning None.")
            return None

        angles = Wire.InteriorAngles(
            wire,
            tolerance=tolerance,
            mantissa=mantissa,
            silent=True,
        )
        if not isinstance(angles, list) or len(angles) != len(edges):
            if not silent:
                print("Wire.Representation - Error: Could not compute the interior angles of the input wire. Returning None.")
            return None

        lengths = [
            Edge.Length(
                edge,
                mantissa=15,
                tolerance=tolerance,
                silent=True,
            )
            for edge in edges
        ]
        if any(length is None for length in lengths):
            return None
        lengths = [float(length) for length in lengths]

        if normalize:
            min_length = min(lengths)
            if min_length <= tolerance:
                if not silent:
                    print("Wire.Representation - Error: The input wire contains a zero-length edge. Returning None.")
                return None
            lengths = [length / min_length for length in lengths]

        pairs = list(zip(lengths, angles))
        if rotate and pairs:
            min_index = min(range(len(lengths)), key=lambda index: lengths[index])
            pairs = pairs[min_index:] + pairs[:min_index]

        representation = []
        for length, angle in pairs:
            representation.append(round(length, mantissa))
            representation.append(round(angle, mantissa))
        return representation


    @staticmethod
    def Reverse(wire, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a wire that has the reverse direction of the input wire.

        On the PythonOCC backend, native Edge curves are orientation-reversed rather
        than reconstructed from their endpoints.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        transferDictionaries : bool , optional
            If set to True, dictionaries of the input wire and its subtopologies are transferred to the new wire. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The reversed wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Reverse - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.Reverse - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                result = Core.InstanceCall(
                    wire,
                    "Reverse",
                    transferDictionaries,
                    tolerance,
                )
            except Exception:
                result = None
            if Topology.IsInstance(result, "Wire"):
                if transferDictionaries:
                    result = Topology.SetDictionary(
                        result,
                        Topology.Dictionary(wire),
                        silent=True,
                    )
                return result
            if not silent:
                print("Wire.Reverse - Error: The native backend could not reverse the input wire. Returning None.")
            return None

        # Legacy TopologicCore path.
        original_vertices = Topology.Vertices(wire, silent=True)
        edges = Topology.Edges(wire, silent=True)
        edge_selectors = []
        if transferDictionaries:
            for edge in edges:
                selector = Topology.Centroid(edge)
                selector = Topology.SetDictionary(
                    selector,
                    Topology.Dictionary(edge),
                    silent=True,
                )
                edge_selectors.append(selector)

        vertices = list(original_vertices)
        vertices.reverse()
        return_wire = Wire.ByVertices(
            vertices,
            close=Wire.IsClosed(wire, tolerance=tolerance, silent=True),
            tolerance=tolerance,
            silent=silent,
        )
        if not Topology.IsInstance(return_wire, "Wire"):
            return None

        if transferDictionaries:
            return_wire = Topology.TransferDictionariesBySelectors(
                return_wire,
                selectors=edge_selectors,
                tranEdges=True,
                tolerance=tolerance,
            )
            return_wire = Topology.TransferDictionariesBySelectors(
                return_wire,
                selectors=original_vertices,
                tranVertices=True,
                tolerance=tolerance,
            )

        return Topology.SetDictionary(
            return_wire,
            Topology.Dictionary(wire),
            silent=silent,
        )

    @staticmethod
    def Ribbon(wire,
               thickness: float = 1.0,
               thicknessKey: str = "thickness",
               offset: float = 1.0,
               offsetKey: str = "offset",
               stepOffsetA: float = 0,
               stepOffsetB: float = 0,
               stepOffsetKeyA: str = "stepOffsetA",
               stepOffsetKeyB: str = "stepOffsetB",
               reverse: bool = False,
               bisectors: bool = False,
               transferDictionaries: bool = False,
               epsilon: float = 0.01,
               tolerance: float = 0.0001, 
               silent: bool = False,
               numWorkers: int = None):
        """
        Creates a ribbon (face or shell) wire from the input wire. A positive offset value results in an offset to the interior of an anti-clockwise wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        thickness : float , optional
            The desired thickness of the ribbon. Default is 1.0.
        thicknessKey : str , optional
            The edge dictionary key under which to find the thickness value. The thickness is the width of the ribbon. If a value cannot be found, the thickness input parameter value is used instead. Default is "thickness".
        offset : float , optional
            The desired offset distance. An offset is measured prependicularly from the input wire to the nearest parallel edge that belongs to the ribbon. Default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. Default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. Default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. Default is "stepOffsetB".
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. Default is False.
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned ribbon (i.e. shell). If not, the returned ribbon is a face. Default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the created ribbon. Otherwise, they are not. Default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). Default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        wire_1 = Wire.ByOffset(wire,
                               offset = offset,
                               offsetKey = offsetKey,
                               stepOffsetA = stepOffsetA,
                               stepOffsetB = stepOffsetB,
                               stepOffsetKeyA = stepOffsetKeyA,
                               stepOffsetKeyB = stepOffsetKeyB,
                               reverse = reverse,
                               bisectors = False,
                               transferDictionaries = False,
                               epsilon = epsilon,
                               tolerance = tolerance,
                               silent = silent,
                               numWorkers = numWorkers)
        
        wire_2 = Wire.ByOffset(wire_1,
                               offset = thickness,
                               offsetKey = thicknessKey,
                               stepOffsetA = stepOffsetA,
                               stepOffsetB = stepOffsetB,
                               stepOffsetKeyA = stepOffsetKeyA,
                               stepOffsetKeyB = stepOffsetKeyB,
                               reverse = reverse,
                               bisectors = False,
                               transferDictionaries = False,
                               epsilon = epsilon,
                               tolerance = tolerance,
                               silent = silent,
                               numWorkers = numWorkers)
        
        b_cluster = Wire.Bisectors(wire_1,
                               offset = thickness,
                               offsetKey = thicknessKey,
                               stepOffsetA = stepOffsetA,
                               stepOffsetB = stepOffsetB,
                               stepOffsetKeyA = stepOffsetKeyA,
                               stepOffsetKeyB = stepOffsetKeyB,
                               reverse = reverse,
                               transferDictionaries = False,
                               epsilon = epsilon,
                               tolerance = tolerance,
                               silent = silent,
                               numWorkers = numWorkers)
        
        final_wire = Topology.Merge(wire_1, wire_2, tolerance=tolerance)
        # Fuse vertices:
        vertices = Topology.Vertices(final_wire)
        new_vertices = Vertex.Fuse(vertices, tolerance=tolerance)
        final_wire = Topology.ReplaceVertices(final_wire, verticesA=vertices, verticesB=new_vertices)

        b_edges = [Edge.SetLength(e, Edge.Length(e)+epsilon) for e in Topology.Edges(b_cluster)]
        final_wire = Cluster.ByTopologies(Topology.Edges(final_wire)+b_edges)
        
        # Build selectors list to find the correct faces later
        selectors = []
        all_dictionaries = []
        edges_1 = Topology.Edges(wire)
        for i, edge_1 in enumerate(edges_1):
            d = Topology.Dictionary(edge_1)
            o = Dictionary.ValueAtKey(d, offsetKey, offset)
            t = Dictionary.ValueAtKey(d, thicknessKey, thickness)
            c = Topology.Centroid(edge_1)
            if reverse == True:
                fac = -1
            else:
                fac = 1
            s = Vertex.ByOffset2DRelativeToEdge(c, edge_1, offset = (o+t*0.5)*fac, tolerance = tolerance)
            all_dictionaries.append(d)
            s = Topology.SetDictionary(s, d)
            selectors.append(s)
        bounding_rect = Wire.BoundingRectangle(final_wire)
        bounding_face = Face.ByWire(bounding_rect)
        bounding_shell = Topology.Slice(bounding_face, final_wire)

        shell_faces = Topology.Faces(bounding_shell)
        good_faces = []
        for shell_face in shell_faces:
            for s in selectors:
                if Vertex.IsInternal(s, shell_face, tolerance=epsilon):
                    good_faces.append(shell_face)
        
        Topology.Show(shell_faces, selectors, backgroundColor="orange", faceColor="red")
        shell = Shell.ByFaces(good_faces)
        print("Shell is:", shell)
        
        if Topology.IsInstance(shell, "shell"):
            Topology.Show(shell, backgroundColor="orange", faceColor="red")
            if transferDictionaries:
                shell = Topology.TransferDictionariesBySelectors(shell, selectors, tranFaces=True, tolerance=epsilon)
            if Topology.IsInstance(shell, "shell"):
                # If the bisectors are False, transform the shell into a face and merge and transfer dictionaries.
                if bisectors == False:
                    eb = Shell.ExternalBoundary(shell)
                    ib_list = Shell.InternalBoundaries(shell)
                    f = Face.ByWires(eb, ib_list)
                    if Topology.IsInstance(f, "face"):
                        if transferDictionaries:
                            d = Dictionary.ByMergedDictionaries(all_dictionaries)
                        f = Topology.SetDictionary(f, d)
                        return f
                    else:
                        if not silent:
                            print("Wire.Ribbon - Error: Could not create the final face. Returning None.")
                        return None
                else:
                    return shell
        if not silent:
            print("Wire.Ribbon - Error: Could not create the final shell. Returning None.")
        return None
    
    @staticmethod
    def Roof(face, angle: float = 45, boundary: bool = True, tolerance: float = 0.001, silent: bool = False):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optioal
            The desired angle in degrees of the roof. Default is 45.
        boundary : bool , optional
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number as it was found to work better)
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        from topologicpy.Core import Core
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
                if not silent:
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
            If set to True, error and warning messages are suppressed. Default is False.
            
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
            return wire
        new_wire = Wire.ByVertices(new_vertices, close=Wire.IsClosed(wire), tolerance=tolerance, silent=True)
        if not Topology.IsInstance(new_wire, "wire"):
            if not silent:
                print("Wire.Simplify - Warning: Could not generate a simplified wire. Returning the original wire.")
            return wire
        return new_wire


    @staticmethod
    def Skeleton(face, boundary: bool = True, tolerance: float = 0.001, silent: bool = False):
        """
        Creates a straight skeleton.

        This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
        and depends on the bundled polyskel implementation.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        boundary : bool , optional
            If set to True, the original boundary is returned as part of the skeleton topology. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created straight skeleton topology.

        """
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Wire.Skeleton - Error: The input face parameter is not a valid face. Returning None.")
            return None
        return Wire.Roof(
            face,
            angle=0,
            boundary=boundary,
            tolerance=tolerance,
            silent=silent,
        )
    
    @staticmethod
    def Spiral(origin = None, radiusA : float = 0.05, radiusB : float = 0.5, height : float = 1, turns : int = 10, sides : int = 36, clockwise : bool = False, reverse : bool = False, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a spiral.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the spiral. Default is None which results in the spiral being placed at (0, 0, 0).
        radiusA : float , optional
            The initial radius of the spiral. Default is 0.05.
        radiusB : float , optional
            The final radius of the spiral. Default is 0.5.
        height : float , optional
            The height of the spiral. Default is 1.
        turns : int , optional
            The number of turns of the spiral. Default is 10.
        sides : int , optional
            The number of sides of one full turn in the spiral. Default is 36.
        clockwise : bool , optional
            If set to True, the spiral will be oriented in a clockwise fashion. Otherwise, it will be oriented in an anti-clockwise fashion. Default is False.
        reverse : bool , optional
            If set to True, the spiral will increase in height from the center to the circumference. Otherwise, it will increase in height from the conference to the center. Default is False.
        direction : list , optional
            The vector representing the up direction of the spiral. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the spiral. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
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
            if not silent:
                print("Wire.Spiral - Error: the input origin is not a valid topologic Vertex. Returning None.")
            return None
        if radiusA <= 0:
            if not silent:
                print("Wire.Spiral - Error: the input radiusA cannot be less than or equal to zero. Returning None.")
            return None
        if radiusB <= 0:
            if not silent:
                print("Wire.Spiral - Error: the input radiusB cannot be less than or equal to zero. Returning None.")
            return None
        if radiusA == radiusB:
            if not silent:
                print("Wire.Spiral - Error: the inputs radiusA and radiusB cannot be equal. Returning None.")
            return None
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if turns <= 0:
            if not silent:
                print("Wire.Spiral - Error: the input turns cannot be less than or equal to zero. Returning None.")
            return None
        if sides < 3:
            if not silent:
                print("Wire.Spiral - Error: the input sides cannot be less than three. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            if not silent:
                print("Wire.Spiral - Error: the input placement string is not one of center, lowerleft, upperleft, lowerright, or upperright. Returning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) <= tolerance:
            if not silent:
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
    def Split(wire, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Splits the input wire into segments at vertices where more than two edges meet.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of split wire segments.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Split - Error: The input wire parameter is not a valid wire. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                result = Core.WireUtility.Split(wire, tolerance)
            except Exception:
                result = None
            if isinstance(result, list) and len(result) > 0:
                return result
            if result is not None:
                return result
            # A simple wire has nothing to split.
            if Wire.IsManifold(wire, silent=True, tolerance=tolerance):
                return [wire]
            if not silent:
                print("Wire.Split - Error: The native backend could not split the input wire. Returning None.")
            return None

        # Legacy TopologicCore path.
        def vertexDegree(vertex, host):
            edges = []
            try:
                Core.InstanceCall(vertex, "Edges", host, edges)
            except Exception:
                try:
                    vertex.Edges(host, edges)
                except Exception:
                    return 0
            return len(edges)

        def vertexOtherEdge(vertex, edge, host):
            edges = []
            try:
                Core.InstanceCall(vertex, "Edges", host, edges)
            except Exception:
                vertex.Edges(host, edges)
            if len(edges) < 2:
                return None
            if Topology.IsSame(edges[0], edge):
                return edges[-1]
            return edges[0]

        def edgeOtherVertex(edge, vertex):
            vertices = Topology.Vertices(edge, silent=True)
            if not isinstance(vertices, list) or len(vertices) < 2:
                return None
            if Topology.IsSame(vertex, vertices[0]):
                return vertices[-1]
            return vertices[0]

        def edgeInList(edge, edgeList):
            for candidate in edgeList:
                if Topology.IsSame(candidate, edge):
                    return True
            return False

        vertices = Topology.Vertices(wire, silent=True)
        hubs = [vertex for vertex in vertices if vertexDegree(vertex, wire) > 2]
        wires = []
        global_edges = []

        for vertex in hubs:
            hub_edges = []
            try:
                Core.InstanceCall(vertex, "Edges", wire, hub_edges)
            except Exception:
                vertex.Edges(wire, hub_edges)

            for hub_edge in hub_edges:
                if edgeInList(hub_edge, global_edges):
                    continue

                current_edge = hub_edge
                other_vertex = edgeOtherVertex(current_edge, vertex)
                wire_edges = []

                while other_vertex is not None and vertexDegree(other_vertex, wire) == 2:
                    if not edgeInList(current_edge, global_edges):
                        global_edges.append(current_edge)
                        wire_edges.append(current_edge)
                    current_edge = vertexOtherEdge(other_vertex, current_edge, wire)
                    if current_edge is None:
                        break
                    other_vertex = edgeOtherVertex(current_edge, other_vertex)

                if current_edge is not None and not edgeInList(current_edge, global_edges):
                    global_edges.append(current_edge)
                    wire_edges.append(current_edge)

                if len(wire_edges) > 1:
                    merged = Topology.SelfMerge(
                        Cluster.ByTopologies(wire_edges),
                        tolerance=tolerance,
                    )
                    if merged is not None:
                        wires.append(merged)
                elif len(wire_edges) == 1:
                    wires.append(wire_edges[0])

        if len(wires) < 1:
            return [wire]
        return wires
    

    @staticmethod
    def Square(origin=None, size: float = 1.0, diagonals: bool = False, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. Default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. Default is 1.0.
        diagonals : bool , optional
            If set to True, the diagonals of the square are included. Default is False.
        direction : list , optional
            The vector representing the up direction of the square. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created square.

        """
        return Wire.Rectangle(
            origin=origin,
            width=size,
            length=size,
            diagonals=diagonals,
            direction=direction,
            placement=placement,
            tolerance=tolerance,
            silent=silent,
        )
    
    @staticmethod
    def Squircle(origin = None, radius: float = 0.5, sides: int = 121, a: float = 2.0, b: float = 2.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a Squircle which is a hybrid between a circle and a square. See https://en.wikipedia.org/wiki/Squircle

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the squircle. Default is None which results in the squircle being placed at (0, 0, 0).
        radius : float , optional
            The desired radius of the squircle. Default is 0.5.
        sides : int , optional
            The desired number of sides of the squircle. Default is 121.
        a : float , optional
            The "a" factor affects the x position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. Default is 2.0.
        b : float , optional
            The "b" factor affects the y position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. Default is 2.0.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            return Wire.Circle(origin=origin, radius=radius, sides=sides, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
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
    def Star(origin= None, radiusA: float = 0.5, radiusB: float = 0.2, rays: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the star. Default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. Default is 1.0.
        radiusB : float , optional
            The outer radius of the star. Default is 0.4.
        rays : int , optional
            The number of star rays. Default is 8.
        direction : list , optional
            The vector representing the up direction of the star. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def StartEndVertices(wire, silent: bool = False, tolerance: float = 0.0001):
        """
        Returns the start and end vertices of the input wire.

        The input wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            A two-item list containing the start and end vertices.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        if Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not an open wire. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                result = Core.WireUtility.StartEndVertices(wire, tolerance)
            except Exception:
                result = None
            if (
                isinstance(result, list)
                and len(result) == 2
                and all(Topology.IsInstance(vertex, "Vertex") for vertex in result)
            ):
                return result
            if not silent:
                print("Wire.StartEndVertices - Error: The native backend could not determine the wire endpoints. Returning None.")
            return None

        # Legacy TopologicCore path.
        vertices = Topology.Vertices(wire, silent=True)
        endPoints = [
            vertex
            for vertex in vertices
            if Vertex.Degree(vertex, hostTopology=wire) == 1
        ]
        if len(endPoints) < 2:
            if not silent:
                print("Wire.StartEndVertices - Error: Could not find the end vertices of the input wire. Returning None.")
            return None

        super_edges = Topology.SuperTopologies(
            endPoints[0],
            wire,
            topologyType="edge",
        )
        if not isinstance(super_edges, list) or len(super_edges) == 0:
            if not silent:
                print("Wire.StartEndVertices - Error: Could not determine endpoint orientation. Returning None.")
            return None

        edge1 = super_edges[0]
        start = Edge.StartVertex(edge1, silent=True)
        if Topology.IsSame(endPoints[0], start):
            return [endPoints[0], endPoints[1]]
        return [endPoints[1], endPoints[0]]
    

    @staticmethod
    def StartVertex(wire, silent: bool = False, tolerance: float = 0.0001):
        """
        Returns the start vertex of the input wire.

        The input wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The start vertex of the input wire.

        """
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.StartVertex - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        endpoints = Wire.StartEndVertices(
            wire,
            silent=silent,
            tolerance=tolerance,
        )
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            return None
        return endpoints[0]

    @staticmethod
    def Straighten(wire, host, obstacles: list = None, portals: list = None,
                tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a new Wire obtained by recursively replacing segments of the
        input wire with the longest possible straight edge that:
        1. Is fully embedded in the given host.
        2. Avoids intersection with an optional list of obstacle topologies.
        3. Continues to pass through (intersects) an optional list of portal
        topologies that the original input wire intersects.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input path wire whose vertices define the route to be
            straightened.
        host : topologic_core.Topology
            The host within which the straightened edges must lie.
        obstacles : list, optional
            The list of topologies with which the straightened edges must not intersect.
        portals : list, optional
            The list of topologies with which the straightened edges must intersect.
            Portals with which the original wire does NOT intersect are ignored.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        wire : topologic_core.Wire
            A new Wire whose vertices define the recursively straightened path.
        """
        from bisect import bisect_left, bisect_right

        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        # ----------------------------------------------------------------------
        # Defaults
        # ----------------------------------------------------------------------

        if obstacles is None:
            obstacles = []

        if portals is None:
            portals = []

        # ----------------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------------

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print(
                    "Wire.Straighten - Error: "
                    "The input wire parameter is not a valid Wire. Returning None."
                )
            return None

        if not Topology.IsInstance(host, "Topology"):
            if not silent:
                print(
                    "Wire.Straighten - Error: "
                    "The input host parameter is not a valid Topology. Returning None."
                )
            return None

        if not isinstance(portals, list):
            if not silent:
                print(
                    "Wire.Straighten - Error: "
                    "The input portals parameter is not a valid list. Returning None."
                )
            return None

        if not isinstance(obstacles, list):
            if not silent:
                print(
                    "Wire.Straighten - Error: "
                    "The input obstacles parameter is not a valid list. Returning None."
                )
            return None

        # ----------------------------------------------------------------------
        # Bind frequently-used methods locally
        # ----------------------------------------------------------------------

        is_instance = Topology.IsInstance
        is_same = Topology.IsSame
        difference = Topology.Difference
        intersect = Topology.Intersect

        edge_by_vertices = Edge.ByStartVertexEndVertex
        wire_by_vertices = Wire.ByVertices
        parameter_at_vertex = Wire.ParameterAtVertex

        # ----------------------------------------------------------------------
        # Filter inputs once
        # ----------------------------------------------------------------------

        obstacle_list = [
            o for o in obstacles
            if is_instance(o, "Topology")
        ]

        portal_list = [
            p for p in portals
            if is_instance(p, "Topology")
        ]

        ob_cluster = (
            Cluster.ByTopologies(obstacle_list)
            if obstacle_list
            else None
        )

        # ----------------------------------------------------------------------
        # Remove unnecessary vertices before doing expensive work
        # ----------------------------------------------------------------------

        wire = Wire.RemoveCollinearEdges(
            wire,
            angTolerance=0.1,
            tolerance=tolerance,
        )

        vertices = Topology.Vertices(wire)
        n = len(vertices)

        if n <= 2:
            return wire

        # ----------------------------------------------------------------------
        # Candidate-edge validation
        # ----------------------------------------------------------------------

        def _edge_is_valid(v_start, v_end):
            if is_same(v_start, v_end):
                return True

            edge = edge_by_vertices(
                v_start,
                v_end,
                tolerance=tolerance,
            )

            if not is_instance(edge, "Edge"):
                return False

            # Host containment is normally the most important rejection test.
            if difference(edge, host) is not None:
                return False

            if ob_cluster is not None:
                if intersect(edge, ob_cluster) is not None:
                    return False

            return True

        # ----------------------------------------------------------------------
        # Find the FARTHEST valid endpoint first.
        #
        # This is the main optimisation. The old implementation searched:
        #
        #     start+1, start+2, ... end
        #
        # and therefore evaluated every candidate even when the longest edge was
        # valid. Searching backwards allows immediate exit at the first success.
        # ----------------------------------------------------------------------

        def _find_longest_valid_index(start_idx, local_vertices):
            v_start = local_vertices[start_idx]

            for j in range(len(local_vertices) - 1, start_idx, -1):
                if _edge_is_valid(v_start, local_vertices[j]):
                    return j

            # Preserve the original fallback behaviour.
            return start_idx + 1

        # ----------------------------------------------------------------------
        # Straighten an ordered vertex sequence
        # ----------------------------------------------------------------------

        def _straighten_vertices(local_vertices):
            m = len(local_vertices)

            if m <= 2:
                return local_vertices[:]

            result = [local_vertices[0]]
            idx = 0

            while idx < m - 1:
                idx = _find_longest_valid_index(
                    idx,
                    local_vertices,
                )
                result.append(local_vertices[idx])

            return result

        # ----------------------------------------------------------------------
        # No portals
        # ----------------------------------------------------------------------

        if not portal_list:
            new_vertices = _straighten_vertices(vertices)

            if len(new_vertices) < 2:
                return wire

            return wire_by_vertices(
                new_vertices,
                close=False,
                silent=True,
            )

        # ----------------------------------------------------------------------
        # Locate portal cuts
        #
        # This is done once. Unlike the old implementation, we do not construct
        # sub-wires and recursively call Straighten for every interval.
        # ----------------------------------------------------------------------

        cuts = []

        for portal in portal_list:
            inter = intersect(wire, portal)

            if not is_instance(inter, "Topology"):
                continue

            centroid = Topology.Centroid(inter)

            if not is_instance(centroid, "Vertex"):
                continue

            u_target = parameter_at_vertex(
                wire,
                centroid,
                silent=True,
            )

            if u_target is not None:
                v_on_wire = centroid

            else:
                shortest_edge = Topology.ShortestEdge(
                    centroid,
                    wire,
                    silent=True,
                )

                if not is_instance(shortest_edge, "Edge"):
                    continue

                v_on_wire = Edge.EndVertex(shortest_edge)

                if not is_instance(v_on_wire, "Vertex"):
                    continue

                u_target = parameter_at_vertex(
                    wire,
                    v_on_wire,
                    silent=True,
                )

            if u_target is None:
                continue

            if 0.0 < u_target < 1.0:
                cuts.append((u_target, v_on_wire))

        # ----------------------------------------------------------------------
        # No actual portal intersections
        # ----------------------------------------------------------------------

        if not cuts:
            new_vertices = _straighten_vertices(vertices)

            if len(new_vertices) < 2:
                return wire

            return wire_by_vertices(
                new_vertices,
                close=False,
                silent=True,
            )

        # ----------------------------------------------------------------------
        # Sort and deduplicate portal cuts
        # ----------------------------------------------------------------------

        cuts.sort(key=lambda x: x[0])

        unique_cuts = []
        last_u = None

        for u, v in cuts:
            if last_u is None or abs(u - last_u) > tolerance:
                unique_cuts.append((u, v))
                last_u = u

        cuts = unique_cuts

        # ----------------------------------------------------------------------
        # Cache the wire parameter of every original vertex ONCE.
        #
        # The previous implementation recalculated these values for every
        # portal interval.
        # ----------------------------------------------------------------------

        vertex_parameters = []

        for v in vertices:
            u = parameter_at_vertex(
                wire,
                v,
                silent=True,
            )

            if u is not None:
                vertex_parameters.append((u, v))

        vertex_parameters.sort(key=lambda x: x[0])

        parameter_values = [
            item[0]
            for item in vertex_parameters
        ]

        # ----------------------------------------------------------------------
        # Define interval boundaries.
        #
        # We retain the actual portal intersection vertex, avoiding repeated
        # Wire.VertexByParameter calls.
        # ----------------------------------------------------------------------

        boundaries = [
            (0.0, vertices[0]),
            *cuts,
            (1.0, vertices[-1]),
        ]

        result_vertices = []

        # ----------------------------------------------------------------------
        # Straighten each portal interval directly.
        #
        # No temporary wires.
        # No recursive Straighten calls.
        # No repeated obstacle-cluster creation.
        # No repeated collinear-edge removal.
        # ----------------------------------------------------------------------

        for (a, v_a), (b, v_b) in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            if b - a <= tolerance:
                continue

            lo = bisect_right(parameter_values, a)
            hi = bisect_left(parameter_values, b)

            segment_vertices = [v_a]

            for _, v in vertex_parameters[lo:hi]:
                if not is_same(segment_vertices[-1], v):
                    segment_vertices.append(v)

            if not is_same(segment_vertices[-1], v_b):
                segment_vertices.append(v_b)

            if len(segment_vertices) < 2:
                continue

            straight_vertices = _straighten_vertices(
                segment_vertices
            )

            if not straight_vertices:
                continue

            if not result_vertices:
                result_vertices.extend(straight_vertices)

            elif is_same(
                result_vertices[-1],
                straight_vertices[0],
            ):
                result_vertices.extend(
                    straight_vertices[1:]
                )

            else:
                result_vertices.extend(
                    straight_vertices
                )

        if len(result_vertices) < 2:
            return wire

        return wire_by_vertices(
            result_vertices,
            close=False,
            silent=True,
        )

    @staticmethod
    def Trapezoid(origin= None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the trapezoid. Default is None which results in the trapezoid being placed at (0, 0, 0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. Default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. Default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. Default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. Default is 0.0.
        length : float , optional
            The length of the trapezoid. Default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            The location of the origin of the T-shape. Default is None which results in the T-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the T-shape. Default is 1.0.
        length : float , optional
            The overall length of the T-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the T-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the T-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the T-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the T-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
    def VertexDistance(wire, vertex, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the distance along the input wire from the input origin vertex to the input vertex.

        On the PythonOCC backend, distance is measured along the actual OCCT curves
        rather than along endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            The input vertex.
        origin : topologic_core.Vertex , optional
            The origin of the distance. If set to None, the start vertex of the input wire is used. Default is None.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The distance of the input vertex from the input origin along the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.VertexDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Wire.VertexDistance - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None

        wire_length = Wire.Length(
            wire,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )
        if wire_length is None or wire_length <= tolerance:
            if not silent:
                print("Wire.VertexDistance - Error: The input wire parameter is degenerate. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                distance = Core.WireUtility.DistanceAtPoint(
                    wire,
                    vertex,
                    origin,
                    tolerance,
                )
            except Exception:
                distance = None
            if distance is None:
                if not silent:
                    print("Wire.VertexDistance - Error: The input vertex or origin does not lie on the input wire. Returning None.")
                return None
            return round(float(distance), mantissa)

        # Legacy TopologicCore path.
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(
                wire,
                silent=True,
                tolerance=tolerance,
            )
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.VertexDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(vertex, wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.VertexDistance - Error: The input vertex parameter is not internal to the input wire. Returning None.")
            return None

        def _distance_from_start(target):
            total_distance = 0.0
            for edge in Wire.Edges(wire, silent=True):
                if Vertex.IsInternal(target, edge, tolerance=tolerance, silent=True):
                    local_distance = Vertex.Distance(
                        Edge.StartVertex(edge, silent=True),
                        target,
                        mantissa=None,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if local_distance is None:
                        return None
                    return total_distance + float(local_distance)
                edge_length = Edge.Length(
                    edge,
                    mantissa=15,
                    tolerance=tolerance,
                    silent=True,
                )
                if edge_length is None:
                    return None
                total_distance += float(edge_length)
            return None

        distance_vertex = _distance_from_start(vertex)
        distance_origin = _distance_from_start(origin)
        if distance_vertex is None:
            if not silent:
                print("Wire.VertexDistance - Error: The input vertex parameter is not internal to the input wire. Returning None.")
            return None
        if distance_origin is None:
            if not silent:
                print("Wire.VertexDistance - Error: The input origin parameter is not internal to the input wire. Returning None.")
            return None
        return round(abs(distance_origin - distance_vertex), mantissa)


    @staticmethod
    def VertexByDistance(wire, distance: float = 0.0, origin=None, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex along the input wire offset by the input distance from the input origin.

        On the PythonOCC backend, the offset is evaluated by exact curvilinear
        distance across the ordered OCCT edges.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        distance : float , optional
            The offset distance. Default is 0.
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the start vertex of the input wire is used. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.VertexByDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        wire_length = Wire.Length(
            wire,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )
        if wire_length is None or wire_length <= tolerance:
            if not silent:
                print("Wire.VertexByDistance - Error: The input wire parameter is degenerate. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.VertexByDistance - Error: The input wire parameter is non-manifold. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                result = Core.WireUtility.PointAtDistance(
                    wire,
                    distance,
                    origin,
                    tolerance,
                )
            except Exception:
                result = None
            if not Topology.IsInstance(result, "Vertex") and not silent:
                print("Wire.VertexByDistance - Error: Could not evaluate the requested distance on the input wire. Returning None.")
            return result

        # Legacy TopologicCore path.
        def _compute_u(u):
            try:
                text = str(u)
                decimal_places = len(text.split(".")[1]) if "." in text else 0
            except Exception:
                decimal_places = 12
            u = -(int(u) - u)
            return round(u, decimal_places)

        if abs(distance) <= tolerance:
            return Wire.StartVertex(wire, silent=silent, tolerance=tolerance)
        if abs(distance - wire_length) <= tolerance:
            return Wire.EndVertex(wire, silent=silent, tolerance=tolerance)

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(wire, silent=True, tolerance=tolerance)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.VertexByDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(origin, wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.VertexByDistance - Error: The input origin parameter is not internal to the input wire. Returning None.")
            return None

        start = Wire.StartVertex(wire, silent=True, tolerance=tolerance)
        end = Wire.EndVertex(wire, silent=True, tolerance=tolerance)
        if Vertex.IsCoincident(start, origin, tolerance=tolerance, silent=True):
            u = float(distance) / float(wire_length)
        elif Vertex.IsCoincident(end, origin, tolerance=tolerance, silent=True):
            u = 1.0 - float(distance) / float(wire_length)
        else:
            origin_distance = Wire.VertexDistance(
                wire,
                origin,
                mantissa=15,
                tolerance=tolerance,
                silent=True,
            )
            if origin_distance is None:
                return None
            u = (float(origin_distance) + float(distance)) / float(wire_length)

        return Wire.VertexByParameter(
            wire,
            u=_compute_u(u),
            tolerance=tolerance,
            silent=silent,
        )
    



    @staticmethod
    def ParameterAtVertex(wire, vertex, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the global normalized arc-length parameter of a vertex on a manifold wire.

        The returned parameter ranges from 0.0 at the start of the wire to 1.0 at
        its end. On the PythonOCC backend, local positions on curved edges are
        measured by exact curvilinear length.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            A vertex that lies on the wire.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            Distance tolerance for matching the vertex to the wire. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The global parameter in the range [0, 1], or None if the operation fails.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ParameterAtVertex - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Wire.ParameterAtVertex - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.ParameterAtVertex - Error: The input wire is non-manifold. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                value = Core.WireUtility.ParameterAtPoint(
                    wire,
                    vertex,
                    tolerance,
                )
            except Exception:
                value = None
            if value is None:
                if not silent:
                    print("Wire.ParameterAtVertex - Error: The input vertex does not appear to lie on the wire. Returning None.")
                return None
            return round(float(value), mantissa)

        # Legacy TopologicCore path.
        edges = Wire.Edges(wire, silent=True)
        if not isinstance(edges, list) or len(edges) == 0:
            if not silent:
                print("Wire.ParameterAtVertex - Error: The wire has no edges. Returning None.")
            return None

        edge_lengths = [
            Edge.Length(edge, mantissa=15, tolerance=tolerance, silent=True)
            for edge in edges
        ]
        if any(length is None for length in edge_lengths):
            return None
        total_length = sum(float(length) for length in edge_lengths)
        if total_length <= tolerance:
            if not silent:
                print("Wire.ParameterAtVertex - Error: The wire has zero length. Returning None.")
            return None

        start = Wire.StartVertex(wire, silent=True, tolerance=tolerance)
        end = Wire.EndVertex(wire, silent=True, tolerance=tolerance)
        if Topology.IsInstance(start, "Vertex") and Vertex.IsCoincident(vertex, start, tolerance=tolerance, silent=True):
            return 0.0
        if Topology.IsInstance(end, "Vertex") and Vertex.IsCoincident(vertex, end, tolerance=tolerance, silent=True):
            return 1.0

        accumulated = 0.0
        for edge, edge_length in zip(edges, edge_lengths):
            distance_to_edge = Vertex.Distance(
                vertex,
                edge,
                mantissa=15,
                tolerance=tolerance,
                silent=True,
            )
            if distance_to_edge is not None and distance_to_edge <= tolerance:
                start_edge = Edge.StartVertex(edge, silent=True)
                end_edge = Edge.EndVertex(edge, silent=True)
                distance_start = Vertex.Distance(
                    start_edge,
                    vertex,
                    mantissa=None,
                    tolerance=tolerance,
                    silent=True,
                )
                distance_end = Vertex.Distance(
                    end_edge,
                    vertex,
                    mantissa=None,
                    tolerance=tolerance,
                    silent=True,
                )
                if distance_start is None or distance_end is None:
                    return None
                denominator = distance_start + distance_end
                local_u = 0.0 if denominator == 0 else distance_start / denominator
                global_u = (accumulated + local_u * float(edge_length)) / total_length
                return round(global_u, mantissa)
            accumulated += float(edge_length)

        if not silent:
            print("Wire.ParameterAtVertex - Error: The input vertex does not appear to lie on the wire. Returning None.")
        return None


    @staticmethod
    def VertexByParameter(wire, u: float = 0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex at the input global normalized arc-length parameter on a manifold wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        u : float , optional
            The normalized parameter along the wire. A parameter of 0 returns the start location and a parameter of 1 returns the end location. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The vertex at the input parameter.

        """
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.VertexByParameter - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        try:
            u = float(u)
        except Exception:
            if not silent:
                print("Wire.VertexByParameter - Error: The input u parameter is not a valid number. Returning None.")
            return None
        if u < 0.0 or u > 1.0:
            if not silent:
                print("Wire.VertexByParameter - Error: The input u parameter is not within the valid range of [0, 1]. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.VertexByParameter - Error: The input wire parameter is non-manifold. Returning None.")
            return None

        if Wire._UseNativeWireBackend():
            try:
                result = Core.WireUtility.PointAtParameter(
                    wire,
                    u,
                    tolerance,
                )
            except Exception:
                result = None
            if not Topology.IsInstance(result, "Vertex") and not silent:
                print("Wire.VertexByParameter - Error: Could not evaluate the input parameter on the wire. Returning None.")
            return result

        # Legacy TopologicCore path.
        if u == 0.0:
            return Wire.StartVertex(wire, silent=silent, tolerance=tolerance)
        if u == 1.0:
            return Wire.EndVertex(wire, silent=silent, tolerance=tolerance)

        edges = Wire.Edges(wire, silent=True)
        if not isinstance(edges, list) or len(edges) == 0:
            return None

        edge_lengths = []
        total_length = 0.0
        for edge in edges:
            edge_length = Edge.Length(
                edge,
                mantissa=15,
                tolerance=tolerance,
                silent=True,
            )
            if edge_length is None:
                return None
            edge_length = float(edge_length)
            edge_lengths.append(edge_length)
            total_length += edge_length

        if total_length <= tolerance:
            return None

        target = u * total_length
        accumulated_length = 0.0
        current_edge = None
        current_length = None
        for edge, edge_length in zip(edges, edge_lengths):
            if target <= accumulated_length + edge_length:
                current_edge = edge
                current_length = edge_length
                break
            accumulated_length += edge_length

        if current_edge is None or current_length is None or current_length <= tolerance:
            return None

        residual_u = (target - accumulated_length) / current_length
        return Edge.VertexByParameter(
            current_edge,
            residual_u,
            tolerance=tolerance,
            silent=silent,
        )


    @staticmethod
    def Vertices(wire, silent: bool = False) -> list:
        """
        Returns the list of vertices of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of vertices.

        """
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Vertices - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        vertices = []
        try:
            Core.InstanceCall(wire, "Vertices", None, vertices)
        except Exception:
            try:
                result = Core.InstanceCall(wire, "Vertices")
                if isinstance(result, list):
                    vertices = result
                else:
                    return None
            except Exception:
                return None
        return vertices

