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
    def Arc(startVertex, middleVertex, endVertex, sides: int = 16, close: bool = True, polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a circular-arc Wire through three vertices by delegating curve creation to Edge.ArcByVertices."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        edge = Edge.ArcByVertices(startVertex, middleVertex, endVertex, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        wire = Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)
        if not Topology.IsInstance(wire, "Wire"):
            return None
        if close:
            chord = Edge.ByStartVertexEndVertex(
                Edge.EndVertex(edge, silent=True),
                Edge.StartVertex(edge, silent=True),
                tolerance=tolerance,
                silent=True,
            )
            if Topology.IsInstance(chord, "Edge"):
                edges = (Topology.Edges(wire, silent=True) or []) + [chord]
                closed_wire = Wire.ByEdges(edges, orient=True, tolerance=tolerance, silent=True)
                if Topology.IsInstance(closed_wire, "Wire"):
                    wire = closed_wire
        return wire

    
    @staticmethod
    def ArcByEdge(edge, sagitta: float = 1, absolute: bool = True, sides: int = 16, close: bool = True, polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a circular-arc Wire from a linear chord Edge and a sagitta."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge") or not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.ArcByEdge - Error: The input edge must be a geometrically linear Edge. Returning None.")
            return None
        try:
            sagitta = float(sagitta)
        except Exception:
            return None
        if sagitta <= 0.0:
            return None
        length = sagitta if absolute else Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True) * sagitta
        normal_edge = Edge.NormalEdge(edge, length=length, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(normal_edge, "Edge"):
            return None
        return Wire.Arc(
            Edge.StartVertex(edge, silent=True),
            Edge.EndVertex(normal_edge, silent=True),
            Edge.EndVertex(edge, silent=True),
            sides=sides,
            close=close,
            polyline=polyline,
            tolerance=tolerance,
            silent=silent,
        )




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
    def _OrderedEdges(wire, startVertex=None, tolerance: float = 0.0001, silent: bool = False):
        """Returns the edges of a simple wire in oriented head-to-tail traversal order.

        Existing edge geometry is preserved. If an edge must be reversed, ``Edge.Reverse``
        is used; if the active backend cannot reverse that edge exactly, the method returns
        ``None`` rather than rebuilding the edge from its endpoints.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tol = 0.0001

        edges = Wire.Edges(wire, silent=True) or []
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
        if not edges:
            return None

        representatives = []
        adjacency = {}
        edge_nodes = []

        def node_index(vertex):
            for i, rep in enumerate(representatives):
                if Vertex.IsCoincident(vertex, rep, tolerance=tol, silent=True):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1

        for i, edge in enumerate(edges):
            a_v = Edge.StartVertex(edge, silent=True)
            b_v = Edge.EndVertex(edge, silent=True)
            if not Topology.IsInstance(a_v, "Vertex") or not Topology.IsInstance(b_v, "Vertex"):
                return None
            a = node_index(a_v)
            b = node_index(b_v)
            edge_nodes.append((a, b))
            adjacency.setdefault(a, []).append(i)
            adjacency.setdefault(b, []).append(i)

        # A simple manifold path/cycle has degree <= 2 everywhere.
        if any(len(indices) > 2 for indices in adjacency.values()):
            return None

        open_nodes = [node for node, indices in adjacency.items() if len(indices) == 1]
        if len(open_nodes) not in (0, 2):
            return None
        closed = len(open_nodes) == 0

        start_node = None
        if Topology.IsInstance(startVertex, "Vertex"):
            for i, rep in enumerate(representatives):
                if Vertex.IsCoincident(startVertex, rep, tolerance=tol, silent=True):
                    start_node = i
                    break
            if start_node is None or (not closed and start_node not in open_nodes):
                return None
        elif closed:
            # Keep the stored first edge's orientation as the seam/direction hint.
            start_node = edge_nodes[0][0]
        else:
            # Prefer the endpoint whose sole incident edge is already oriented away from it.
            first, second = open_nodes
            first_edge = edges[adjacency[first][0]]
            second_edge = edges[adjacency[second][0]]
            first_forward = Vertex.IsCoincident(
                Edge.StartVertex(first_edge, silent=True), representatives[first], tolerance=tol, silent=True
            )
            second_forward = Vertex.IsCoincident(
                Edge.StartVertex(second_edge, silent=True), representatives[second], tolerance=tol, silent=True
            )
            if first_forward and not second_forward:
                start_node = first
            elif second_forward and not first_forward:
                start_node = second
            else:
                # Deterministic fallback only; this does not change edge geometry.
                start_node = min(open_nodes)

        ordered = []
        used = set()
        current = start_node

        while len(used) < len(edges):
            candidates = [i for i in adjacency.get(current, []) if i not in used]
            if not candidates:
                break

            # At a closed-wire seam there may be two candidates. Prefer the one already
            # oriented away from the current node so the stored direction is retained.
            selected = candidates[0]
            if len(candidates) > 1:
                forward = [i for i in candidates if edge_nodes[i][0] == current]
                if forward:
                    selected = forward[0]

            source = edges[selected]
            a, b = edge_nodes[selected]
            if a == current:
                oriented = source
                nxt = b
            elif b == current:
                oriented = Edge.Reverse(source, tolerance=tol, silent=True)
                nxt = a
            else:
                return None

            if not Topology.IsInstance(oriented, "Edge"):
                if not silent:
                    print("Wire._OrderedEdges - Error: An edge could not be reversed without altering its geometry. Returning None.")
                return None

            ordered.append(oriented)
            used.add(selected)
            current = nxt

        if len(used) != len(edges):
            return None
        if closed and current != start_node:
            return None
        if not closed and current not in open_nodes:
            return None
        return ordered

    @staticmethod
    def _EdgeLengthByParameters(edge, uA: float = 0.0, uB: float = 1.0, tolerance: float = 0.0001):
        """Returns curve length between two normalized edge parameters.

        Exact backend trimming is used when available. Otherwise a geometry-query-only
        adaptive polyline integration is used. The fallback never reconstructs topology.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            return None
        try:
            a = float(uA)
            b = float(uB)
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return None
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        if abs(b - a) <= 1.0e-15:
            return 0.0

        # Exact path where the backend can trim curves.
        try:
            part = Edge.TrimByParameters(edge, uA=a, uB=b, tolerance=tol, silent=True)
            if Topology.IsInstance(part, "Edge"):
                value = Edge.Length(part, mantissa=None, tolerance=tol, silent=True)
                if value is not None:
                    return float(value)
        except Exception:
            pass

        # Linear edges are exact under normalized interpolation.
        try:
            if Edge.IsLinear(edge, tolerance=tol, silent=True):
                total = Edge.Length(edge, mantissa=None, tolerance=tol, silent=True)
                return None if total is None else abs(b-a) * float(total)
        except Exception:
            pass

        def xyz(u):
            v = Edge.VertexByParameter(edge, u=u, tolerance=tol, silent=True)
            if not Topology.IsInstance(v, "Vertex"):
                return None
            c = Vertex.Coordinates(v, mantissa=None)
            if not isinstance(c, (list, tuple)) or len(c) < 3:
                return None
            try:
                return (float(c[0]), float(c[1]), float(c[2]))
            except Exception:
                return None

        def dist(p, q):
            return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)

        p0 = xyz(a)
        p1 = xyz(b)
        if p0 is None or p1 is None:
            return None

        # Adaptive chord refinement. This is used only for measurement on backends
        # lacking exact curved trimming; it never changes the edge itself.
        target_error = max(tol * 0.01, 1.0e-10)

        def integrate(x0, x1, q0, q1, depth):
            xm = 0.5 * (x0 + x1)
            qm = xyz(xm)
            if qm is None:
                return None
            chord = dist(q0, q1)
            split = dist(q0, qm) + dist(qm, q1)
            if depth <= 0 or abs(split - chord) <= target_error:
                return split
            left = integrate(x0, xm, q0, qm, depth-1)
            if left is None:
                return None
            right = integrate(xm, x1, qm, q1, depth-1)
            if right is None:
                return None
            return left + right

        return integrate(a, b, p0, p1, 18)

    @staticmethod
    def _DistanceFromStart(wire, vertex, tolerance: float = 0.0001, silent: bool = False):
        """Returns curvilinear distance from the traversal start to a vertex on a simple wire."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=silent)
        if not isinstance(edges, list) or not edges:
            return None

        accumulated = 0.0
        for edge in edges:
            edge_length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
            if edge_length is None:
                return None
            edge_length = float(edge_length)
            u = Edge.ParameterAtVertex(edge, vertex, mantissa=None, tolerance=tolerance, silent=True)
            if u is not None:
                try:
                    u = max(0.0, min(1.0, float(u)))
                except Exception:
                    return None
                if u <= 1.0e-12:
                    return accumulated
                if u >= 1.0 - 1.0e-12:
                    return accumulated + edge_length
                local = Wire._EdgeLengthByParameters(edge, 0.0, u, tolerance=tolerance)
                return None if local is None else accumulated + float(local)
            accumulated += edge_length
        return None

    @staticmethod
    def _VertexAtDistanceFromStart(wire, distance: float, tolerance: float = 0.0001, silent: bool = False):
        """Returns a vertex at curvilinear distance from the traversal start of a simple wire."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        try:
            distance = float(distance)
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tol, silent=silent)
        if not isinstance(edges, list) or not edges:
            return None

        lengths = []
        total = 0.0
        for edge in edges:
            length = Edge.Length(edge, mantissa=None, tolerance=tol, silent=True)
            if length is None:
                return None
            length = float(length)
            lengths.append(length)
            total += length

        if total <= tol or distance < -tol or distance > total + tol:
            return None
        distance = max(0.0, min(total, distance))
        if distance <= tol:
            return Edge.StartVertex(edges[0], silent=True)
        if abs(distance-total) <= tol:
            return Edge.EndVertex(edges[-1], silent=True)

        accumulated = 0.0
        for edge, length in zip(edges, lengths):
            if distance <= accumulated + length + tol:
                local = max(0.0, min(length, distance-accumulated))
                if local <= tol:
                    return Edge.StartVertex(edge, silent=True)
                if abs(local-length) <= tol:
                    return Edge.EndVertex(edge, silent=True)
                return Edge.VertexByDistance(
                    edge,
                    distance=local,
                    origin=Edge.StartVertex(edge, silent=True),
                    mantissa=None,
                    tolerance=tol,
                    silent=True,
                )
            accumulated += length
        return None

    @staticmethod
    def IsPolyline(wire, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """Returns True if every constituent edge is geometrically linear."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.IsPolyline - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            tol = float(tolerance)
        except Exception:
            return None
        if tol <= 0:
            return None
        edges = Wire.Edges(wire, silent=True) or []
        if not edges:
            return None
        return all(bool(Edge.IsLinear(edge, tolerance=tol, silent=True)) for edge in edges)

    @staticmethod
    def ByEdges(edges: list, orient: bool = False, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a wire from edges while preserving the constituent edge geometry."""
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        if not isinstance(edges, list):
            return None
        edge_list = [e for e in edges if Topology.IsInstance(e, "Edge")]
        if not edge_list:
            if not silent:
                print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
            return None
        try:
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tol = 0.0001

        def construct(source_edges):
            result = None
            # Prefer the backend wire constructor. It preserves actual curve geometry
            # and, on the PythonOCC backend, the first edge gives a stable direction hint.
            try:
                result = Core.Wire.ByEdges(source_edges, tol)
            except TypeError:
                try:
                    result = Core.Wire.ByEdges(source_edges)
                except Exception:
                    result = None
            except Exception:
                result = None
            if Topology.IsInstance(result, "Wire"):
                return result
            try:
                result = Topology.SelfMerge(Cluster.ByTopologies(source_edges), tolerance=tol)
            except Exception:
                result = None
            return result if Topology.IsInstance(result, "Wire") else None

        wire = construct(edge_list)
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ByEdges - Error: The operation failed. Returning None.")
            return None

        # Preserve historical dictionary transfer behavior, but match actual topology
        # before falling back to endpoint equivalence (important for arcs sharing endpoints).
        result_edges = Wire.Edges(wire, silent=True) or []
        updated_edges = []
        changed = False
        for result_edge in result_edges:
            source = None
            for candidate in edge_list:
                try:
                    if Topology.IsSame(result_edge, candidate):
                        source = candidate
                        break
                except Exception:
                    pass
            if source is None:
                rs = Edge.StartVertex(result_edge, silent=True)
                re = Edge.EndVertex(result_edge, silent=True)
                for candidate in edge_list:
                    cs = Edge.StartVertex(candidate, silent=True)
                    ce = Edge.EndVertex(candidate, silent=True)
                    if all(Topology.IsInstance(v, "Vertex") for v in (rs, re, cs, ce)):
                        same = (Vertex.Distance(rs, cs) <= tol and Vertex.Distance(re, ce) <= tol)
                        rev = (Vertex.Distance(rs, ce) <= tol and Vertex.Distance(re, cs) <= tol)
                        if same or rev:
                            source = candidate
                            break
            updated = result_edge
            if source is not None:
                d = Topology.Dictionary(source, silent=True)
                if d:
                    candidate = Topology.SetDictionary(updated, d, silent=True)
                    if Topology.IsInstance(candidate, "Edge"):
                        updated = candidate
                        changed = True
            updated_edges.append(updated)

        if changed and len(updated_edges) == len(result_edges):
            rebuilt = construct(updated_edges)
            if Topology.IsInstance(rebuilt, "Wire"):
                wire = rebuilt

        if transferDictionaries:
            source_cluster = Cluster.ByTopologies(edge_list)
            for wire_edge in Wire.Edges(wire, silent=True) or []:
                internal = Topology.InternalVertex(wire_edge, tolerance=tol, silent=True)
                if not Topology.IsInstance(internal, "Vertex") or source_cluster is None:
                    continue
                enclosing = Vertex.EnclosingEdges(internal, source_cluster, exclusive=False, tolerance=tol, silent=True)
                if isinstance(enclosing, list) and enclosing:
                    dictionaries = [Topology.Dictionary(e, silent=True) for e in enclosing]
                    merged = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                    if merged:
                        Topology.SetDictionary(wire_edge, merged, silent=True)

        if orient and Wire.IsManifold(wire, tolerance=tol, silent=True):
            desired_start = Edge.StartVertex(edge_list[0], silent=True)
            oriented = Wire.OrientEdges(
                wire,
                desired_start,
                transferDictionaries=transferDictionaries,
                tolerance=tol,
                silent=True,
            )
            if Topology.IsInstance(oriented, "Wire"):
                wire = oriented
        return wire
    
    @staticmethod
    def ByEdge(edge, sides: int = 1, polyline: bool = False, silent: bool = False):
        """
        Creates a Wire by subdividing or sampling one input Edge.

        ``sides`` always means the number of Edge subtopologies in the returned Wire.
        In curved mode the source Edge is trimmed into exact curve segments. In
        polyline mode the source Edge is sampled at equal normalized parameters and
        straight chord Edges are created.

        Closed curved Edges accept any ``sides >= 1``. Closed polyline Edges require
        at least three sides. Open Edges accept any ``sides >= 1`` in either mode.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Wire.ByEdge - Error: The input edge parameter is not a valid Edge. Returning None.")
            return None
        try:
            numeric_sides = float(sides)
            sides = int(numeric_sides)
        except Exception:
            if not silent:
                print("Wire.ByEdge - Error: The input sides parameter is not a valid integer. Returning None.")
            return None
        if not math.isfinite(numeric_sides) or abs(numeric_sides - sides) > 1.0e-12 or sides < 1:
            if not silent:
                print("Wire.ByEdge - Error: The input sides parameter must be an integer greater than or equal to one. Returning None.")
            return None

        closed = bool(Edge.IsClosed(edge, silent=True))
        if not closed:
            start = Edge.StartVertex(edge, silent=True)
            end = Edge.EndVertex(edge, silent=True)
            if Topology.IsInstance(start, "Vertex") and Topology.IsInstance(end, "Vertex"):
                closed = bool(Vertex.IsCoincident(start, end, tolerance=0.0001, silent=True))

        if polyline and closed and sides < 3:
            if not silent:
                print("Wire.ByEdge - Error: A closed polyline requires at least three sides. Returning None.")
            return None

        if not polyline:
            if sides == 1:
                return Wire.ByEdges([edge], orient=True, silent=silent)
            segments = []
            for i in range(sides):
                segment = Edge.TrimByParameters(
                    edge,
                    uA=float(i) / float(sides),
                    uB=float(i + 1) / float(sides),
                    silent=True,
                )
                if not Topology.IsInstance(segment, "Edge"):
                    if not silent:
                        print("Wire.ByEdge - Error: Could not trim the input Edge into the requested curved segments. Returning None.")
                    return None
                segments.append(segment)
            return Wire.ByEdges(segments, orient=True, silent=silent)

        if closed:
            vertices = [
                Edge.VertexByParameter(edge, u=float(i) / float(sides), silent=True)
                for i in range(sides)
            ]
            if not all(Topology.IsInstance(v, "Vertex") for v in vertices):
                return None
            return Wire.ByVertices(vertices, close=True, silent=silent)

        vertices = [
            Edge.VertexByParameter(edge, u=float(i) / float(sides), silent=True)
            for i in range(sides + 1)
        ]
        if not all(Topology.IsInstance(v, "Vertex") for v in vertices):
            return None
        return Wire.ByVertices(vertices, close=False, silent=silent)


    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        if not Topology.IsInstance(cluster, "Cluster"):
            print("Wire.ByEdges - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = Topology.Edges(cluster)
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def ByOffset(
        wire,
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
        numWorkers: int = None,
    ):
        """
        Creates an offset Wire.

        For geometrically curved Wires, the PythonOCC backend uses
        ``BRepOffsetAPI_MakeOffset`` so circular, B-spline, and NURBS Edges remain
        genuine curves. Curved Wires with per-Edge varying offset distances are
        rejected because rebuilding those Edges independently would destroy exact
        corner/join geometry. On non-PythonOCC backends curved Wires are likewise
        rejected rather than silently converted to chords.

        The historical TopologicPy algorithm is retained unchanged for polylines,
        including per-Edge offsets, step offsets, bisectors, and its existing
        dictionary-transfer behaviour.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        offset : float , optional
            The desired offset distance. A positive value offsets to the interior
            of an anti-clockwise closed Wire. Default is 1.0.
        offsetKey : str , optional
            Edge dictionary key used to override ``offset``. Default is "offset".
        stepOffsetA : float , optional
            Historical polyline step offset along the previous Edge. Default is 0.
        stepOffsetB : float , optional
            Historical polyline step offset along the next Edge. Default is 0.
        stepOffsetKeyA : str , optional
            Vertex dictionary key for ``stepOffsetA``. Default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            Vertex dictionary key for ``stepOffsetB``. Default is "stepOffsetB".
        reverse : bool , optional
            If True, reverses the offset direction. Default is False.
        bisectors : bool , optional
            If True, include seam Edges between the source and offset Wire.
            Default is False.
        transferDictionaries : bool , optional
            If True, transfer available dictionaries to the result. Default is False.
        epsilon : float , optional
            Historical polyline cleanup tolerance. Default is 0.01.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If True, suppress diagnostics. Default is False.
        numWorkers : int , optional
            Historical dictionary-transfer worker count.

        Returns
        -------
        topologic_core.Wire
            The offset Wire, or None when the requested operation cannot be
            performed without degrading curved geometry.
        """
        import math

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
                print("Wire.ByOffset - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None

        try:
            offset = float(offset)
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Wire.ByOffset - Error: Invalid offset or tolerance. Returning None.")
            return None

        if not math.isfinite(offset):
            if not silent:
                print("Wire.ByOffset - Error: The input offset must be finite. Returning None.")
            return None

        source_edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(source_edges, list) or not source_edges:
            return None

        is_polyline = bool(Wire.IsPolyline(wire, tolerance=tolerance, silent=True))

        # ------------------------------------------------------------------
        # Native curve-preserving offset.
        # ------------------------------------------------------------------
        if not is_polyline:
            try:
                planar = bool(Topology.IsPlanar(wire, tolerance=tolerance))
            except TypeError:
                try:
                    planar = bool(Topology.IsPlanar(wire))
                except Exception:
                    planar = False
            except Exception:
                planar = False

            if not planar:
                if not silent:
                    print("Wire.ByOffset - Error: Curved Wire offset currently requires a planar Wire. Returning None.")
                return None

            factor = -1.0 if reverse else 1.0
            effective_offsets = []

            for edge in source_edges:
                value = offset
                if isinstance(offsetKey, str):
                    dictionary = Topology.Dictionary(edge, silent=True)
                    if dictionary:
                        try:
                            candidate = Dictionary.ValueAtKey(
                                dictionary,
                                key=offsetKey,
                                defaultValue=offset,
                            )
                        except TypeError:
                            try:
                                candidate = Dictionary.ValueAtKey(dictionary, offsetKey)
                            except Exception:
                                candidate = offset
                        except Exception:
                            candidate = offset

                        if isinstance(candidate, (int, float)):
                            value = float(candidate)

                if not math.isfinite(float(value)):
                    return None
                effective_offsets.append(float(value) * factor)

            native_offset = effective_offsets[0]
            if any(abs(value - native_offset) > tolerance for value in effective_offsets[1:]):
                if not silent:
                    print("Wire.ByOffset - Error: Curved Wires require one uniform offset distance. Per-Edge varying offsets would destroy exact curve joins. Returning None.")
                return None

            if abs(native_offset) <= tolerance:
                return wire

            try:
                is_topologic_core = bool(Topology._IsTopologicCoreBackend())
            except Exception:
                is_topologic_core = True

            if is_topologic_core:
                if not silent:
                    print("Wire.ByOffset - Error: The active backend cannot offset this curved Wire without approximation. Returning None.")
                return None

            def wrap_offset_shape(shape):
                result = None
                try:
                    if Core.HasAttribute("Topology", "ByOcctShape"):
                        result = Core.Topology.ByOcctShape(shape)
                except Exception:
                    result = None

                if Topology.IsInstance(result, "Wire"):
                    return result

                if result is not None:
                    wires = Topology.Wires(result, silent=True) or []
                    wires = [candidate for candidate in wires if Topology.IsInstance(candidate, "Wire")]
                    if len(wires) == 1:
                        return wires[0]

                    if len(wires) > 1:
                        edges = []
                        for candidate in wires:
                            edges.extend(Wire.Edges(candidate, silent=True) or [])
                        merged = Wire.ByEdges(
                            edges,
                            orient=True,
                            tolerance=tolerance,
                            silent=True,
                        )
                        if Topology.IsInstance(merged, "Wire"):
                            return merged
                return None

            try:
                from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
                from OCC.Core.GeomAbs import GeomAbs_Arc
                from OCC.Core.TopoDS import topods

                shape = getattr(wire, "shape", None)
                if shape is None or shape.IsNull():
                    return None

                occ_wire = topods.Wire(shape)
                open_result = not bool(Wire.IsClosed(wire, tolerance=tolerance, silent=True))

                try:
                    maker = BRepOffsetAPI_MakeOffset(
                        occ_wire,
                        GeomAbs_Arc,
                        open_result,
                    )
                except Exception:
                    maker = BRepOffsetAPI_MakeOffset()
                    maker.Init(
                        occ_wire,
                        GeomAbs_Arc,
                        open_result,
                    )

                maker.Perform(native_offset, 0.0)
                if hasattr(maker, "IsDone") and not maker.IsDone():
                    return None

                result = wrap_offset_shape(maker.Shape())
            except Exception:
                result = None

            if not Topology.IsInstance(result, "Wire"):
                if not silent:
                    print("Wire.ByOffset - Error: Native curve-preserving offset construction failed. Returning None.")
                return None

            # Transfer Edge dictionaries by traversal correspondence when OCCT
            # preserves the section count.
            if transferDictionaries:
                result_edges = Wire._OrderedEdges(result, tolerance=tolerance, silent=True)
                if isinstance(result_edges, list) and len(result_edges) == len(source_edges):
                    updated_edges = []
                    for source_edge, result_edge in zip(source_edges, result_edges):
                        dictionary = Topology.Dictionary(source_edge, silent=True)
                        updated = result_edge
                        if dictionary:
                            candidate = Topology.SetDictionary(updated, dictionary, silent=True)
                            if Topology.IsInstance(candidate, "Edge"):
                                updated = candidate
                        updated_edges.append(updated)

                    rebuilt = Wire.ByEdges(
                        updated_edges,
                        orient=True,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if Topology.IsInstance(rebuilt, "Wire"):
                        result = rebuilt

                wire_dictionary = Topology.Dictionary(wire, silent=True)
                if wire_dictionary:
                    candidate = Topology.SetDictionary(result, wire_dictionary, silent=True)
                    if Topology.IsInstance(candidate, "Wire"):
                        result = candidate

            if bisectors:
                source_vertices = Topology.Vertices(wire, silent=True) or []
                result_vertices = Topology.Vertices(result, silent=True) or []

                seams = []
                for source_vertex in source_vertices:
                    if not result_vertices:
                        break

                    nearest = min(
                        result_vertices,
                        key=lambda candidate: Vertex.Distance(source_vertex, candidate),
                    )

                    if Vertex.Distance(source_vertex, nearest) > tolerance:
                        seam = Edge.ByStartVertexEndVertex(
                            source_vertex,
                            nearest,
                            tolerance=tolerance,
                            silent=True,
                        )
                        if Topology.IsInstance(seam, "Edge"):
                            seams.append(seam)

                if seams:
                    merged = Topology.SelfMerge(
                        Cluster.ByTopologies([result] + seams, silent=True),
                        tolerance=tolerance,
                    )
                    if Topology.IsInstance(merged, "Wire"):
                        result = merged
                    else:
                        if not silent:
                            print("Wire.ByOffset - Error: Could not include bisectors while retaining a valid Wire. Returning None.")
                        return None

            return result

        # Curves have already been handled above. The historical algorithm below
        # is deliberately retained only for polylines.
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
            if Core.HasAttribute("Wire", "ByVertices"):
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
    def Circle(
        origin=None,
        radius: float = 0.5,
        sides: int = 16,
        spokes: bool = False,
        fromAngle: float = 0.0,
        toAngle: float = 360.0,
        close: bool = True,
        direction: list = [0, 0, 1],
        placement: str = "center",
        polyline: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a circular Wire.

        When ``polyline`` is False, ``sides`` specifies the number of exact
        circular-arc Edge subtopologies. Geometric accuracy is independent of
        this segmentation count.

        When ``polyline`` is True, ``sides`` specifies the number of straight
        segments of the historical regular-polygon approximation. The polygon
        vertices are constructed analytically at equal angular increments rather
        than by sampling the parameter space of an exact circular Edge.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            Placement origin. If None, the global origin is used. Default is None.
        radius : float , optional
            Circle radius. Default is 0.5.
        sides : int , optional
            Number of exact arc Edges, or straight segments in polyline mode.
            Default is 16.
        spokes : bool , optional
            If True, add radial straight edges from the center to perimeter
            junction vertices where historically applicable. Default is False.
        fromAngle : float , optional
            Beginning of the requested angular range in degrees. Default is 0.
        toAngle : float , optional
            End of the requested angular range in degrees. Default is 360.
        close : bool , optional
            For a partial circle, if True add a straight closing chord. A complete
            360-degree circle is already closed. Default is True.
        direction : list , optional
            Circle-plane normal. Default is [0, 0, 1].
        placement : str , optional
            One of "center", "lowerleft", "upperleft", "lowerright", or
            "upperright". Default is "center".
        polyline : bool , optional
            If True, create the historical straight-edge approximation.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Wire
            The created circular Wire.
        """
        import math

        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print(
                    "Wire.Circle - Error: The input origin parameter is not a "
                    "valid vertex. Returning None."
                )
            return None

        try:
            radius = abs(float(radius))
            numeric_sides = float(sides)
            sides = int(numeric_sides)
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print(
                    "Wire.Circle - Error: One or more numerical input parameters "
                    "are invalid. Returning None."
                )
            return None

        if (
            not math.isfinite(radius)
            or not math.isfinite(numeric_sides)
            or not math.isfinite(fromAngle)
            or not math.isfinite(toAngle)
            or not math.isfinite(tolerance)
            or abs(numeric_sides - sides) > 1.0e-12
            or radius <= tolerance
            or sides < 1
            or tolerance <= 0.0
        ):
            if not silent:
                print(
                    "Wire.Circle - Error: Invalid radius, sides, angular range, "
                    "or tolerance. Returning None."
                )
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print(
                    "Wire.Circle - Error: The input direction parameter is not "
                    "a valid 3D vector. Returning None."
                )
            return None

        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None

        if not all(math.isfinite(value) for value in direction):
            return None

        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            if not silent:
                print(
                    "Wire.Circle - Error: The input direction vector has zero "
                    "magnitude. Returning None."
                )
            return None

        placement = str(placement).lower().strip()

        if placement not in [
            "center",
            "lowerleft",
            "upperleft",
            "lowerright",
            "upperright",
        ]:
            if not silent:
                print(
                    "Wire.Circle - Error: The input placement parameter is not "
                    "recognized. Returning None."
                )
            return None

        while toAngle < fromAngle:
            toAngle += 360.0

        angle_range = toAngle - fromAngle

        if angle_range <= tolerance or angle_range > 360.0 + tolerance:
            if not silent:
                print(
                    "Wire.Circle - Error: The angular range must be greater than "
                    "zero and no greater than 360 degrees. Returning None."
                )
            return None

        full_circle = abs(angle_range - 360.0) <= tolerance

        # ======================================================================
        # Historical polygonal mode
        # ======================================================================
        #
        # Do NOT derive these vertices by sampling Edge.Circle.
        #
        # "sides" has always meant an inscribed regular polygon in this mode, so
        # its vertices must be separated by equal geometric angles. This also
        # guarantees the analytical area
        #
        #     0.5 * n * r^2 * sin(2*pi/n)
        #
        # expected by the faceted primitive APIs.
        # ======================================================================

        if bool(polyline):

            if full_circle and sides < 3:
                if not silent:
                    print(
                        "Wire.Circle - Error: A closed polygonal circle requires "
                        "at least three sides. Returning None."
                    )
                return None

            ox = Vertex.X(origin, mantissa=None)
            oy = Vertex.Y(origin, mantissa=None)
            oz = Vertex.Z(origin, mantissa=None)

            if ox is None or oy is None or oz is None:
                return None

            vertices = []

            count = sides if full_circle else sides + 1

            for i in range(count):
                angle = math.radians(
                    fromAngle
                    + angle_range * float(i) / float(sides)
                )

                vertex = Vertex.ByCoordinates(
                    math.sin(angle) * radius + ox,
                    math.cos(angle) * radius + oy,
                    oz,
                )

                if not Topology.IsInstance(vertex, "Vertex"):
                    return None

                vertices.append(vertex)

            # Preserve the historical traversal orientation.
            vertices.reverse()

            base_wire = Wire.ByVertices(
                vertices,
                close=True if full_circle else bool(close),
                tolerance=tolerance,
                silent=True,
            )

            if not Topology.IsInstance(base_wire, "Wire"):
                if not silent:
                    print(
                        "Wire.Circle - Error: Could not create the polygonal "
                        "circle. Returning None."
                    )
                return None

            perimeter_edges = Wire.Edges(base_wire, silent=True) or []

            if spokes and (full_circle or not close):
                junctions = [
                    Edge.StartVertex(edge, silent=True)
                    for edge in perimeter_edges
                ]

                if not full_circle and perimeter_edges:
                    junctions.append(
                        Edge.EndVertex(
                            perimeter_edges[-1],
                            silent=True,
                        )
                    )

                spoke_edges = []

                for vertex in junctions:
                    spoke = Edge.ByStartVertexEndVertex(
                        origin,
                        vertex,
                        tolerance=tolerance,
                        silent=True,
                    )

                    if Topology.IsInstance(spoke, "Edge"):
                        spoke_edges.append(spoke)

                if spoke_edges:
                    candidate = Wire.ByEdges(
                        perimeter_edges + spoke_edges,
                        tolerance=tolerance,
                        silent=True,
                    )

                    if Topology.IsInstance(candidate, "Wire"):
                        base_wire = candidate

            # Historical placement convention.
            if placement == "lowerleft":
                base_wire = Topology.Translate(
                    base_wire,
                    radius,
                    radius,
                    0,
                )

            elif placement == "upperleft":
                base_wire = Topology.Translate(
                    base_wire,
                    radius,
                    -radius,
                    0,
                )

            elif placement == "lowerright":
                base_wire = Topology.Translate(
                    base_wire,
                    -radius,
                    radius,
                    0,
                )

            elif placement == "upperright":
                base_wire = Topology.Translate(
                    base_wire,
                    -radius,
                    -radius,
                    0,
                )

            if direction != [0.0, 0.0, 1.0]:
                base_wire = Topology.Orient(
                    base_wire,
                    origin=origin,
                    dirA=[0, 0, 1],
                    dirB=direction,
                )

            return (
                base_wire
                if Topology.IsInstance(base_wire, "Wire")
                else None
            )

        # ======================================================================
        # Exact curved mode
        # ======================================================================

        canonical_origin = Vertex.Origin()

        if full_circle:
            curve = Edge.Circle(
                origin=canonical_origin,
                radius=radius,
                placement="center",
                tolerance=tolerance,
                silent=True,
            )

        else:
            # Historical Wire.Circle convention:
            # theta=0 lies on +Y.
            curve = Edge.Arc(
                origin=canonical_origin,
                radius=radius,
                fromAngle=90.0 - toAngle,
                toAngle=90.0 - fromAngle,
                direction=[0, 0, 1],
                placement="center",
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(curve, "Edge"):
            return None

        refs = {
            "center": [0.0, 0.0, 0.0],
            "lowerleft": [-radius, -radius, 0.0],
            "upperleft": [-radius, radius, 0.0],
            "lowerright": [radius, -radius, 0.0],
            "upperright": [radius, radius, 0.0],
        }

        source_origin = Vertex.ByCoordinates(
            *refs[placement]
        )

        curve = Topology.OrientAndPlace(
            curve,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(curve, "Edge"):
            return None

        wire = Wire.ByEdge(
            curve,
            sides=sides,
            polyline=False,
            silent=silent,
        )

        if not Topology.IsInstance(wire, "Wire"):
            return None

        if not full_circle and close:
            chord = Edge.ByStartVertexEndVertex(
                Edge.EndVertex(curve, silent=True),
                Edge.StartVertex(curve, silent=True),
                tolerance=tolerance,
                silent=True,
            )

            if Topology.IsInstance(chord, "Edge"):
                closed_wire = Wire.ByEdges(
                    (Topology.Edges(wire, silent=True) or [])
                    + [chord],
                    orient=True,
                    tolerance=tolerance,
                    silent=True,
                )

                if Topology.IsInstance(closed_wire, "Wire"):
                    wire = closed_wire

        if spokes and (full_circle or not close):
            center = Topology.OrientAndPlace(
                Vertex.Origin(),
                originA=source_origin,
                originB=origin,
                dirA=[0, 0, 1],
                dirB=direction,
                tolerance=tolerance,
                silent=True,
            )

            if Topology.IsInstance(center, "Vertex"):
                spoke_edges = []

                for vertex in Topology.Vertices(
                    wire,
                    silent=True,
                ) or []:

                    spoke = Edge.ByStartVertexEndVertex(
                        center,
                        vertex,
                        tolerance=tolerance,
                        silent=True,
                    )

                    if Topology.IsInstance(spoke, "Edge"):
                        spoke_edges.append(spoke)

                if spoke_edges:
                    candidate = Wire.ByEdges(
                        (Topology.Edges(wire, silent=True) or [])
                        + spoke_edges,
                        tolerance=tolerance,
                        silent=True,
                    )

                    if Topology.IsInstance(candidate, "Wire"):
                        wire = candidate

        return wire

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
        Returns the edges of the input wire.

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
        # _ = wire.Edges(None, edges) # H to Core
        try:
            _ = Core.InstanceCall(wire, "Edges", None, edges)
        except Exception:
            edges = None
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
    def Ellipse(origin=None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates an elliptical Wire by delegating the exact curve to Edge.Ellipse and segmentation to Wire.ByEdge."""
        result = Wire.EllipseAll(
            origin=origin, inputMode=inputMode, width=width, length=length,
            focalLength=focalLength, eccentricity=eccentricity,
            majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength,
            sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close,
            direction=direction, placement=placement, polyline=polyline,
            tolerance=tolerance, silent=silent,
        )
        if result is None:
            if not silent:
                print("Wire.Ellipse - Error: Could not create an ellipse. Returning None.")
            return None
        return result["ellipse"]


    @staticmethod
    def EllipseAll(origin=None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates an elliptical Wire and returns it together with the traditional ellipse parameters."""
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            inputMode = int(inputMode)
            width = abs(float(width)); length = abs(float(length))
            focalLength = abs(float(focalLength)); eccentricity = abs(float(eccentricity))
            majorAxisLength = abs(float(majorAxisLength)); minorAxisLength = abs(float(minorAxisLength))
            fromAngle = float(fromAngle); toAngle = float(toAngle); tolerance = float(tolerance)
        except Exception:
            return None
        if inputMode not in [1, 2, 3, 4] or tolerance <= 0.0:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft"]:
            return None

        if inputMode == 1:
            if width <= tolerance or length <= tolerance: return None
            w, l = width, length; a, b = 0.5 * width, 0.5 * length
            c = math.sqrt(abs(a * a - b * b)); e = c / a if a > tolerance else 0.0
        elif inputMode == 2:
            if focalLength <= tolerance or eccentricity <= 0.0 or eccentricity >= 1.0: return None
            c, e = focalLength, eccentricity; a = c / e
            b2 = a * a - c * c
            if b2 <= tolerance * tolerance: return None
            b = math.sqrt(b2); w, l = 2.0 * a, 2.0 * b
        elif inputMode == 3:
            if focalLength <= tolerance or minorAxisLength <= tolerance: return None
            c, b = focalLength, minorAxisLength; a = math.sqrt(b * b + c * c); e = c / a; w, l = 2.0 * a, 2.0 * b
        else:
            if majorAxisLength <= tolerance or minorAxisLength <= tolerance: return None
            a, b = majorAxisLength, minorAxisLength; c = math.sqrt(abs(a * a - b * b)); e = c / a if a > tolerance else 0.0; w, l = 2.0 * a, 2.0 * b

        while toAngle < fromAngle:
            toAngle += 360.0
        sweep = toAngle - fromAngle
        if sweep <= 1.0e-12 or sweep > 360.0 + 1.0e-9:
            return None
        full = abs(sweep - 360.0) <= 1.0e-9

        # Map the historical Wire angle convention (+Y at zero, CCW output) to
        # the Edge.Ellipse convention (+X at zero, CCW).
        edge = Edge.Ellipse(
            origin=origin,
            inputMode=inputMode,
            width=width,
            length=length,
            focalLength=focalLength,
            eccentricity=eccentricity,
            majorAxisLength=majorAxisLength,
            minorAxisLength=minorAxisLength,
            fromAngle=90.0 - toAngle,
            toAngle=90.0 - fromAngle,
            direction=direction,
            placement=placement,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(edge, "Edge"):
            return None
        wire = Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)
        if not Topology.IsInstance(wire, "Wire"):
            return None
        if not full and close:
            chord = Edge.ByStartVertexEndVertex(Edge.EndVertex(edge, silent=True), Edge.StartVertex(edge, silent=True), tolerance=tolerance, silent=True)
            if Topology.IsInstance(chord, "Edge"):
                candidate = Wire.ByEdges((Topology.Edges(wire, silent=True) or []) + [chord], orient=True, tolerance=tolerance, silent=True)
                if Topology.IsInstance(candidate, "Wire"):
                    wire = candidate

        # Foci follow the same placement/orientation convention as Edge.Ellipse.
        canonical_origin = Vertex.Origin()
        f1 = Vertex.ByCoordinates(c, 0.0, 0.0)
        f2 = Vertex.ByCoordinates(-c, 0.0, 0.0)
        source_origin = canonical_origin if placement == "center" else Vertex.ByCoordinates(-a, -b, 0.0)
        foci = Cluster.ByTopologies([f1, f2])
        foci = Topology.OrientAndPlace(foci, originA=source_origin, originB=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance, silent=True)
        return {"ellipse": wire, "foci": foci, "a": a, "b": b, "c": c, "e": e, "w": w, "l": l}


    @staticmethod
    def EndVertex(wire, silent: bool = False):
        """
        Returns the end vertex of the input wire. The wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Vertex
            The end vertex of the input wire.

        """
        if not Topology.IsInstance(wire, "wire"):
            if not silent:
                print("Wire.EndVertex - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.EndVertex - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        if Wire.IsClosed(wire):
            if not silent:
                print("Wire.EndVertex - Error: The input wire parameter is not an open wire. Returning None.")
            return None
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
            The desired tolerance. Default is 0.0001.
        mantissa : int , optional
            The length of the desired mantissa. Default is 6.
        
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
    def Fillet(
        wire,
        radius: float = 0,
        sides: int = 1,
        radiusKey: str = None,
        polyline: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Fillets (rounds) the corners of the input Wire.

        Fillets are first constructed as exact circular Arc Edges. If ``polyline``
        is False, each fillet is divided into ``sides`` genuine circular Arc Edges.
        If ``polyline`` is True, each fillet is approximated by ``sides`` straight
        Edge segments.

        The input Wire must be planar, manifold, and composed of geometrically
        linear Edges.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        radius : float , optional
            The default fillet radius. Default is 0.
        sides : int , optional
            The number of Edges used to represent each fillet. When ``polyline``
            is False these are exact circular Arc Edges. When ``polyline`` is True
            they are straight Edge segments. Default is 16.
        radiusKey : str , optional
            If specified, each corner Vertex dictionary is queried for this key.
            A valid non-negative numerical value overrides ``radius`` at that
            corner. Default is None.
        polyline : bool , optional
            If True, each fillet is represented by straight Edge segments.
            If False, genuine circular Arc Edges are used. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Wire
            The filleted Wire.

        """
        import math

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Vertex import Vertex

        # ---------------------------------------------------------------------
        # Validate inputs.
        # ---------------------------------------------------------------------

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print(
                    "Wire.Fillet - Error: The input wire parameter is not a "
                    "valid Wire. Returning None."
                )
            return None

        if not Wire.IsManifold(wire):
            if not silent:
                print(
                    "Wire.Fillet - Error: The input Wire is not manifold. "
                    "Returning None."
                )
            return None

        if not Topology.IsPlanar(wire):
            if not silent:
                print(
                    "Wire.Fillet - Error: The input Wire is not planar. "
                    "Returning None."
                )
            return None

        try:
            radius = abs(float(radius))
            sides = int(math.floor(float(sides)))
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print(
                    "Wire.Fillet - Error: One or more input parameters are "
                    "invalid. Returning None."
                )
            return None

        if not math.isfinite(radius) or sides < 1:
            if not silent:
                print(
                    "Wire.Fillet - Error: The radius must be finite and sides "
                    "must be at least 1. Returning None."
                )
            return None

        # ---------------------------------------------------------------------
        # Determine the plane and flatten the Wire.
        # ---------------------------------------------------------------------

        bounding_face = Face.BoundingRectangle(
            wire,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(bounding_face, "Face"):
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not determine the plane of the "
                    "input Wire. Returning None."
                )
            return None

        normal = Face.Normal(
            bounding_face,
            mantissa=None,
        )

        if not isinstance(normal, list) or len(normal) != 3:
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not determine the normal of the "
                    "input Wire. Returning None."
                )
            return None

        origin = Vertex.Origin()

        flat_wire = Topology.Flatten(
            wire,
            origin=origin,
            direction=normal,
        )

        if not Topology.IsInstance(flat_wire, "Wire"):
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not flatten the input Wire. "
                    "Returning None."
                )
            return None

        closed = bool(
            Wire.IsClosed(
                flat_wire,
                tolerance=tolerance,
                silent=True,
            )
        )

        ordered_edges = Wire._OrderedEdges(
            flat_wire,
            tolerance=tolerance,
            silent=True,
        )

        if not isinstance(ordered_edges, list) or len(ordered_edges) < 1:
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not determine the ordered Edges "
                    "of the input Wire. Returning None."
                )
            return None

        # This algorithm is explicitly a line-line fillet operation. Do not use
        # endpoint chords of curved Edges implicitly.
        for edge in ordered_edges:
            if not Edge.IsLinear(
                edge,
                tolerance=tolerance,
                silent=True,
            ):
                if not silent:
                    print(
                        "Wire.Fillet - Error: The input Wire contains a curved "
                        "Edge. Wire.Fillet currently operates only on "
                        "geometrically linear Edges. Returning None."
                    )
                return None

        edge_count = len(ordered_edges)

        if edge_count == 1:
            return wire

        # ---------------------------------------------------------------------
        # Store the current endpoints and available lengths of every Edge.
        # ---------------------------------------------------------------------

        starts = [
            Edge.StartVertex(edge, silent=True)
            for edge in ordered_edges
        ]

        ends = [
            Edge.EndVertex(edge, silent=True)
            for edge in ordered_edges
        ]

        lengths = [
            Edge.Length(
                edge,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )
            for edge in ordered_edges
        ]

        if any(
            not Topology.IsInstance(vertex, "Vertex")
            for vertex in starts + ends
        ):
            return None

        if any(
            length is None
            or not math.isfinite(float(length))
            or float(length) <= tolerance
            for length in lengths
        ):
            return None

        lengths = [float(length) for length in lengths]

        start_trim = [0.0] * edge_count
        end_trim = [0.0] * edge_count

        fillets_after = {}
        fillet_created = False

        # ---------------------------------------------------------------------
        # Helper: numerical 3D vector operations.
        # ---------------------------------------------------------------------

        def coordinates(vertex):
            values = Vertex.Coordinates(
                vertex,
                mantissa=None,
            )
            return [
                float(values[0]),
                float(values[1]),
                float(values[2]),
            ]

        def magnitude(vector):
            return math.sqrt(
                sum(value * value for value in vector)
            )

        # ---------------------------------------------------------------------
        # Process every corner.
        #
        # Corner i is the common point between Edge i and Edge i + 1.
        # For a closed Wire, the final Edge also joins the first Edge.
        # ---------------------------------------------------------------------

        corner_count = edge_count if closed else edge_count - 1

        for i in range(corner_count):
            j = (i + 1) % edge_count

            previous_edge = ordered_edges[i]
            next_edge = ordered_edges[j]

            corner = ends[i]

            if not Vertex.IsCoincident(
                corner,
                starts[j],
                tolerance=tolerance,
                silent=True,
            ):
                if not silent:
                    print(
                        "Wire.Fillet - Error: Consecutive Edges are not "
                        "connected. Returning None."
                    )
                return None

            corner_radius = radius

            # -------------------------------------------------------------
            # Per-Vertex radius override.
            # -------------------------------------------------------------

            if isinstance(radiusKey, str):
                dictionary = Topology.Dictionary(
                    corner,
                    silent=True,
                )

                if dictionary is not None:
                    try:
                        value = Dictionary.ValueAtKey(
                            dictionary,
                            radiusKey,
                        )
                    except Exception:
                        value = None

                    if isinstance(value, (int, float)):
                        value = float(value)

                        if (
                            math.isfinite(value)
                            and value >= 0.0
                        ):
                            corner_radius = value

            if corner_radius <= tolerance:
                continue

            # -------------------------------------------------------------
            # The directed previous Edge points into the corner and the next
            # Edge points away from it.
            #
            # For the fillet construction we need two rays pointing outward
            # from the corner.
            # -------------------------------------------------------------

            incoming_direction = Edge.Direction(
                previous_edge,
                mantissa=None,
            )

            outgoing_direction = Edge.Direction(
                next_edge,
                mantissa=None,
            )

            if (
                not isinstance(incoming_direction, list)
                or not isinstance(outgoing_direction, list)
            ):
                return None

            previous_ray = Vector.Reverse(
                incoming_direction
            )

            next_ray = outgoing_direction

            if (
                Vector.IsParallel(previous_ray, next_ray)
                or Vector.IsAntiParallel(previous_ray, next_ray)
            ):
                continue

            # Edge.Angle uses the traversal directions. This is the turning
            # angle of the Wire. For a circular fillet:
            #
            # tangent distance = r * tan(turn / 2)
            # centre distance  = r / cos(turn / 2)
            #
            turn_angle = Edge.Angle(
                previous_edge,
                next_edge,
                mantissa=12,
            )

            if turn_angle is None:
                continue

            try:
                turn_angle = float(turn_angle)
            except Exception:
                continue

            if (
                not math.isfinite(turn_angle)
                or turn_angle <= 1.0e-9
                or turn_angle >= 180.0 - 1.0e-9
            ):
                continue

            half_angle = 0.5 * math.radians(
                turn_angle
            )

            cosine = math.cos(
                half_angle
            )

            if abs(cosine) <= 1.0e-12:
                continue

            tangent_distance = (
                corner_radius
                * math.tan(half_angle)
            )

            centre_distance = (
                corner_radius
                / cosine
            )

            if (
                not math.isfinite(tangent_distance)
                or not math.isfinite(centre_distance)
                or tangent_distance <= tolerance
            ):
                continue

            # -------------------------------------------------------------
            # Make sure this fillet does not consume either adjacent Edge.
            #
            # Existing trimming at the opposite endpoint is included, so two
            # neighbouring fillets cannot silently cross one another.
            # -------------------------------------------------------------

            if (
                start_trim[i] + tangent_distance
                >= lengths[i] - tolerance
                or
                end_trim[j] + tangent_distance
                >= lengths[j] - tolerance
            ):
                if not silent:
                    print(
                        "Wire.Fillet - Warning: The specified fillet radius is "
                        "too large at one corner. Skipping this fillet."
                    )
                continue

            bisector = Vector.Bisect(
                previous_ray,
                next_ray,
            )

            if not isinstance(bisector, list) or len(bisector) != 3:
                continue

            tangent_a = Topology.TranslateByDirectionDistance(
                corner,
                direction=previous_ray,
                distance=tangent_distance,
            )

            tangent_b = Topology.TranslateByDirectionDistance(
                corner,
                direction=next_ray,
                distance=tangent_distance,
            )

            centre = Topology.TranslateByDirectionDistance(
                corner,
                direction=bisector,
                distance=centre_distance,
            )

            if not all(
                Topology.IsInstance(vertex, "Vertex")
                for vertex in [
                    tangent_a,
                    tangent_b,
                    centre,
                ]
            ):
                continue

            # -------------------------------------------------------------
            # Construct a point on the desired circular Arc between the two
            # tangent points. This is the point on the fillet circle nearest
            # the original corner.
            # -------------------------------------------------------------

            corner_xyz = coordinates(
                corner
            )

            centre_xyz = coordinates(
                centre
            )

            radial = [
                corner_xyz[k] - centre_xyz[k]
                for k in range(3)
            ]

            radial_length = magnitude(
                radial
            )

            if radial_length <= tolerance:
                continue

            radial = [
                value / radial_length
                for value in radial
            ]

            middle = Vertex.ByCoordinates(
                centre_xyz[0] + corner_radius * radial[0],
                centre_xyz[1] + corner_radius * radial[1],
                centre_xyz[2] + corner_radius * radial[2],
            )

            if not Topology.IsInstance(middle, "Vertex"):
                continue

            # -------------------------------------------------------------
            # First create one genuine circular Arc Edge.
            # -------------------------------------------------------------

            fillet_edge = Edge.ArcByVertices(
                tangent_a,
                middle,
                tangent_b,
                tolerance=tolerance,
                silent=True,
            )

            if not Topology.IsInstance(fillet_edge, "Edge"):
                if not silent:
                    print(
                        "Wire.Fillet - Warning: Could not construct a circular "
                        "fillet at one corner. Skipping this fillet."
                    )
                continue

            # -------------------------------------------------------------
            # Then use the common Wire.ByEdge sampling/segmentation contract.
            #
            # polyline=False -> exact circular Arc segments.
            # polyline=True  -> straight chord segments.
            # -------------------------------------------------------------

            fillet_wire = Wire.ByEdge(
                fillet_edge,
                sides=sides,
                polyline=polyline,
                silent=True,
            )

            if not Topology.IsInstance(fillet_wire, "Wire"):
                continue

            fillet_edges = Wire._OrderedEdges(
                fillet_wire,
                startVertex=tangent_a,
                tolerance=tolerance,
                silent=True,
            )

            if (
                not isinstance(fillet_edges, list)
                or len(fillet_edges) != sides
            ):
                continue

            # Commit the trimming only after the fillet has been constructed
            # successfully.
            ends[i] = tangent_a
            starts[j] = tangent_b

            end_trim[i] = tangent_distance
            start_trim[j] = tangent_distance

            fillets_after[i] = fillet_edges
            fillet_created = True

        # Nothing needed filleting.
        if not fillet_created:
            return wire

        # ---------------------------------------------------------------------
        # Rebuild the Wire from trimmed original Edges plus the newly created
        # exact/segmented fillets.
        # ---------------------------------------------------------------------

        result_edges = []

        for i, source_edge in enumerate(
            ordered_edges
        ):
            start = starts[i]
            end = ends[i]

            segment_length = Vertex.Distance(
                start,
                end,
            )

            if (
                segment_length is not None
                and segment_length > tolerance
            ):
                segment = Edge.ByStartVertexEndVertex(
                    start,
                    end,
                    tolerance=tolerance,
                    silent=True,
                )

                if not Topology.IsInstance(segment, "Edge"):
                    if not silent:
                        print(
                            "Wire.Fillet - Error: Could not rebuild a trimmed "
                            "Edge. Returning None."
                        )
                    return None

                # Preserve the dictionary of the source linear Edge.
                source_dictionary = Topology.Dictionary(
                    source_edge,
                    silent=True,
                )

                if source_dictionary:
                    candidate = Topology.SetDictionary(
                        segment,
                        source_dictionary,
                        silent=True,
                    )

                    if Topology.IsInstance(candidate, "Edge"):
                        segment = candidate

                result_edges.append(
                    segment
                )

            if i in fillets_after:
                result_edges.extend(
                    fillets_after[i]
                )

        if len(result_edges) < 1:
            return None

        result_flat = Wire.ByEdges(
            result_edges,
            orient=True,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(result_flat, "Wire"):
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not construct the filleted "
                    "Wire. Returning None."
                )
            return None

        # ---------------------------------------------------------------------
        # Restore the original plane.
        # ---------------------------------------------------------------------

        result = Topology.Unflatten(
            result_flat,
            origin=origin,
            direction=normal,
        )

        if not Topology.IsInstance(result, "Wire"):
            if not silent:
                print(
                    "Wire.Fillet - Error: Could not restore the filleted Wire "
                    "to its original plane. Returning None."
                )
            return None

        # Preserve the Wire dictionary.
        wire_dictionary = Topology.Dictionary(
            wire,
            silent=True,
        )

        if wire_dictionary:
            candidate = Topology.SetDictionary(
                result,
                wire_dictionary,
                silent=True,
            )

            if Topology.IsInstance(candidate, "Wire"):
                result = candidate

        return result
    
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
    def GoldenSpiral(width: float = 1.0, maxIterations: int = 10, clockwise: bool = False, sides: int = 96, origin=None, placement: str = "center", direction: list = [0, 0, 1], mantissa: int = 6, polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a golden spiral Wire by delegating the single exact curve to Edge.GoldenSpiral."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        edge = Edge.GoldenSpiral(width=width, maxIterations=maxIterations, clockwise=clockwise, origin=origin, placement=placement, direction=direction, mantissa=mantissa, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)



    @staticmethod
    def Helix(origin=None, radius: float = 0.5, height: float = 1.0, turns: float = 1.0, sides: int = 16, clockwise: bool = False, direction: list = [0, 0, 1], placement: str = "center", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a helical Wire from one smooth Edge.Helix and Wire.ByEdge."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        # Edge.Helix.sides controls its internal B-spline approximation. Wire.sides
        # is reserved exclusively for output topological segmentation, so the Edge
        # constructor uses its established default internal resolution.
        edge = Edge.Helix(origin=origin, radius=radius, height=height, turns=turns, clockwise=clockwise, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)

    @staticmethod
    def Hyperbola(origin=None, a: float = 1.0, b: float = 0.5, fromParameter: float = -1.0, toParameter: float = 1.0, branch: str = "right", sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a hyperbolic Wire from one exact Edge.Hyperbola and Wire.ByEdge."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        edge = Edge.Hyperbola(origin=origin, a=a, b=b, fromParameter=fromParameter, toParameter=toParameter, branch=branch, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)

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
    def Interpolate(
        wires: list,
        n: int = 5,
        outputType: str = "default",
        mapping: str = "default",
        tolerance: float = 0.0001,
    ):
        """
        Creates intermediate Wires between successive input Wires.

        Polyline inputs retain the historical vertex-based interpolation. When
        one or more input Wires contain genuine curves, corresponding Edges are
        interpolated as genuine B-spline curves on the PythonOCC backend rather
        than being replaced by straight endpoint chords. Curved inputs must have
        matching Edge counts and compatible open/closed topology.

        Parameters
        ----------
        wires : list
            Ordered input Wires.
        n : int , optional
            Number of intermediate Wires between each pair. Default is 5.
        outputType : str , optional
            ``"default"``/``"contours"``, ``"raster"``/``"zigzag"``/
            ``"toolpath"``, or ``"grid"``. Default is ``"default"``.
        mapping : str , optional
            Historical polyline vertex mapping: ``"default"``/``"repeat"`` or
            ``"nearest"``. Curved interpolation requires matching Edge topology.
            Default is ``"default"``.
        tolerance : float , optional
            Geometric tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The interpolated contours and optional connecting ridges.
        """
        import math

        from topologicpy.Cluster import Cluster
        from topologicpy.Edge import Edge
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not isinstance(wires, list):
            return None

        wires = [wire for wire in wires if Topology.IsInstance(wire, "Wire")]
        if len(wires) < 2:
            return None

        try:
            n = int(n)
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return None

        if n < 0:
            return None

        output_type = str(outputType).lower()
        if output_type not in [
            "default",
            "contours",
            "raster",
            "zigzag",
            "toolpath",
            "grid",
        ]:
            return None

        if output_type in ["default", "contours"]:
            output_type = "contours"
        elif output_type in ["raster", "zigzag", "toolpath"]:
            output_type = "zigzag"

        mapping_name = str(mapping).lower()
        if mapping_name not in ["default", "nearest", "repeat"]:
            print("Wire.Interpolate - Error: The mapping input parameter is not recognized. Returning None.")
            return None

        all_polyline = all(
            bool(Wire.IsPolyline(wire, tolerance=tolerance, silent=True))
            for wire in wires
        )

        # ------------------------------------------------------------------
        # Curve-aware interpolation.
        # ------------------------------------------------------------------
        if not all_polyline:
            try:
                is_topologic_core = bool(Topology._IsTopologicCoreBackend())
            except Exception:
                is_topologic_core = True

            if is_topologic_core:
                print(
                    "Wire.Interpolate - Error: The active backend cannot create "
                    "genuine interpolated curves from curved input Wires. Returning None."
                )
                return None

            ordered_edges = [
                Wire._OrderedEdges(
                    wire,
                    tolerance=tolerance,
                    silent=True,
                )
                for wire in wires
            ]

            if any(not isinstance(edges, list) or not edges for edges in ordered_edges):
                return None

            edge_count = len(ordered_edges[0])
            if any(len(edges) != edge_count for edges in ordered_edges[1:]):
                print(
                    "Wire.Interpolate - Error: Curved input Wires must contain "
                    "the same number of corresponding Edges. Returning None."
                )
                return None

            closed_flags = [
                bool(Wire.IsClosed(wire, tolerance=tolerance, silent=True))
                for wire in wires
            ]
            if any(flag != closed_flags[0] for flag in closed_flags[1:]):
                print(
                    "Wire.Interpolate - Error: Curved input Wires must have "
                    "matching open/closed topology. Returning None."
                )
                return None

            def blend_vertex(vertex_a, vertex_b, fraction):
                a = Vertex.Coordinates(vertex_a, mantissa=None)
                b = Vertex.Coordinates(vertex_b, mantissa=None)
                if (
                    not isinstance(a, (list, tuple))
                    or not isinstance(b, (list, tuple))
                    or len(a) < 3
                    or len(b) < 3
                ):
                    return None

                return Vertex.ByCoordinates(
                    (1.0 - fraction) * float(a[0]) + fraction * float(b[0]),
                    (1.0 - fraction) * float(a[1]) + fraction * float(b[1]),
                    (1.0 - fraction) * float(a[2]) + fraction * float(b[2]),
                )

            def wrap_occ_edge(occ_edge):
                result = None

                try:
                    if Core.HasAttribute("Edge", "ByOcctShape"):
                        result = Core.Edge.ByOcctShape(occ_edge)
                except Exception:
                    result = None

                if not Topology.IsInstance(result, "Edge"):
                    try:
                        if Core.HasAttribute("Topology", "ByOcctShape"):
                            result = Core.Topology.ByOcctShape(occ_edge)
                    except Exception:
                        result = None

                return result if Topology.IsInstance(result, "Edge") else None

            def interpolate_edge(edge_a, edge_b, fraction):
                linear_a = bool(
                    Edge.IsLinear(
                        edge_a,
                        tolerance=tolerance,
                        silent=True,
                    )
                )
                linear_b = bool(
                    Edge.IsLinear(
                        edge_b,
                        tolerance=tolerance,
                        silent=True,
                    )
                )

                if linear_a and linear_b:
                    start = blend_vertex(
                        Edge.StartVertex(edge_a, silent=True),
                        Edge.StartVertex(edge_b, silent=True),
                        fraction,
                    )
                    end = blend_vertex(
                        Edge.EndVertex(edge_a, silent=True),
                        Edge.EndVertex(edge_b, silent=True),
                        fraction,
                    )

                    if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                        return None

                    return Edge.ByStartVertexEndVertex(
                        start,
                        end,
                        tolerance=tolerance,
                        silent=True,
                    )

                closed_a = bool(Edge.IsClosed(edge_a, silent=True))
                closed_b = bool(Edge.IsClosed(edge_b, silent=True))

                if closed_a != closed_b:
                    return None

                periodic = closed_a and closed_b
                sample_count = 16 if periodic else 17

                try:
                    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                    from OCC.Core.GeomAPI import GeomAPI_Interpolate
                    from OCC.Core.TColgp import TColgp_HArray1OfPnt
                    from OCC.Core.gp import gp_Pnt

                    points = TColgp_HArray1OfPnt(1, sample_count)

                    for index in range(sample_count):
                        if periodic:
                            parameter = float(index) / float(sample_count)
                        else:
                            parameter = float(index) / float(sample_count - 1)

                        vertex_a = Edge.VertexByParameter(
                            edge_a,
                            u=parameter,
                            tolerance=tolerance,
                            silent=True,
                        )
                        vertex_b = Edge.VertexByParameter(
                            edge_b,
                            u=parameter,
                            tolerance=tolerance,
                            silent=True,
                        )

                        blended = blend_vertex(
                            vertex_a,
                            vertex_b,
                            fraction,
                        )
                        if not Topology.IsInstance(blended, "Vertex"):
                            return None

                        coordinates = Vertex.Coordinates(
                            blended,
                            mantissa=None,
                        )
                        points.SetValue(
                            index + 1,
                            gp_Pnt(
                                float(coordinates[0]),
                                float(coordinates[1]),
                                float(coordinates[2]),
                            ),
                        )

                    interpolator = GeomAPI_Interpolate(
                        points,
                        periodic,
                        tolerance,
                    )
                    interpolator.Perform()

                    if not interpolator.IsDone():
                        return None

                    curve = interpolator.Curve()
                    occ_edge = BRepBuilderAPI_MakeEdge(curve).Edge()

                    return wrap_occ_edge(occ_edge)

                except Exception:
                    return None

            def intermediate_wire(edges_a, edges_b, fraction):
                result_edges = []

                for edge_a, edge_b in zip(edges_a, edges_b):
                    edge = interpolate_edge(
                        edge_a,
                        edge_b,
                        fraction,
                    )
                    if not Topology.IsInstance(edge, "Edge"):
                        return None
                    result_edges.append(edge)

                return Wire.ByEdges(
                    result_edges,
                    orient=True,
                    tolerance=tolerance,
                    silent=True,
                )

            contours = []

            for pair_index in range(len(wires) - 1):
                if pair_index == 0:
                    contours.append(wires[pair_index])

                edges_a = ordered_edges[pair_index]
                edges_b = ordered_edges[pair_index + 1]

                for step in range(1, n + 1):
                    fraction = float(step) / float(n + 1)

                    contour = intermediate_wire(
                        edges_a,
                        edges_b,
                        fraction,
                    )
                    if not Topology.IsInstance(contour, "Wire"):
                        print(
                            "Wire.Interpolate - Error: Could not construct one "
                            "of the curved intermediate Wires. Returning None."
                        )
                        return None

                    contours.append(contour)

                contours.append(wires[pair_index + 1])

            ridges = []

            if output_type in ["grid", "zigzag"]:
                def contour_vertices(contour):
                    edges = Wire._OrderedEdges(
                        contour,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if not isinstance(edges, list) or not edges:
                        return []

                    vertices = [
                        Edge.StartVertex(edge, silent=True)
                        for edge in edges
                    ]

                    if not Wire.IsClosed(
                        contour,
                        tolerance=tolerance,
                        silent=True,
                    ):
                        vertices.append(
                            Edge.EndVertex(
                                edges[-1],
                                silent=True,
                            )
                        )

                    return vertices

                vertex_sets = [
                    contour_vertices(contour)
                    for contour in contours
                ]

                if any(not vertices for vertices in vertex_sets):
                    return None

                for index in range(len(vertex_sets) - 1):
                    vertices_a = vertex_sets[index]
                    vertices_b = vertex_sets[index + 1]

                    if len(vertices_a) != len(vertices_b):
                        return None

                    if output_type == "grid":
                        pairs = zip(vertices_a, vertices_b)
                    elif index % 2 == 0:
                        pairs = [(vertices_a[-1], vertices_b[-1])]
                    else:
                        pairs = [(vertices_a[0], vertices_b[0])]

                    for start, end in pairs:
                        ridge = Edge.ByStartVertexEndVertex(
                            start,
                            end,
                            tolerance=tolerance,
                            silent=True,
                        )
                        if Topology.IsInstance(ridge, "Edge"):
                            ridges.append(ridge)

            return Topology.SelfMerge(
                Cluster.ByTopologies(
                    contours + ridges,
                    silent=True,
                ),
                tolerance=tolerance,
            )

        # ------------------------------------------------------------------
        # Historical polyline interpolation.
        # ------------------------------------------------------------------
        def nearest_vertex(vertex, vertices):
            distances = [
                Vertex.Distance(vertex, candidate)
                for candidate in vertices
            ]
            return vertices[distances.index(min(distances))]

        def replicate(vertices, mapping_mode="default"):
            vertices = Helper.Repeat(vertices)
            final_list = vertices

            if mapping_mode == "nearest":
                final_list = [vertices[0]]

                for index in range(len(vertices) - 1):
                    loop_a = vertices[index]
                    loop_b = vertices[index + 1]
                    nearest_vertices = [
                        nearest_vertex(vertex, loop_b)
                        for vertex in loop_a
                    ]
                    final_list.append(nearest_vertices)

            return final_list

        def process(vertices_a, vertices_b):
            contours = [vertices_a]

            for step in range(1, n + 1):
                fraction = float(step) / float(n + 1)
                temporary = []

                for index in range(len(vertices_a)):
                    segment = Edge.ByStartVertexEndVertex(
                        vertices_a[index],
                        vertices_b[index],
                        tolerance=tolerance,
                        silent=True,
                    )
                    if not Topology.IsInstance(segment, "Edge"):
                        return None

                    vertex = Edge.VertexByParameter(
                        segment,
                        u=fraction,
                        tolerance=tolerance,
                        silent=True,
                    )
                    temporary.append(vertex)

                contours.append(temporary)

            return contours

        vertices = [
            Topology.SubTopologies(
                wire,
                subTopologyType="vertex",
            )
            for wire in wires
        ]
        vertices = replicate(
            vertices,
            mapping_mode=mapping_name,
        )

        contours = []
        final_wires = []

        for index in range(len(vertices) - 1):
            vertices_a = vertices[index]
            vertices_b = vertices[index + 1]

            contour_sets = process(
                vertices_a,
                vertices_b,
            )
            if contour_sets is None:
                return None

            contours += contour_sets

            for contour_vertices in contour_sets:
                contour = Wire.ByVertices(
                    contour_vertices,
                    close=Wire.IsClosed(
                        wires[index],
                        tolerance=tolerance,
                        silent=True,
                    ),
                    tolerance=tolerance,
                    silent=True,
                )
                if Topology.IsInstance(contour, "Wire"):
                    final_wires.append(contour)

        contours.append(vertices[-1])
        final_wires.append(wires[-1])

        ridges = []

        if output_type in ["grid", "zigzag"]:
            for index in range(len(contours) - 1):
                vertices_a = contours[index]
                vertices_b = contours[index + 1]

                if output_type == "grid":
                    pairs = zip(vertices_a, vertices_b)
                elif index % 2 == 0:
                    pairs = [(vertices_a[-1], vertices_b[-1])]
                else:
                    pairs = [(vertices_a[0], vertices_b[0])]

                for start, end in pairs:
                    ridge = Edge.ByStartVertexEndVertex(
                        start,
                        end,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if Topology.IsInstance(ridge, "Edge"):
                        ridges.append(ridge)

        return Topology.SelfMerge(
            Cluster.ByTopologies(
                final_wires + ridges,
                silent=True,
            ),
            tolerance=tolerance,
        )
    
    @staticmethod
    def Invert(wire, silent: bool = False, tolerance: float = 0.0001):
        """
        Reverses the traversal direction of the input Wire while preserving its
        constituent Edge geometry.

        This method is retained as an alias of :meth:`Wire.Reverse` for backward
        compatibility. Curved Edges are reversed natively and are never rebuilt
        from endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The reversed Wire.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Invert - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None

        return Wire.Reverse(
            wire,
            transferDictionaries=True,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def IsClosed(wire, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """Returns True if the input wire is closed."""
        if not Topology.IsInstance(wire, "Wire"):
            return None
        try:
            return bool(Core.InstanceCall(wire, "IsClosed"))
        except Exception:
            try:
                return bool(Core.WireUtility.IsClosed(wire, tolerance))
            except TypeError:
                try:
                    return bool(Core.WireUtility.IsClosed(wire))
                except Exception:
                    pass
            except Exception:
                pass
        if not silent:
            print("Wire.IsClosed - Error: Could not determine whether the input wire is closed. Returning None.")
        return None
    
    @staticmethod
    def IsManifold(wire, silent: bool = False, tolerance: float = 0.0001) -> bool:
        """Returns True when no coincident wire vertex has degree greater than two."""
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.IsManifold - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tol = 0.0001
        reps = []
        degrees = []
        for edge in Wire.Edges(wire, silent=True) or []:
            for v in (Edge.StartVertex(edge, silent=True), Edge.EndVertex(edge, silent=True)):
                if not Topology.IsInstance(v, "Vertex"):
                    continue
                idx = None
                for i, rep in enumerate(reps):
                    if Vertex.IsCoincident(v, rep, tolerance=tol, silent=True):
                        idx = i
                        break
                if idx is None:
                    reps.append(v)
                    degrees.append(1)
                else:
                    degrees[idx] += 1
                    if degrees[idx] > 2:
                        return False
        return bool(reps)

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
        """Returns the sum of the actual geometric lengths of the wire edges."""
        from topologicpy.Edge import Edge
        if not Topology.IsInstance(wire, "Wire"):
            return None
        total = 0.0
        try:
            for edge in Wire.Edges(wire, silent=True) or []:
                value = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
                if value is None:
                    return None
                total += float(value)
            return float(total) if mantissa is None else round(total, int(mantissa))
        except Exception:
            if not silent:
                print("Wire.Length - Error: Could not calculate the length of the input wire. Returning None.")
            return None

    @staticmethod
    def Line(origin=None,
            length: float = 1,
            direction: list = [1, 0, 0],
            sides: int = 2,
            placement: str = "center",
            tolerance: float = 0.0001,
            silent: bool = True):
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Line - Error: The input origin is not a valid vertex. Returning None.")
            return None

        try:
            length = float(length)
            sides = int(sides)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Wire.Line - Error: One or more numerical input parameters are invalid. Returning None.")
            return None

        if length <= 0:
            if not silent:
                print("Wire.Line - Error: The input length is less than or equal to zero. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Wire.Line - Error: The input direction is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
        except Exception:
            if not silent:
                print("Wire.Line - Error: The input direction is not numerical. Returning None.")
            return None

        if sum(value * value for value in direction) <= tolerance * tolerance:
            if not silent:
                print("Wire.Line - Error: The input direction has zero magnitude. Returning None.")
            return None

        if sides < 2:
            if not silent:
                print("Wire.Line - Error: The number of sides cannot be less than two. Consider using Edge.Line() instead. Returning None.")
            return None

        placement = str(placement).lower().strip()
        if placement not in ("center", "start", "end"):
            if not silent:
                print('Wire.Line - Error: The placement must be "center", "start", or "end". Returning None.')
            return None

        full_edge = Edge.Line(origin=origin, length=length, direction=direction, placement=placement)
        if not Topology.IsInstance(full_edge, "Edge"):
            return None

        vertices = [Edge.StartVertex(full_edge)]
        for i in range(1, sides):
            vertex = Edge.VertexByParameter(full_edge, float(i) / float(sides))
            if not Topology.IsInstance(vertex, "Vertex"):
                return None
            vertices.append(vertex)
        vertices.append(Edge.EndVertex(full_edge))

        edges = []
        for i in range(sides):
            edge = Edge.ByStartVertexEndVertex(vertices[i], vertices[i + 1])
            if not Topology.IsInstance(edge, "Edge"):
                return None
            edges.append(edge)

        result = Wire.ByEdges(edges, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Wire"):
            if not silent:
                print("Wire.Line - Error: Could not create the subdivided Wire. Returning None.")
            return None
        return result


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
    def Miter(
        wire,
        offset: float = 0,
        offsetKey: str = None,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Miters the corners of a planar polyline by trimming each adjacent linear
        Edge and joining the resulting trim points with a straight segment.

        This is intentionally a linear/polyline operation. Curved input Edges are
        rejected rather than silently replaced by endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        offset : float
            The desired trim distance along each adjacent Edge.
        offsetKey : str , optional
            If specified, each corner Vertex dictionary is queried for this key.
            A valid non-negative numerical value overrides ``offset`` at that corner.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The mitered polyline Wire.
        """
        import math

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Miter - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Miter - Error: The input Wire is not manifold. Returning None.")
            return None
        try:
            planar = bool(Topology.IsPlanar(wire, tolerance=tolerance))
        except TypeError:
            try:
                planar = bool(Topology.IsPlanar(wire))
            except Exception:
                planar = False
        except Exception:
            planar = False

        if not planar:
            if not silent:
                print("Wire.Miter - Error: The input Wire is not planar. Returning None.")
            return None
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Miter - Error: Wire.Miter is a linear-only operation and does not accept curved Edges. Returning None.")
            return None

        try:
            offset = abs(float(offset))
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Wire.Miter - Error: Invalid offset or tolerance. Returning None.")
            return None

        if not math.isfinite(offset):
            return None
        if offset <= tolerance:
            return wire

        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(edges, list) or not edges:
            return None

        closed = bool(Wire.IsClosed(wire, tolerance=tolerance, silent=True))
        edge_count = len(edges)

        if edge_count < 2:
            return wire

        lengths = []
        for edge in edges:
            length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
            if length is None:
                return None
            lengths.append(float(length))

        start_trim = [0.0] * edge_count
        end_trim = [0.0] * edge_count
        corners = {}

        corner_indices = range(edge_count) if closed else range(1, edge_count)

        for corner_index in corner_indices:
            previous_index = (corner_index - 1) % edge_count
            next_index = corner_index % edge_count

            previous_edge = edges[previous_index]
            next_edge = edges[next_index]
            corner = Edge.StartVertex(next_edge, silent=True)

            if not Topology.IsInstance(corner, "Vertex"):
                return None

            corner_offset = offset

            if isinstance(offsetKey, str):
                dictionary = Topology.Dictionary(corner, silent=True)
                if dictionary:
                    try:
                        value = Dictionary.ValueAtKey(dictionary, offsetKey)
                    except Exception:
                        value = None
                    if isinstance(value, (int, float)):
                        value = float(value)
                        if math.isfinite(value) and value >= 0.0:
                            corner_offset = value

            if corner_offset <= tolerance:
                corners[corner_index] = (corner, corner)
                continue

            angle = Edge.Angle(previous_edge, next_edge, mantissa=12)
            if angle is None:
                corners[corner_index] = (corner, corner)
                continue

            try:
                angle = float(angle)
            except Exception:
                angle = 0.0

            if abs(angle) <= 1.0e-9 or abs(angle - 180.0) <= 1.0e-9:
                corners[corner_index] = (corner, corner)
                continue

            if (
                start_trim[previous_index] + corner_offset >= lengths[previous_index] - tolerance
                or
                end_trim[next_index] + corner_offset >= lengths[next_index] - tolerance
            ):
                if not silent:
                    print("Wire.Miter - Warning: The requested offset is too large at one corner. Leaving that corner unchanged.")
                corners[corner_index] = (corner, corner)
                continue

            previous_direction = Edge.Direction(previous_edge, mantissa=None)
            next_direction = Edge.Direction(next_edge, mantissa=None)

            if (
                not isinstance(previous_direction, (list, tuple))
                or not isinstance(next_direction, (list, tuple))
            ):
                return None

            before = Topology.TranslateByDirectionDistance(
                corner,
                direction=Vector.Reverse(previous_direction),
                distance=corner_offset,
            )
            after = Topology.TranslateByDirectionDistance(
                corner,
                direction=next_direction,
                distance=corner_offset,
            )

            if not Topology.IsInstance(before, "Vertex") or not Topology.IsInstance(after, "Vertex"):
                return None

            corner_dictionary = Topology.Dictionary(corner, silent=True)
            if corner_dictionary:
                candidate = Topology.SetDictionary(before, corner_dictionary, silent=True)
                if Topology.IsInstance(candidate, "Vertex"):
                    before = candidate
                candidate = Topology.SetDictionary(after, corner_dictionary, silent=True)
                if Topology.IsInstance(candidate, "Vertex"):
                    after = candidate

            end_trim[previous_index] = corner_offset
            start_trim[next_index] = corner_offset
            corners[corner_index] = (before, after)

        vertices = []

        if closed:
            before_zero, after_zero = corners.get(
                0,
                (
                    Edge.StartVertex(edges[0], silent=True),
                    Edge.StartVertex(edges[0], silent=True),
                ),
            )
            vertices.append(after_zero)

            for corner_index in range(1, edge_count):
                before, after = corners.get(
                    corner_index,
                    (
                        Edge.StartVertex(edges[corner_index], silent=True),
                        Edge.StartVertex(edges[corner_index], silent=True),
                    ),
                )
                vertices.extend([before, after])

            vertices.append(before_zero)

        else:
            vertices.append(Edge.StartVertex(edges[0], silent=True))

            for corner_index in range(1, edge_count):
                before, after = corners.get(
                    corner_index,
                    (
                        Edge.StartVertex(edges[corner_index], silent=True),
                        Edge.StartVertex(edges[corner_index], silent=True),
                    ),
                )
                if not Vertex.IsCoincident(vertices[-1], before, tolerance=tolerance, silent=True):
                    vertices.append(before)
                if not Vertex.IsCoincident(vertices[-1], after, tolerance=tolerance, silent=True):
                    vertices.append(after)

            end_vertex = Edge.EndVertex(edges[-1], silent=True)
            if not Vertex.IsCoincident(vertices[-1], end_vertex, tolerance=tolerance, silent=True):
                vertices.append(end_vertex)

        filtered_vertices = []
        for vertex in vertices:
            if not Topology.IsInstance(vertex, "Vertex"):
                continue
            if (
                not filtered_vertices
                or not Vertex.IsCoincident(
                    filtered_vertices[-1],
                    vertex,
                    tolerance=tolerance,
                    silent=True,
                )
            ):
                filtered_vertices.append(vertex)

        if (
            closed
            and len(filtered_vertices) > 1
            and Vertex.IsCoincident(
                filtered_vertices[0],
                filtered_vertices[-1],
                tolerance=tolerance,
                silent=True,
            )
        ):
            filtered_vertices.pop()

        minimum = 3 if closed else 2
        if len(filtered_vertices) < minimum:
            return wire

        result = Wire.ByVertices(
            filtered_vertices,
            close=closed,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            if not silent:
                print("Wire.Miter - Error: Could not construct the mitered Wire. Returning None.")
            return None

        wire_dictionary = Topology.Dictionary(wire, silent=True)
        if wire_dictionary:
            candidate = Topology.SetDictionary(result, wire_dictionary, silent=True)
            if Topology.IsInstance(candidate, "Wire"):
                result = candidate

        return result
    
    @staticmethod
    def Normal(wire, outputType="xyz", mantissa=6):
        """
        Returns the normal vector to the input wire. A normal vector of a wire is a vector perpendicular to it.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. Default is "xyz".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

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
    def OrientEdges(wire, vertexA, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Returns a head-to-tail wire beginning at ``vertexA`` without flattening curves."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertexA, "Vertex"):
            return None
        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            return None
        ordered = Wire._OrderedEdges(wire, startVertex=vertexA, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list):
            if not silent:
                print("Wire.OrientEdges - Error: Could not orient all edges without altering their geometry. Returning None.")
            return None
        result = Wire.ByEdges(ordered, orient=False, transferDictionaries=transferDictionaries, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Wire"):
            return None
        if transferDictionaries:
            d = Topology.Dictionary(wire, silent=True)
            if d:
                candidate = Topology.SetDictionary(result, d, silent=True)
                if Topology.IsInstance(candidate, "Wire"):
                    result = candidate
        return result

    @staticmethod
    def Planarize(
        wire,
        origin=None,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns a planarized version of the input Wire while preserving curve geometry.

        A best-fit plane normal is derived from the Wire vertices. If ``origin`` is
        None, the centroid of the Wire is used as the plane origin. On the PythonOCC
        backend the complete Wire is projected normally onto that plane using OCCT's
        native normal-projection algorithm, preserving lines, circular arcs, B-splines,
        and NURBS curves. On backends without native curve projection, an exact fallback
        is used only for polylines.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        origin : topologic_core.Vertex , optional
            The origin of the receiving plane. If None, the centroid of the input
            Wire is used. Default is None.
        mantissa : int , optional
            The number of decimal places used to derive the plane equation.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The planarized Wire, or None if exact curve preservation is unavailable.

        """
        import math

        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Planarize - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Wire.Planarize - Error: The input tolerance parameter is invalid. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Planarize - Error: Could not determine a valid plane origin. Returning None.")
            return None

        vertices = Topology.Vertices(wire, silent=True) or []
        if len(vertices) < 3:
            if not silent:
                print("Wire.Planarize - Error: At least three Wire vertices are required to determine a plane. Returning None.")
            return None

        plane_equation = Vertex.PlaneEquation(vertices, mantissa=mantissa)
        if not isinstance(plane_equation, dict):
            if not silent:
                print("Wire.Planarize - Error: Could not determine a best-fit plane. Returning None.")
            return None

        try:
            normal = [
                float(plane_equation["a"]),
                float(plane_equation["b"]),
                float(plane_equation["c"]),
            ]
            magnitude = math.sqrt(sum(value * value for value in normal))
            if not math.isfinite(magnitude) or magnitude <= tolerance:
                raise ValueError
            normal = [value / magnitude for value in normal]
        except Exception:
            if not silent:
                print("Wire.Planarize - Error: Could not determine a valid plane normal. Returning None.")
            return None

        # Size the finite receiving Face from the actual OCCT bounds when possible.
        # This is important for curved Edges whose extrema can lie well beyond their
        # topological end vertices. The generous factor also allows a user-supplied
        # plane origin that is not at the Wire centroid.
        plane_size = 1.0
        try:
            is_topologic_core = bool(Topology._IsTopologicCoreBackend())
        except Exception:
            is_topologic_core = True

        if not is_topologic_core:
            try:
                from OCC.Core.Bnd import Bnd_Box
                from OCC.Core.BRepBndLib import brepbndlib

                source_shape = getattr(wire, "shape", None)
                if source_shape is not None and not source_shape.IsNull():
                    box = Bnd_Box()
                    brepbndlib.Add(source_shape, box)
                    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
                    ox, oy, oz = Vertex.Coordinates(origin, mantissa=None)
                    corners = [
                        (x, y, z)
                        for x in (xmin, xmax)
                        for y in (ymin, ymax)
                        for z in (zmin, zmax)
                    ]
                    max_distance = max(
                        math.sqrt((x-ox)**2 + (y-oy)**2 + (z-oz)**2)
                        for x, y, z in corners
                    )
                    if math.isfinite(max_distance):
                        plane_size = max(1.0, 4.0 * max_distance, 1000.0 * tolerance)
            except Exception:
                pass

        if plane_size <= 1.0:
            try:
                distances = [Vertex.Distance(origin, vertex) for vertex in vertices]
                distances = [float(value) for value in distances if value is not None]
                if distances:
                    plane_size = max(1.0, 4.0 * max(distances), 1000.0 * tolerance)
            except Exception:
                plane_size = 1.0

        plane_face = Face.RectangleByPlaneEquation(
            origin=origin,
            width=plane_size,
            length=plane_size,
            equation=plane_equation,
            tolerance=tolerance,
        )
        if not Topology.IsInstance(plane_face, "Face"):
            if not silent:
                print("Wire.Planarize - Error: Could not construct the receiving plane. Returning None.")
            return None

        if not is_topologic_core:
            try:
                from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_NormalProjection
                from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopoDS import topods

                source_shape = getattr(wire, "shape", None)
                target_shape = getattr(plane_face, "shape", None)

                if source_shape is not None and target_shape is not None:
                    projector = BRepOffsetAPI_NormalProjection(target_shape)
                    projector.Add(source_shape)
                    projector.SetLimit(False)
                    projector.Compute3d(True)
                    projector.Build()

                    if not hasattr(projector, "IsDone") or projector.IsDone():
                        projected_shape = projector.Projection()

                        if projected_shape is not None and not projected_shape.IsNull():
                            # Prefer native projected wires when OCCT returns them.
                            projected_wires = []
                            explorer = TopExp_Explorer(projected_shape, TopAbs_WIRE)
                            while explorer.More():
                                occ_wire = topods.Wire(explorer.Current())
                                candidate = None
                                try:
                                    if Core.HasAttribute("Wire", "ByOcctShape"):
                                        candidate = Core.Wire.ByOcctShape(occ_wire)
                                except Exception:
                                    candidate = None
                                if Topology.IsInstance(candidate, "Wire"):
                                    projected_wires.append(candidate)
                                explorer.Next()

                            if len(projected_wires) == 1:
                                result = projected_wires[0]
                            else:
                                projected_edges = []
                                if projected_wires:
                                    for projected_wire in projected_wires:
                                        projected_edges.extend(Wire.Edges(projected_wire, silent=True) or [])
                                else:
                                    explorer = TopExp_Explorer(projected_shape, TopAbs_EDGE)
                                    while explorer.More():
                                        occ_edge = topods.Edge(explorer.Current())
                                        candidate = None
                                        try:
                                            if Core.HasAttribute("Edge", "ByOcctShape"):
                                                candidate = Core.Edge.ByOcctShape(occ_edge)
                                        except Exception:
                                            candidate = None
                                        if Topology.IsInstance(candidate, "Edge"):
                                            projected_edges.append(candidate)
                                        explorer.Next()

                                result = Wire.ByEdges(
                                    projected_edges,
                                    orient=True,
                                    tolerance=tolerance,
                                    silent=True,
                                ) if projected_edges else None

                            if Topology.IsInstance(result, "Wire"):
                                dictionary = Topology.Dictionary(wire, silent=True)
                                if dictionary:
                                    updated = Topology.SetDictionary(result, dictionary, silent=True)
                                    if Topology.IsInstance(updated, "Wire"):
                                        result = updated
                                return result
            except Exception:
                pass

        # Exact fallback for polylines: orthogonal projection of a straight segment
        # onto a plane is still a straight segment. Curved Edges are never chorded.
        if Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            source_edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
            if not isinstance(source_edges, list) or not source_edges:
                return None

            def project_vertex(vertex):
                projected = Vertex.Project(
                    vertex,
                    plane_face,
                    direction=normal,
                    mantissa=mantissa,
                    tolerance=tolerance,
                )
                if Topology.IsInstance(projected, "Vertex"):
                    return projected
                return Vertex.Project(
                    vertex,
                    plane_face,
                    direction=[-normal[0], -normal[1], -normal[2]],
                    mantissa=mantissa,
                    tolerance=tolerance,
                )

            projected_edges = []
            for edge in source_edges:
                start = project_vertex(Edge.StartVertex(edge, silent=True))
                end = project_vertex(Edge.EndVertex(edge, silent=True))
                if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                    return None

                projected = Edge.ByStartVertexEndVertex(
                    start,
                    end,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(projected, "Edge"):
                    return None
                projected_edges.append(projected)

            result = Wire.ByEdges(
                projected_edges,
                orient=True,
                tolerance=tolerance,
                silent=silent,
            )
            if Topology.IsInstance(result, "Wire"):
                dictionary = Topology.Dictionary(wire, silent=True)
                if dictionary:
                    updated = Topology.SetDictionary(result, dictionary, silent=True)
                    if Topology.IsInstance(updated, "Wire"):
                        result = updated
            return result

        if not silent:
            print("Wire.Planarize - Error: The active backend could not planarize this curved Wire without approximating it. Returning None.")
        return None


    @staticmethod
    def Project(
        wire,
        face,
        direction: list = None,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Projects the input Wire onto the input Face while preserving curve geometry.

        On the PythonOCC backend the complete Wire is projected natively using
        OpenCascade's cylindrical projection algorithm. Lines, circular arcs,
        B-splines, and NURBS curves therefore remain genuine curves. On backends
        without a native curve projection, a fallback is used only when both the
        receiving Face is planar and the source Wire is a polyline, because in
        that case projecting each linear Edge by its endpoints is exact.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        face : topologic_core.Face
            The receiving Face.
        direction : list , optional
            Projection direction. If None, the reverse of the receiving Face
            normal is used. Default is None.
        mantissa : int , optional
            The number of decimal places used by Face.Normal. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The projected Wire, or None if an exact curve-preserving projection
            cannot be constructed.
        """
        import math

        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Project - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Wire.Project - Error: The input face parameter is not a valid Face. Returning None.")
            return None

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Wire.Project - Error: The input tolerance parameter is invalid. Returning None.")
            return None

        if direction is None:
            normal = Face.Normal(face, outputType="xyz", mantissa=mantissa)
            try:
                direction = [-float(normal[0]), -float(normal[1]), -float(normal[2])]
            except Exception:
                if not silent:
                    print("Wire.Project - Error: Could not determine the receiving Face normal. Returning None.")
                return None

        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
            magnitude = math.sqrt(sum(value * value for value in direction))
        except Exception:
            magnitude = 0.0

        if not math.isfinite(magnitude) or magnitude <= tolerance:
            if not silent:
                print("Wire.Project - Error: The input direction is not a valid non-zero vector. Returning None.")
            return None

        direction = [value / magnitude for value in direction]

        def wrap_occ_wire(occ_wire):
            result = None
            try:
                if Core.HasAttribute("Wire", "ByOcctShape"):
                    result = Core.Wire.ByOcctShape(occ_wire)
            except Exception:
                result = None
            if not Topology.IsInstance(result, "Wire"):
                try:
                    if Core.HasAttribute("Topology", "ByOcctShape"):
                        result = Core.Topology.ByOcctShape(occ_wire)
                except Exception:
                    result = None
            return result if Topology.IsInstance(result, "Wire") else None

        # Native cylindrical projection preserves the actual Edge geometry.
        try:
            is_topologic_core = bool(Topology._IsTopologicCoreBackend())
        except Exception:
            is_topologic_core = True

        if not is_topologic_core:
            try:
                from OCC.Core.BRepProj import BRepProj_Projection
                from OCC.Core.gp import gp_Dir

                wire_shape = getattr(wire, "shape", None)
                face_shape = getattr(face, "shape", None)

                if wire_shape is not None and face_shape is not None:
                    projection = BRepProj_Projection(
                        wire_shape,
                        face_shape,
                        gp_Dir(direction[0], direction[1], direction[2]),
                    )

                    if projection.IsDone():
                        projected_wires = []
                        projection.Init()

                        while projection.More():
                            candidate = wrap_occ_wire(projection.Current())
                            if Topology.IsInstance(candidate, "Wire"):
                                projected_wires.append(candidate)
                            projection.Next()

                        if len(projected_wires) == 1:
                            return projected_wires[0]

                        if len(projected_wires) > 1:
                            all_edges = []
                            for projected_wire in projected_wires:
                                all_edges.extend(Wire.Edges(projected_wire, silent=True) or [])

                            merged = Wire.ByEdges(
                                all_edges,
                                orient=True,
                                tolerance=tolerance,
                                silent=True,
                            )
                            if Topology.IsInstance(merged, "Wire"):
                                return merged

                            if not silent:
                                print("Wire.Project - Error: The projection produced multiple disconnected Wire results. Returning None.")
                            return None
            except Exception:
                pass

        # Exact fallback: a constant-direction projection of a straight segment
        # onto a plane remains a straight segment. Do not use this fallback for
        # curved source Edges or non-planar receiving Faces.
        try:
            target_is_planar = bool(Topology.IsPlanar(face, tolerance=tolerance))
        except TypeError:
            try:
                target_is_planar = bool(Topology.IsPlanar(face))
            except Exception:
                target_is_planar = False
        except Exception:
            target_is_planar = False

        if target_is_planar and Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            source_edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
            if not isinstance(source_edges, list) or not source_edges:
                return None

            projected_edges = []
            for edge in source_edges:
                start = Vertex.Project(
                    Edge.StartVertex(edge, silent=True),
                    face,
                    direction=direction,
                    mantissa=mantissa,
                    tolerance=tolerance,
                )
                end = Vertex.Project(
                    Edge.EndVertex(edge, silent=True),
                    face,
                    direction=direction,
                    mantissa=mantissa,
                    tolerance=tolerance,
                )

                if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                    return None

                projected = Edge.ByStartVertexEndVertex(
                    start,
                    end,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(projected, "Edge"):
                    return None
                projected_edges.append(projected)

            return Wire.ByEdges(
                projected_edges,
                orient=True,
                tolerance=tolerance,
                silent=silent,
            )

        if not silent:
            print("Wire.Project - Error: The active backend could not project this Wire without approximating its curves. Returning None.")
        return None

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
    def RemoveCollinearEdges(
        wire,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Removes redundant consecutive linear collinear Edges while preserving curves.

        Curved Edges are never rebuilt or replaced. Only runs of consecutive,
        geometrically linear Edges that are collinear and continue in the same
        traversal direction are merged into one linear Edge.

        Parameters
        ----------
        wire : topologic_core.Wire or topologic_core.Cluster
            The input Wire, or a Cluster containing Wires.
        angTolerance : float , optional
            Maximum angular deviation in degrees for two consecutive linear Edges
            to be treated as a continuation. Default is 0.1.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology
            The simplified Wire, or a merged topology if a Cluster/non-manifold
            input produces multiple components.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
            angTolerance = max(abs(float(angTolerance)), 0.0)
        except Exception:
            if not silent:
                print("Wire.RemoveCollinearEdges - Error: Invalid tolerance input. Returning None.")
            return None

        if Topology.IsInstance(wire, "Cluster"):
            wires = Topology.Wires(wire, silent=True) or []
            processed = [
                Wire.RemoveCollinearEdges(
                    item,
                    angTolerance=angTolerance,
                    tolerance=tolerance,
                    silent=silent,
                )
                for item in wires
            ]
            processed = [item for item in processed if item is not None]
            if not processed:
                if not silent:
                    print("Wire.RemoveCollinearEdges - Error: No valid Wires were produced. Returning None.")
                return None
            if len(processed) == 1:
                return processed[0]
            return Topology.SelfMerge(
                Cluster.ByTopologies(processed, silent=True),
                tolerance=tolerance,
            )

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.RemoveCollinearEdges - Error: The input is not a valid Wire. Returning None.")
            return None

        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            components = Wire.Split(wire) or []
            processed = []
            for component in components:
                if Topology.IsInstance(component, "Wire"):
                    item = Wire.RemoveCollinearEdges(
                        component,
                        angTolerance=angTolerance,
                        tolerance=tolerance,
                        silent=silent,
                    )
                    if item is not None:
                        processed.append(item)
                elif Topology.IsInstance(component, "Edge"):
                    processed.append(component)

            if not processed:
                return wire
            if len(processed) == 1:
                return processed[0]
            return Topology.SelfMerge(
                Cluster.ByTopologies(processed, silent=True),
                tolerance=tolerance,
            )

        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(edges, list) or not edges:
            return wire

        closed = bool(Wire.IsClosed(wire, tolerance=tolerance, silent=True))

        def mergeable(edge_a, edge_b):
            if not Edge.IsLinear(edge_a, tolerance=tolerance, silent=True):
                return False
            if not Edge.IsLinear(edge_b, tolerance=tolerance, silent=True):
                return False

            end_a = Edge.EndVertex(edge_a, silent=True)
            start_b = Edge.StartVertex(edge_b, silent=True)
            if not Topology.IsInstance(end_a, "Vertex") or not Topology.IsInstance(start_b, "Vertex"):
                return False
            if not Vertex.IsCoincident(end_a, start_b, tolerance=tolerance, silent=True):
                return False

            try:
                if not bool(Edge.IsCollinear(edge_a, edge_b, tolerance=tolerance)):
                    return False
            except Exception:
                return False

            angle = Edge.Angle(edge_a, edge_b, mantissa=12)
            if angle is None:
                return False
            try:
                return abs(float(angle)) <= angTolerance
            except Exception:
                return False

        # Rotate a closed Wire to begin immediately after a non-mergeable seam.
        # This keeps a mergeable linear run from being split across list ends.
        if closed and len(edges) > 1:
            break_index = None
            for index in range(len(edges)):
                previous = edges[index - 1]
                current = edges[index]
                if not mergeable(previous, current):
                    break_index = index
                    break
            if break_index is not None and break_index > 0:
                edges = edges[break_index:] + edges[:break_index]

        groups = [[edges[0]]]
        for edge in edges[1:]:
            if mergeable(groups[-1][-1], edge):
                groups[-1].append(edge)
            else:
                groups.append([edge])

        # If no non-mergeable seam exists on a closed Wire, leave the topology
        # untouched; collapsing an entire closed collinear cycle is undefined.
        if closed and len(groups) == 1 and len(groups[0]) == len(edges):
            return wire

        def merge_group(group):
            if len(group) == 1:
                return group[0]

            start = Edge.StartVertex(group[0], silent=True)
            end = Edge.EndVertex(group[-1], silent=True)
            if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                return None
            if Vertex.Distance(start, end) <= tolerance:
                return None

            merged = Edge.ByStartVertexEndVertex(
                start,
                end,
                tolerance=tolerance,
                silent=True,
            )
            if not Topology.IsInstance(merged, "Edge"):
                return None

            dictionaries = [
                Topology.Dictionary(edge, silent=True)
                for edge in group
            ]
            dictionaries = [dictionary for dictionary in dictionaries if dictionary]
            if dictionaries:
                try:
                    dictionary = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                    if dictionary:
                        candidate = Topology.SetDictionary(merged, dictionary, silent=True)
                        if Topology.IsInstance(candidate, "Edge"):
                            merged = candidate
                except Exception:
                    pass

            return merged

        new_edges = []
        changed = False

        for group in groups:
            merged = merge_group(group)
            if not Topology.IsInstance(merged, "Edge"):
                return wire
            new_edges.append(merged)
            changed = changed or len(group) > 1

        if not changed:
            return wire

        result = Wire.ByEdges(
            new_edges,
            orient=True,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            return wire

        wire_dictionary = Topology.Dictionary(wire, silent=True)
        if wire_dictionary:
            candidate = Topology.SetDictionary(result, wire_dictionary, silent=True)
            if Topology.IsInstance(candidate, "Wire"):
                result = candidate

        return result

    @staticmethod
    def Representation(
        wire,
        normalize: bool = True,
        rotate: bool = True,
        mantissa: int = 6,
        tolerance: float = 0.0001
    ):
        """
        Returns a normalized representation of a closed wire with alternating edge lengths and interior angles.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        normalize : bool , optional
            If set to True, the edge lengths are normalized such that the shortest
            edge has a length of 1. Default is True.
        rotate : bool , optional
            If set to True, the representation is rotated such that the shortest
            edge appears first. Default is True.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The representation list consisting of alternating edge lengths and
            interior angles.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Representation - Error: The input wire parameter is not a valid wire. Returning None.")
            return None

        if not Wire.IsClosed(wire):
            print("Wire.Representation - Error: The input wire parameter is not closed. Returning None.")
            return None

        if not Wire.IsManifold(wire):
            print("Wire.Representation - Error: The input wire parameter is non-manifold. Returning None.")
            return None

        edges = Topology.Edges(wire)
        if not isinstance(edges, list) or len(edges) < 3:
            print("Wire.Representation - Error: Could not retrieve a valid list of edges from the input wire. Returning None.")
            return None

        angles = Wire.InteriorAngles(
            wire,
            tolerance=tolerance,
            mantissa=mantissa
        )
        if not isinstance(angles, list) or len(angles) != len(edges):
            print("Wire.Representation - Error: Could not compute the interior angles of the input wire. Returning None.")
            return None

        lengths = [Edge.Length(edge) for edge in edges]

        if normalize:
            min_length = min(lengths)
            if min_length <= tolerance:
                print("Wire.Representation - Error: The input wire contains a zero-length edge. Returning None.")
                return None
            lengths = [length / min_length for length in lengths]

        # Keep each edge length paired with the interior angle at the end
        # of that edge. This guarantees a representation of length 2*N.
        pairs = list(zip(lengths, angles))

        if rotate and pairs:
            # Rotate using edge length only. Do not search the flattened
            # representation because angle values must not affect the rotation.
            min_index = min(range(len(lengths)), key=lambda i: lengths[i])
            pairs = pairs[min_index:] + pairs[:min_index]

        representation = []
        for length, angle in pairs:
            representation.append(round(length, mantissa))
            representation.append(round(angle, mantissa))

        return representation

    @staticmethod
    def Reverse(wire, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Reverses wire traversal while preserving each edge's actual geometry."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Reverse - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list):
            return None
        reversed_edges = []
        for source in reversed(ordered):
            edge = Edge.Reverse(source, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(edge, "Edge"):
                if not silent:
                    print("Wire.Reverse - Error: An edge could not be reversed without altering its geometry. Returning None.")
                return None
            if transferDictionaries:
                d = Topology.Dictionary(source, silent=True)
                if d:
                    candidate = Topology.SetDictionary(edge, d, silent=True)
                    if Topology.IsInstance(candidate, "Edge"):
                        edge = candidate
            reversed_edges.append(edge)
        result = Wire.ByEdges(reversed_edges, orient=False, transferDictionaries=transferDictionaries, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Wire"):
            return None
        if transferDictionaries:
            d = Topology.Dictionary(wire, silent=True)
            if d:
                candidate = Topology.SetDictionary(result, d, silent=True)
                if Topology.IsInstance(candidate, "Wire"):
                    result = candidate
        return result

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
    def Roof(face, angle: float = 45, boundary: bool = True, tolerance: float = 0.001):
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
    def Simplify(wire, method="douglas-peucker", tolerance=0.0001, silent=False):
        """
        Simplifies a polyline Wire using a point-based simplification algorithm.

        ``Wire.Simplify`` is intentionally a linear/polyline operation. Curved
        Edges are rejected rather than silently converted to endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input Wire.
        method : str , optional
            One of ``"douglas-peucker"``, ``"visvalingam-whyatt"``, or
            ``"reumann-witkam"``. Default is ``"douglas-peucker"``.
        tolerance : float , optional
            Algorithm tolerance. Default is 0.0001.
        silent : bool , optional
            If True, suppress diagnostics. Default is False.

        Returns
        -------
        topologic_core.Wire
            The simplified polyline Wire.
        """
        import math

        from topologicpy.Cluster import Cluster
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        def perpendicular_distance(point, line_start, line_end):
            p = Vertex.Coordinates(point, mantissa=None)
            a = Vertex.Coordinates(line_start, mantissa=None)
            b = Vertex.Coordinates(line_end, mantissa=None)

            if p is None or a is None or b is None:
                return float("inf")

            ab = [b[i] - a[i] for i in range(3)]
            ap = [p[i] - a[i] for i in range(3)]

            denominator = math.sqrt(sum(value * value for value in ab))
            if denominator <= 1.0e-12:
                return math.sqrt(sum(value * value for value in ap))

            cross = [
                ap[1] * ab[2] - ap[2] * ab[1],
                ap[2] * ab[0] - ap[0] * ab[2],
                ap[0] * ab[1] - ap[1] * ab[0],
            ]
            numerator = math.sqrt(sum(value * value for value in cross))
            return numerator / denominator

        def douglas_peucker(points, tol):
            if len(points) <= 2:
                return list(points)

            start_point = points[0]
            end_point = points[-1]
            max_distance = 0.0
            max_index = 0

            for index in range(1, len(points) - 1):
                distance = perpendicular_distance(
                    points[index],
                    start_point,
                    end_point,
                )
                if distance > max_distance:
                    max_distance = distance
                    max_index = index

            if max_distance <= tol:
                return [start_point, end_point]

            first = douglas_peucker(points[: max_index + 1], tol)
            second = douglas_peucker(points[max_index:], tol)
            return first[:-1] + second

        def triangle_area_3d(a, b, c):
            pa = Vertex.Coordinates(a, mantissa=None)
            pb = Vertex.Coordinates(b, mantissa=None)
            pc = Vertex.Coordinates(c, mantissa=None)
            if pa is None or pb is None or pc is None:
                return float("inf")

            ab = [pb[i] - pa[i] for i in range(3)]
            ac = [pc[i] - pa[i] for i in range(3)]
            cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            return 0.5 * math.sqrt(sum(value * value for value in cross))

        def visvalingam_whyatt(points, tol):
            if len(points) <= 2:
                return list(points)

            remove_indices = set()
            for index in range(1, len(points) - 1):
                if triangle_area_3d(
                    points[index - 1],
                    points[index],
                    points[index + 1],
                ) <= tol:
                    remove_indices.add(index)

            return [
                point
                for index, point in enumerate(points)
                if index not in remove_indices
            ]

        def reumann_witkam(points, tol):
            if len(points) <= 2:
                return list(points)

            simplified = [points[0]]
            start_point = points[0]
            index = 1

            while index < len(points) - 1:
                end_point = points[index]
                next_point = points[index + 1]

                if perpendicular_distance(
                    next_point,
                    start_point,
                    end_point,
                ) > tol:
                    simplified.append(end_point)
                    start_point = end_point

                index += 1

            simplified.append(points[-1])
            return simplified

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Simplify - Error: The input wire parameter is not a valid Wire. Returning None.")
            return None

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Wire.Simplify - Error: The input tolerance parameter is invalid. Returning None.")
            return None

        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Simplify - Error: Wire.Simplify is a polyline-only operation and does not accept curved Edges. Returning None.")
            return None

        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            components = Wire.Split(wire) or []
            simplified_components = []

            for component in components:
                if Topology.IsInstance(component, "Edge"):
                    if Edge.Length(component, mantissa=None, tolerance=tolerance, silent=True) > tolerance:
                        simplified_components.append(component)
                elif Topology.IsInstance(component, "Wire"):
                    simplified = Wire.Simplify(
                        component,
                        method=method,
                        tolerance=tolerance,
                        silent=silent,
                    )
                    if simplified is not None:
                        simplified_components.append(simplified)

            if not simplified_components:
                return wire

            return Topology.SelfMerge(
                Cluster.ByTopologies(simplified_components, silent=True),
                tolerance=tolerance,
            )

        ordered_edges = Wire._OrderedEdges(
            wire,
            tolerance=tolerance,
            silent=True,
        )
        if not isinstance(ordered_edges, list) or not ordered_edges:
            return wire

        closed = bool(Wire.IsClosed(wire, tolerance=tolerance, silent=True))

        points = [
            Edge.StartVertex(edge, silent=True)
            for edge in ordered_edges
        ]
        if not closed:
            points.append(Edge.EndVertex(ordered_edges[-1], silent=True))

        if len(points) < 2:
            return wire

        method_name = str(method).lower()

        if "douglas" in method_name:
            new_vertices = douglas_peucker(points, tolerance)
        elif "vis" in method_name:
            new_vertices = visvalingam_whyatt(points, tolerance)
        elif "reu" in method_name:
            new_vertices = reumann_witkam(points, tolerance)
        else:
            if not silent:
                print(
                    f"Wire.Simplify - Warning: Unknown method ({method}). "
                    "Defaulting to 'douglas-peucker'."
                )
            new_vertices = douglas_peucker(points, tolerance)

        # A closed polygon needs at least three distinct vertices; an open one
        # needs at least two.
        minimum = 3 if closed else 2
        if len(new_vertices) < minimum:
            if not silent:
                print("Wire.Simplify - Warning: Simplification removed too many vertices. Returning the original Wire.")
            return wire

        result = Wire.ByVertices(
            new_vertices,
            close=closed,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            if not silent:
                print("Wire.Simplify - Warning: Could not construct a simplified Wire. Returning the original Wire.")
            return wire

        wire_dictionary = Topology.Dictionary(wire, silent=True)
        if wire_dictionary:
            candidate = Topology.SetDictionary(result, wire_dictionary, silent=True)
            if Topology.IsInstance(candidate, "Wire"):
                result = candidate

        return result

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
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created straight skeleton.

        """
        if not Topology.IsInstance(face, "Face"):
            return None
        return Wire.Roof(face, angle=0, boundary=boundary, tolerance=tolerance)
    
    @staticmethod
    def Spiral(origin=None, radiusA: float = 0.05, radiusB: float = 0.5, height: float = 1.0, turns: int = 10, sides: int = 36, clockwise: bool = False, reverse: bool = False, direction: list = [0, 0, 1], placement: str = "center", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates an Archimedean spiral Wire by delegating the single curve to Edge.Spiral."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        edge = Edge.Spiral(origin=origin, radiusA=radiusA, radiusB=radiusB, height=height, turns=turns, clockwise=clockwise, reverse=reverse, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)


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
    def Square(origin= None, size: float = 1.0, diagonals= False, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. Default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. Default is 1.0.
        diagonals : bool , optional
            If set to True, the diagonals of the rectangle are included. Diagonals are split at the centroid of the rectangle. Default is False.
        direction : list , optional
            The vector representing the up direction of the square. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created square.

        """
        return Wire.Rectangle(origin=origin, width=size, length=size, diagonals=diagonals, direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Squircle(origin=None, radius: float = 0.5, sides: int = 121, a: float = 2.0, b: float = 2.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a squircle Wire by delegating the single smooth curve to Edge.Squircle."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        edge = Edge.Squircle(origin=origin, radius=radius, a=a, b=b, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)


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
    def StartEndVertices(wire, silent: bool = False) -> list:
        """
        Returns the start and end vertices of the input wire. The wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of start and end vertices of the input wire

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "wire"):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        
        if Wire.IsClosed(wire):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not an open wire. Returning None.")
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
    def StartVertex(wire, silent: bool = False):
        """
        Returns the start vertex of the input wire. The wire must be manifold and open.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Vertex
            The start vertex of the input wire.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(wire, "wire"):
            if not silent:
                print("Wire.StartVertex - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.StartVertex - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        
        if Wire.IsClosed(wire):
            if not silent:
                print("Wire.StartVertex - Error: The input wire parameter is not an open wire. Returning None.")
            return None
        sv, ev = Wire.StartEndVertices(wire, silent=silent)
        return sv

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
        """Returns curvilinear distance along a simple wire between ``origin`` and ``vertex``."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertex, "Vertex"):
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list) or not ordered:
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Edge.StartVertex(ordered[0], silent=True)
        d1 = Wire._DistanceFromStart(wire, vertex, tolerance=tolerance, silent=True)
        d0 = Wire._DistanceFromStart(wire, origin, tolerance=tolerance, silent=True)
        if d1 is None or d0 is None:
            return None
        value = abs(float(d1)-float(d0))
        return value if mantissa is None else round(value, int(mantissa))

    @staticmethod
    def VertexByDistance(wire, distance: float = 0.0, origin=None, tolerance: float = 0.0001, silent: bool = False):
        """Creates a vertex at signed curvilinear distance along an open simple wire."""
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            return None
        try:
            distance = float(distance)
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return None
        if not math.isfinite(distance):
            return None
        if not Wire.IsManifold(wire, tolerance=tol, silent=True) or Wire.IsClosed(wire, tolerance=tol, silent=True):
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tol, silent=True)
        if not isinstance(ordered, list) or not ordered:
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tol, silent=True)
        if total is None or float(total) <= tol:
            return None
        total = float(total)
        start = Edge.StartVertex(ordered[0], silent=True)
        end = Edge.EndVertex(ordered[-1], silent=True)
        if not Topology.IsInstance(origin, "Vertex"):
            origin = start
        origin_distance = Wire._DistanceFromStart(wire, origin, tolerance=tol, silent=True)
        if origin_distance is None:
            return None
        if Vertex.IsCoincident(origin, end, tolerance=tol, silent=True):
            target = total - distance
        else:
            target = float(origin_distance) + distance
        if target < -tol or target > total + tol:
            return None
        return Wire._VertexAtDistanceFromStart(wire, max(0.0, min(total, target)), tolerance=tol, silent=silent)
    


    @staticmethod
    def Parabola(origin=None, focalLength: float = 0.5, fromParameter: float = -1.0, toParameter: float = 1.0, sides: int = 16, direction: list = [0, 0, 1], placement: str = "vertex", polyline: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """Creates a parabolic Wire from one exact Edge.Parabola and Wire.ByEdge."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        edge = Edge.Parabola(origin=origin, focalLength=focalLength, fromParameter=fromParameter, toParameter=toParameter, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(edge, "Edge"):
            return None
        return Wire.ByEdge(edge, sides=sides, polyline=polyline, silent=silent)

    @staticmethod
    def ParameterAtVertex(wire, vertex, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """Returns the normalized global arc-length parameter of a vertex on a simple wire."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertex, "Vertex"):
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tolerance, silent=True)
        distance = Wire._DistanceFromStart(wire, vertex, tolerance=tolerance, silent=True)
        if total is None or float(total) <= tolerance or distance is None:
            return None
        value = max(0.0, min(1.0, float(distance)/float(total)))
        return value if mantissa is None else round(value, int(mantissa))

    @staticmethod
    def VertexByParameter(wire, u: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """Creates a vertex at a normalized global arc-length parameter on a simple wire."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(wire, "Wire"):
            return None
        try:
            u = float(u)
        except Exception:
            return None
        if u < 0.0 or u > 1.0:
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tolerance, silent=True)
        if total is None or float(total) <= tolerance:
            return None
        return Wire._VertexAtDistanceFromStart(wire, u*float(total), tolerance=tolerance, silent=silent)

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
        # _ = wire.Vertices(None, vertices) # H to Core
        try:
            _ = Core.InstanceCall(wire, "Vertices", None, vertices)
        except Exception:
            vertices = None
        return vertices

