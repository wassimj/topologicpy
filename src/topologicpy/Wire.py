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
    def _OrderedEdges(wire, startVertex=None, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the constituent edges of a simple manifold wire in oriented head-to-tail order.

        Actual edge geometry is preserved. When an input edge must be reoriented, the
        method delegates to :meth:`Edge.Reverse`; it never reconstructs an edge from
        its endpoint chord. Branched or disconnected wires return None.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        startVertex : topologic_core.Vertex , optional
            The desired traversal start. For an open wire this must be an endpoint.
            For a closed wire it may be any junction vertex. If None, a deterministic
            start consistent with the stored edge orientations is selected.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The ordered list of actual edges, or None if one continuous traversal
            cannot be established.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001

        edges = Wire.Edges(wire, silent=True) or []
        edges = [edge for edge in edges if Topology.IsInstance(edge, "Edge")]
        if len(edges) == 0:
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire contains no valid edges. Returning None.")
            return None

        representatives = []
        edge_nodes = []
        adjacency = {}

        def node_index(vertex):
            for i, representative in enumerate(representatives):
                if Vertex.IsCoincident(vertex, representative, tolerance=tolerance, silent=True):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1

        for edge_index, edge in enumerate(edges):
            start = Edge.StartVertex(edge, silent=True)
            end = Edge.EndVertex(edge, silent=True)
            if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                return None
            a = node_index(start)
            b = node_index(end)
            edge_nodes.append((a, b))
            adjacency.setdefault(a, []).append(edge_index)
            adjacency.setdefault(b, []).append(edge_index)

        if any(len(indices) > 2 for indices in adjacency.values()):
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire is non-manifold and has no unique path ordering. Returning None.")
            return None

        open_nodes = [node for node, indices in adjacency.items() if len(indices) == 1]
        if len(open_nodes) not in (0, 2):
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire is disconnected or does not form one simple path/cycle. Returning None.")
            return None
        closed = len(open_nodes) == 0

        start_node = None
        if Topology.IsInstance(startVertex, "Vertex"):
            for i, representative in enumerate(representatives):
                if Vertex.IsCoincident(startVertex, representative, tolerance=tolerance, silent=True):
                    start_node = i
                    break
            if start_node is None or (not closed and start_node not in open_nodes):
                if not silent:
                    print("Wire._OrderedEdges - Error: The requested startVertex is not a valid traversal start. Returning None.")
                return None
        elif closed:
            first_start = Edge.StartVertex(edges[0], silent=True)
            start_node = node_index(first_start)
        else:
            # Prefer the endpoint from which its sole incident edge is already oriented.
            first, second = open_nodes
            first_edge = edges[adjacency[first][0]]
            second_edge = edges[adjacency[second][0]]
            first_forward = Vertex.IsCoincident(
                Edge.StartVertex(first_edge, silent=True), representatives[first], tolerance=tolerance, silent=True
            )
            second_forward = Vertex.IsCoincident(
                Edge.StartVertex(second_edge, silent=True), representatives[second], tolerance=tolerance, silent=True
            )
            if first_forward and not second_forward:
                start_node = first
            elif second_forward and not first_forward:
                start_node = second
            else:
                start_node = min(open_nodes)

        ordered = []
        used = set()
        current_node = start_node

        while len(used) < len(edges):
            candidates = [index for index in adjacency.get(current_node, []) if index not in used]
            if not candidates:
                break

            # At the first vertex of a closed cycle two unused edges are incident.
            # Prefer an edge already oriented away from the chosen start.
            selected = None
            if len(candidates) > 1:
                forward_candidates = []
                for index in candidates:
                    a, _ = edge_nodes[index]
                    if a == current_node:
                        forward_candidates.append(index)
                selected = forward_candidates[0] if forward_candidates else candidates[0]
            else:
                selected = candidates[0]

            source = edges[selected]
            a, b = edge_nodes[selected]
            if a == current_node:
                oriented = source
                next_node = b
            elif b == current_node:
                oriented = Edge.Reverse(source, tolerance=tolerance, silent=True)
                next_node = a
            else:
                return None

            if not Topology.IsInstance(oriented, "Edge"):
                if not silent:
                    print("Wire._OrderedEdges - Error: An edge could not be reoriented without altering its geometry. Returning None.")
                return None

            ordered.append(oriented)
            used.add(selected)
            current_node = next_node

        if len(used) != len(edges):
            if not silent:
                print("Wire._OrderedEdges - Error: The input wire is disconnected. Returning None.")
            return None
        if closed and current_node != start_node:
            return None
        if not closed and current_node not in open_nodes:
            return None
        return ordered

    @staticmethod
    def _DistanceFromStart(wire, vertex, tolerance: float = 0.0001, silent: bool = False):
        """Returns exact curvilinear distance from the traversal start to a vertex on a simple wire."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=silent)
        if not isinstance(edges, list) or len(edges) == 0:
            return None

        accumulated = 0.0
        for edge in edges:
            edge_length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
            if edge_length is None:
                return None
            edge_length = float(edge_length)
            u = Edge.ParameterAtVertex(
                edge,
                vertex,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )
            if u is not None:
                try:
                    u = max(0.0, min(1.0, float(u)))
                except Exception:
                    return None
                if u <= 1.0e-12:
                    return accumulated
                if u >= 1.0 - 1.0e-12:
                    return accumulated + edge_length
                portion = Edge.TrimByParameters(
                    edge,
                    uA=0.0,
                    uB=u,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(portion, "Edge"):
                    return None
                local_length = Edge.Length(portion, mantissa=None, tolerance=tolerance, silent=True)
                return None if local_length is None else accumulated + float(local_length)
            accumulated += edge_length
        return None

    @staticmethod
    def _VertexAtDistanceFromStart(wire, distance: float, tolerance: float = 0.0001, silent: bool = False):
        """Returns a vertex at exact curvilinear distance from the traversal start of a simple wire."""
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        try:
            distance = float(distance)
        except Exception:
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=silent)
        if not isinstance(edges, list) or len(edges) == 0:
            return None
        lengths = []
        total = 0.0
        for edge in edges:
            length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
            if length is None:
                return None
            length = float(length)
            lengths.append(length)
            total += length
        if total <= tolerance or distance < -tolerance or distance > total + tolerance:
            return None
        distance = max(0.0, min(total, distance))
        if distance <= tolerance:
            return Edge.StartVertex(edges[0], silent=True)
        if abs(distance - total) <= tolerance:
            return Edge.EndVertex(edges[-1], silent=True)
        accumulated = 0.0
        for edge, length in zip(edges, lengths):
            if distance <= accumulated + length + tolerance:
                local = max(0.0, min(length, distance - accumulated))
                if local <= tolerance:
                    return Edge.StartVertex(edge, silent=True)
                if abs(local - length) <= tolerance:
                    return Edge.EndVertex(edge, silent=True)
                return Edge.VertexByDistance(
                    edge,
                    distance=local,
                    origin=Edge.StartVertex(edge, silent=True),
                    mantissa=None,
                    tolerance=tolerance,
                    silent=True,
                )
            accumulated += length
        return None



    @staticmethod
    def _ConicEdge(center, axisA, axisB, fromAngle: float, toAngle: float, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates one exact rational quadratic conic Edge over an angular interval.

        The conic is the affine image of the unit circle defined by
        ``center + axisA*cos(t) + axisB*sin(t)``. Circular and elliptical arcs
        are therefore represented exactly. Angular spans larger than 90 degrees
        are internally split into rational quadratic spans while remaining one
        topological Edge.

        Parameters
        ----------
        center : topologic_core.Vertex
            The center of the conic in 3D.
        axisA : list
            The first 3D semi-axis vector.
        axisB : list
            The second 3D semi-axis vector.
        fromAngle : float
            Start angle in degrees.
        toAngle : float
            End angle in degrees. Values smaller than ``fromAngle`` are advanced
            by 360 degrees until a positive sweep is obtained.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The exact conic Edge, or None if construction fails.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(center, "Vertex"):
            if not silent:
                print("Wire._ConicEdge - Error: The input center parameter is not a valid vertex. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            axisA = [float(value) for value in axisA]
            axisB = [float(value) for value in axisB]
        except Exception:
            if not silent:
                print("Wire._ConicEdge - Error: One or more input parameters are invalid. Returning None.")
            return None

        if (
            not math.isfinite(tolerance)
            or tolerance <= 0.0
            or len(axisA) != 3
            or len(axisB) != 3
            or not all(math.isfinite(value) for value in axisA + axisB + [fromAngle, toAngle])
        ):
            if not silent:
                print("Wire._ConicEdge - Error: One or more input parameters are invalid. Returning None.")
            return None

        magnitudeA = math.sqrt(sum(value * value for value in axisA))
        magnitudeB = math.sqrt(sum(value * value for value in axisB))
        if magnitudeA <= tolerance or magnitudeB <= tolerance:
            if not silent:
                print("Wire._ConicEdge - Error: The conic axes are degenerate. Returning None.")
            return None

        while toAngle < fromAngle:
            toAngle += 360.0
        sweep = toAngle - fromAngle
        if sweep <= 1.0e-12 or sweep > 360.0 + 1.0e-9:
            if not silent:
                print("Wire._ConicEdge - Error: The angular sweep must be greater than zero and no greater than 360 degrees. Returning None.")
            return None

        cx = float(Vertex.X(center))
        cy = float(Vertex.Y(center))
        cz = float(Vertex.Z(center))

        span_count = max(1, int(math.ceil(sweep / 90.0)))
        span_angle = sweep / float(span_count)
        control_points = []
        weights = []

        def point(angle_radians, scale=1.0):
            c = math.cos(angle_radians) * scale
            s = math.sin(angle_radians) * scale
            return Vertex.ByCoordinates(
                cx + axisA[0] * c + axisB[0] * s,
                cy + axisA[1] * c + axisB[1] * s,
                cz + axisA[2] * c + axisB[2] * s,
            )

        for i in range(span_count):
            a0 = math.radians(fromAngle + i * span_angle)
            a1 = math.radians(fromAngle + (i + 1) * span_angle)
            am = 0.5 * (a0 + a1)
            weight = math.cos(0.5 * (a1 - a0))
            if weight <= 0.0:
                if not silent:
                    print("Wire._ConicEdge - Error: Could not compute a valid rational conic representation. Returning None.")
                return None

            p0 = point(a0)
            p1 = point(am, scale=1.0 / weight)
            p2 = point(a1)
            if not all(Topology.IsInstance(vertex, "Vertex") for vertex in [p0, p1, p2]):
                return None

            if i == 0:
                control_points.append(p0)
                weights.append(1.0)
            control_points.append(p1)
            weights.append(weight)
            control_points.append(p2)
            weights.append(1.0)

        knots = [0.0, 0.0, 0.0]
        for i in range(1, span_count):
            knot = float(i) / float(span_count)
            knots.extend([knot, knot])
        knots.extend([1.0, 1.0, 1.0])

        edge = Edge.ByNurbsParameters(
            controlPoints=control_points,
            weights=weights,
            knots=knots,
            isRational=True,
            isPeriodic=False,
            degree=2,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(edge, "Edge") and not silent:
            print("Wire._ConicEdge - Error: Could not create the conic edge. Returning None.")
        return edge if Topology.IsInstance(edge, "Edge") else None


    @staticmethod
    def Arc(startVertex,
            middleVertex,
            endVertex,
            sides: int = 16,
            close: bool = True,
            polyline: bool = True,
            tolerance: float = 0.0001,
            silent: bool = False):
        """
        Creates a circular arc Wire through three input vertices.

        When ``polyline`` is False, the Wire is composed of exact rational
        circular-arc Edges. ``sides`` specifies the number of curved Edge
        subtopologies used for the complete arc. When ``polyline`` is True,
        the historical straight-edge approximation is produced instead.

        Parameters
        ----------
        startVertex : topologic_core.Vertex
            The start vertex of the arc.
        middleVertex : topologic_core.Vertex
            A vertex that the arc must pass through.
        endVertex : topologic_core.Vertex
            The end vertex of the arc.
        sides : int , optional
            Number of curved arc Edges when ``polyline`` is False, or number of
            straight segments when ``polyline`` is True. Default is 16.
        close : bool , optional
            If True, a straight chord is added from the arc end back to its start.
            Default is True.
        polyline : bool , optional
            If True, create the historical straight-edge approximation. If False,
            create exact circular curve Edges. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created arc Wire.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        for name, vertex in [
            ("startVertex", startVertex),
            ("middleVertex", middleVertex),
            ("endVertex", endVertex),
        ]:
            if not Topology.IsInstance(vertex, "Vertex"):
                if not silent:
                    print(f"Wire.Arc - Error: The input {name} parameter is not a valid vertex. Returning None.")
                return None

        try:
            sides = int(math.floor(float(sides)))
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Wire.Arc - Error: The input sides or tolerance parameter is invalid. Returning None.")
            return None
        if sides < 1 or not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Wire.Arc - Error: The input sides must be at least 1 and tolerance must be greater than zero. Returning None.")
            return None

        # Preserve the historical polyline construction exactly when requested.
        if bool(polyline):
            import numpy as np

            def circle_arc_points(p1, p2, p3, n):
                p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm <= tolerance:
                    return None
                normal = normal / norm
                midpoint1 = (p1 + p2) / 2
                midpoint2 = (p1 + p3) / 2

                def perpendicular_bisector(pA, pB, midpoint):
                    direction = np.cross(normal, pB - pA)
                    magnitude = np.linalg.norm(direction)
                    if magnitude <= tolerance:
                        return None, None
                    return direction / magnitude, midpoint

                direction1, midpoint1 = perpendicular_bisector(p1, p2, midpoint1)
                direction2, midpoint2 = perpendicular_bisector(p1, p3, midpoint2)
                if direction1 is None or direction2 is None:
                    return None
                A = np.array([direction1, -direction2]).T
                b = midpoint2 - midpoint1
                t1 = np.linalg.lstsq(A, b, rcond=None)[0][0]
                circumcenter = midpoint1 + t1 * direction1

                def rotation_matrix_around_axis(axis, theta):
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)
                    x, y, z = axis
                    return np.array([
                        [cos_theta + x*x*(1-cos_theta), x*y*(1-cos_theta)-z*sin_theta, x*z*(1-cos_theta)+y*sin_theta],
                        [y*x*(1-cos_theta)+z*sin_theta, cos_theta+y*y*(1-cos_theta), y*z*(1-cos_theta)-x*sin_theta],
                        [z*x*(1-cos_theta)-y*sin_theta, z*y*(1-cos_theta)+x*sin_theta, cos_theta+z*z*(1-cos_theta)],
                    ])

                def interpolate_on_arc(p_start, p_end, center, n_points):
                    v_start = p_start - center
                    v_end = p_end - center
                    denominator = np.linalg.norm(v_start) * np.linalg.norm(v_end)
                    if denominator <= tolerance:
                        return None
                    cosine = np.dot(v_start, v_end) / denominator
                    cosine = max(-1.0, min(1.0, float(cosine)))
                    angle_between = np.arccos(cosine)
                    axis = np.cross(v_start, v_end)
                    magnitude = np.linalg.norm(axis)
                    if magnitude <= tolerance:
                        return None
                    axis = axis / magnitude
                    if n_points % 2 == 0:
                        angles = np.linspace(0, angle_between, n_points + 1)
                        arc_points = [center + np.dot(rotation_matrix_around_axis(axis, angle), v_start) for angle in angles]
                        return [p_start] + arc_points[1:]
                    angles = np.linspace(0, angle_between, n_points)
                    return [center + np.dot(rotation_matrix_around_axis(axis, angle), v_start) for angle in angles]

                if n <= 1:
                    return [p1, p3]
                if n == 2:
                    return [p1, p2, p3]
                arc1 = interpolate_on_arc(p1, p2, circumcenter, (n + 1) // 2)
                arc2 = interpolate_on_arc(p2, p3, circumcenter, (n + 1) // 2)
                if arc1 is None or arc2 is None:
                    return None
                return np.vstack([arc1, arc2])

            points = circle_arc_points(
                Vertex.Coordinates(startVertex, mantissa=None),
                Vertex.Coordinates(middleVertex, mantissa=None),
                Vertex.Coordinates(endVertex, mantissa=None),
                sides,
            )
            if points is None:
                if not silent:
                    print("Wire.Arc - Error: The three input vertices are collinear or otherwise degenerate. Returning None.")
                return None
            vertices = [Vertex.ByCoordinates(list(point)) for point in points]
            result = Wire.ByVertices(vertices, close=close, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(result, "Wire") and not silent:
                print("Wire.Arc - Error: Could not create the polyline arc. Returning None.")
            return result if Topology.IsInstance(result, "Wire") else None

        def xyz(vertex):
            return [float(value) for value in Vertex.Coordinates(vertex, mantissa=None)]

        def sub(a, b):
            return [a[i] - b[i] for i in range(3)]

        def add(a, b):
            return [a[i] + b[i] for i in range(3)]

        def mul(a, scalar):
            return [a[i] * scalar for i in range(3)]

        def dot(a, b):
            return sum(a[i] * b[i] for i in range(3))

        def cross(a, b):
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]

        def magnitude(a):
            return math.sqrt(dot(a, a))

        A = xyz(startVertex)
        B = xyz(middleVertex)
        C = xyz(endVertex)
        u = sub(B, A)
        v = sub(C, A)
        w = cross(u, v)
        w2 = dot(w, w)
        if w2 <= tolerance * tolerance:
            if not silent:
                print("Wire.Arc - Error: The three input vertices are collinear. Returning None.")
            return None

        u2 = dot(u, u)
        v2 = dot(v, v)
        offset = mul(add(mul(cross(v, w), u2), mul(cross(w, u), v2)), 1.0 / (2.0 * w2))
        center_xyz = add(A, offset)
        center = Vertex.ByCoordinates(*center_xyz)
        if not Topology.IsInstance(center, "Vertex"):
            return None

        radius_vector = sub(A, center_xyz)
        radius = magnitude(radius_vector)
        if radius <= tolerance:
            return None
        normal_mag = magnitude(w)
        normal = [value / normal_mag for value in w]
        axis_x = [value / radius for value in radius_vector]
        axis_y = cross(normal, axis_x)
        axis_y_mag = magnitude(axis_y)
        if axis_y_mag <= tolerance:
            return None
        axis_y = [value / axis_y_mag for value in axis_y]

        def angle_of(point_xyz):
            radial = sub(point_xyz, center_xyz)
            angle = math.atan2(dot(radial, axis_y), dot(radial, axis_x))
            if angle < 0.0:
                angle += 2.0 * math.pi
            return angle

        middle_angle = angle_of(B)
        end_angle = angle_of(C)
        if end_angle <= 1.0e-12:
            end_angle += 2.0 * math.pi
        if middle_angle <= 1.0e-12 or middle_angle >= end_angle - 1.0e-12:
            if not silent:
                print("Wire.Arc - Error: Could not determine a unique circular sweep through the middle vertex. Returning None.")
            return None

        axisA = [radius * value for value in axis_x]
        axisB = [radius * value for value in axis_y]
        sweep_degrees = math.degrees(end_angle)
        edges = []
        for i in range(sides):
            a0 = sweep_degrees * float(i) / float(sides)
            a1 = sweep_degrees * float(i + 1) / float(sides)
            edge = Wire._ConicEdge(center, axisA, axisB, a0, a1, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(edge, "Edge"):
                if not silent:
                    print("Wire.Arc - Error: Could not create an exact circular arc segment. Returning None.")
                return None
            edges.append(edge)

        if close:
            chord = Edge.ByStartVertexEndVertex(
                Edge.EndVertex(edges[-1], silent=True),
                Edge.StartVertex(edges[0], silent=True),
                tolerance=tolerance,
                silent=True,
            )
            if Topology.IsInstance(chord, "Edge"):
                edges.append(chord)

        result = Wire.ByEdges(edges, orient=True, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Wire") and not silent:
            print("Wire.Arc - Error: Could not create the curved arc Wire. Returning None.")
        return result if Topology.IsInstance(result, "Wire") else None

    

    @staticmethod
    def ArcByEdge(edge,
                  sagitta: float = 1,
                  absolute: bool = True,
                  sides: int = 16,
                  close: bool = True,
                  polyline: bool = True,
                  tolerance: float = 0.0001,
                  silent: bool = False):
        """
        Creates an arc using a geometrically linear input edge as its base chord.

        The input edge defines the arc start and end vertices. The sagitta defines
        the middle vertex. Curved input edges are rejected rather than treated as
        endpoint chords. ``polyline`` is forwarded to :meth:`Wire.Arc`.

        Parameters
        ----------
        edge : topologic_core.Edge
            The geometrically linear base-chord edge.
        sagitta : float , optional
            Sagitta length. Default is 1.
        absolute : bool , optional
            If True, ``sagitta`` is an absolute distance. Otherwise it is a ratio
            of the base-chord length. Default is True.
        sides : int , optional
            Number of curved arc Edges when ``polyline`` is False, or straight
            segments when ``polyline`` is True. Default is 16.
        close : bool , optional
            If True, add the straight base chord to close the arc. Default is True.
        polyline : bool , optional
            If True, create the historical straight-edge approximation. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created arc Wire.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Wire.ArcByEdge - Error: The input edge parameter is not a valid edge. Returning None.")
            return None
        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.ArcByEdge - Error: The input edge must be geometrically linear because it defines the base chord. Returning None.")
            return None
        try:
            sagitta = float(sagitta)
        except Exception:
            return None
        if sagitta <= 0.0:
            if not silent:
                print("Wire.ArcByEdge - Error: The input sagitta parameter must be greater than zero. Returning None.")
            return None

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        length = sagitta if absolute else Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True) * sagitta
        normal_edge = Edge.NormalEdge(edge, length=length, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(normal_edge, "Edge"):
            if not silent:
                print("Wire.ArcByEdge - Warning: Could not construct the sagitta edge. Returning None.")
            return None
        middle = Edge.EndVertex(normal_edge, silent=True)
        return Wire.Arc(
            start,
            middle,
            end,
            sides=sides,
            close=close,
            polyline=polyline,
            tolerance=tolerance,
            silent=silent,
        )




    @staticmethod
    def Bisectors(wire, offset: float = 1.0, offsetKey: str = "offset", stepOffsetA: float = 0, stepOffsetB: float = 0, stepOffsetKeyA: str = "stepOffsetA", stepOffsetKeyB: str = "stepOffsetB", reverse: bool = False, transferDictionaries: bool = False, epsilon: float = 0.01, tolerance: float = 0.0001,  silent: bool = False, numWorkers: int = None):
        """
        Returns the bisectors created by a polyline offset. Curved input wires are rejected because this algorithm is line-based.

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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Bisectors - Error: This implementation supports polyline wires only. Returning None.")
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
        Creates a wire from the input list of edges while preserving their actual geometry.

        Curved edges are passed to the active topology backend unchanged. If orientation
        is requested, edges are reoriented using :meth:`Edge.Reverse`; endpoint chords
        are never substituted for curved geometry.

        Parameters
        ----------
        edges : list
            The input list of edges.
        orient : bool , optional
            If set to True, a manifold wire is oriented head-to-tail. Default is False.
        transferDictionaries : bool , optional
            If set to True, dictionaries from source edges are transferred/merged onto
            corresponding result edges. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created wire, or None if the edges cannot form one wire.
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
        edge_list = [edge for edge in edges if Topology.IsInstance(edge, "Edge")]
        if len(edge_list) == 0:
            if not silent:
                print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
            return None

        def construct(source_edges):
            result = None
            if Wire._UseNativeWireBackend():
                try:
                    result = Core.Wire.ByEdges(source_edges, tolerance)
                except Exception:
                    result = None
            if Topology.IsInstance(result, "Wire"):
                return result
            if len(source_edges) == 1:
                try:
                    result = Core.Wire.ByEdges(source_edges)
                except Exception:
                    result = None
                if Topology.IsInstance(result, "Wire"):
                    return result
            try:
                result = Topology.SelfMerge(
                    Cluster.ByTopologies(source_edges),
                    tolerance=tolerance,
                )
            except Exception:
                result = None
            return result if Topology.IsInstance(result, "Wire") else None

        wire = construct(edge_list)
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.ByEdges - Error: The operation failed. Returning None.")
            return None

        # Transfer dictionaries only by actual edge equivalence, never by endpoint
        # coincidence alone (different arcs may share the same two endpoints).
        result_edges = Wire.Edges(wire, silent=True) or []
        updated_edges = []
        changed = False
        for result_edge in result_edges:
            source_index = None
            for i, source_edge in enumerate(edge_list):
                try:
                    if Topology.IsSame(result_edge, source_edge):
                        source_index = i
                        break
                except Exception:
                    pass
            if source_index is None:
                source_index = Edge.Index(
                    result_edge,
                    edge_list,
                    strict=False,
                    tolerance=tolerance,
                    silent=True,
                )
            updated = result_edge
            if source_index is not None:
                dictionary = Topology.Dictionary(edge_list[source_index], silent=True)
                if dictionary:
                    candidate = Topology.SetDictionary(updated, dictionary, silent=True)
                    if Topology.IsInstance(candidate, "Edge"):
                        updated = candidate
                        changed = True
            updated_edges.append(updated)

        if changed and len(updated_edges) == len(result_edges):
            rebuilt = construct(updated_edges)
            if Topology.IsInstance(rebuilt, "Wire"):
                wire = rebuilt

        if transferDictionaries:
            wire_edges = Wire.Edges(wire, silent=True) or []
            source_cluster = Cluster.ByTopologies(edge_list)
            if source_cluster is not None:
                for wire_edge in wire_edges:
                    internal_vertex = Topology.InternalVertex(wire_edge, tolerance=tolerance, silent=True)
                    if not Topology.IsInstance(internal_vertex, "Vertex"):
                        continue
                    enclosing_edges = Vertex.EnclosingEdges(
                        internal_vertex,
                        source_cluster,
                        exclusive=False,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if isinstance(enclosing_edges, list) and enclosing_edges:
                        dictionaries = [Topology.Dictionary(edge, silent=True) for edge in enclosing_edges]
                        merged = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                        if merged:
                            Topology.SetDictionary(wire_edge, merged, silent=True)

        if orient and Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
            if not isinstance(ordered, list):
                if not silent:
                    print("Wire.ByEdges - Error: Could not orient the input edges without altering their geometry. Returning None.")
                return None
            oriented_wire = construct(ordered)
            if Topology.IsInstance(oriented_wire, "Wire"):
                wire = oriented_wire
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
        Creates an offset of a polyline wire. Curved input wires are rejected by this line-based implementation. A positive offset is toward the interior of an anti-clockwise wire.

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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.ByOffset - Error: This implementation supports polyline wires only. Returning None.")
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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.ByOffsetArea - Error: This implementation supports polyline wires only. Returning None.")
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
    def Circle(origin=None,
               radius: float = 0.5,
               sides: int = 16,
               spokes: bool = False,
               fromAngle: float = 0.0,
               toAngle: float = 360.0,
               close: bool = True,
               direction: list = [0, 0, 1],
               placement: str = "center",
               polyline: bool = True,
               tolerance: float = 0.0001,
               silent: bool = False):
        """
        Creates a circular Wire.

        When ``polyline`` is False, ``sides`` specifies the number of exact
        circular-arc Edge subtopologies. Geometric accuracy is independent of
        this segmentation count. When ``polyline`` is True, ``sides`` retains
        its historical meaning as the number of straight segments.

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
            Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created circular Wire.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.Circle - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None
        try:
            radius = abs(float(radius))
            sides = int(math.floor(float(sides)))
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Wire.Circle - Error: One or more numerical input parameters are invalid. Returning None.")
            return None
        if radius <= tolerance or sides < 1 or tolerance <= 0.0:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            if not silent:
                print("Wire.Circle - Error: The input placement parameter is not recognized. Returning None.")
            return None
        while toAngle < fromAngle:
            toAngle += 360.0
        angle_range = toAngle - fromAngle
        if angle_range <= tolerance or angle_range > 360.0 + tolerance:
            if not silent:
                print("Wire.Circle - Error: The angular range must be greater than zero and no greater than 360 degrees. Returning None.")
            return None
        full_circle = abs(angle_range - 360.0) <= tolerance

        if bool(polyline):
            vertices = []
            for i in range(sides + 1):
                angle = math.radians(fromAngle + angle_range * float(i) / float(sides))
                vertices.append(Vertex.ByCoordinates(
                    math.sin(angle) * radius + Vertex.X(origin),
                    math.cos(angle) * radius + Vertex.Y(origin),
                    Vertex.Z(origin),
                ))
            base_wire = Wire.ByVertices(vertices[::-1], close=False if full_circle else close, tolerance=tolerance, silent=True)
            perimeter_edges = Wire.Edges(base_wire, silent=True) or [] if Topology.IsInstance(base_wire, "Wire") else []
        else:
            # A complete one-segment circle is represented by one genuinely closed
            # circular Edge. For higher segmentation counts (and partial circles),
            # exact rational conic arc Edges are used.
            if full_circle and sides == 1:
                circle_edge = Edge.Circle(
                    origin=origin,
                    radius=radius,
                    direction=[0, 0, 1],
                    placement="center",
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(circle_edge, "Edge"):
                    return None
                edges = [circle_edge]
            else:
                # Preserve the historical Wire.Circle angular convention: theta=0 is +Y,
                # while traversal is counter-clockwise. In standard conic coordinates
                # this corresponds to phi = 90 - theta and reversing the theta interval.
                phi_start = 90.0 - toAngle
                edges = []
                for i in range(sides):
                    a0 = phi_start + angle_range * float(i) / float(sides)
                    a1 = phi_start + angle_range * float(i + 1) / float(sides)
                    edge = Wire._ConicEdge(
                        origin,
                        [radius, 0.0, 0.0],
                        [0.0, radius, 0.0],
                        a0,
                        a1,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if not Topology.IsInstance(edge, "Edge"):
                        if not silent:
                            print("Wire.Circle - Error: Could not create an exact circular arc segment. Returning None.")
                        return None
                    edges.append(edge)
            perimeter_edges = list(edges)
            if not full_circle and close:
                chord = Edge.ByStartVertexEndVertex(
                    Edge.EndVertex(edges[-1], silent=True),
                    Edge.StartVertex(edges[0], silent=True),
                    tolerance=tolerance,
                    silent=True,
                )
                if Topology.IsInstance(chord, "Edge"):
                    edges.append(chord)
            base_wire = Wire.ByEdges(edges, orient=True, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(base_wire, "Wire"):
            if not silent:
                print("Wire.Circle - Error: Could not create the circle. Returning None.")
            return None

        if spokes and (full_circle or not close):
            junctions = []
            if perimeter_edges:
                junctions = [Edge.StartVertex(edge, silent=True) for edge in perimeter_edges]
                if not full_circle:
                    junctions.append(Edge.EndVertex(perimeter_edges[-1], silent=True))
            spoke_edges = []
            for vertex in junctions:
                spoke = Edge.ByStartVertexEndVertex(origin, vertex, tolerance=tolerance, silent=True)
                if Topology.IsInstance(spoke, "Edge"):
                    spoke_edges.append(spoke)
            if spoke_edges:
                combined = Wire.ByEdges((Wire.Edges(base_wire, silent=True) or []) + spoke_edges, tolerance=tolerance, silent=True)
                if Topology.IsInstance(combined, "Wire"):
                    base_wire = combined

        if placement == "lowerleft":
            base_wire = Topology.Translate(base_wire, radius, radius, 0)
        elif placement == "upperleft":
            base_wire = Topology.Translate(base_wire, radius, -radius, 0)
        elif placement == "lowerright":
            base_wire = Topology.Translate(base_wire, -radius, radius, 0)
        elif placement == "upperright":
            base_wire = Topology.Translate(base_wire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            base_wire = Topology.Orient(base_wire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return base_wire

    
    @staticmethod
    def Close(wire, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Closes an open wire by adding straight connector edges between its open ends.

        Existing constituent edges are retained unchanged, including arcs and splines.
        Only newly required closing connections are constructed as straight edges. For
        a simple open wire this adds one edge between its two endpoints. For a connected
        branching wire with several degree-one ends, nearest unpaired ends are joined.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            Retained for API compatibility. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The closed wire, or None if the open ends cannot be paired safely.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Close - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            return wire

        vertices = Wire.Vertices(wire, silent=True) or []
        ends = [vertex for vertex in vertices if Vertex.Degree(vertex, wire) == 1]
        if len(ends) < 2 or len(ends) % 2 != 0:
            if not silent:
                print("Wire.Close - Error: The input wire does not contain an even number of open end vertices. Returning None.")
            return None

        connectors = []
        remaining = list(ends)
        while remaining:
            a = remaining.pop(0)
            if not remaining:
                return None
            distances = [
                Vertex.Distance(a, b, mantissa=None, tolerance=tolerance, silent=True)
                for b in remaining
            ]
            valid = [(d, i) for i, d in enumerate(distances) if d is not None]
            if not valid:
                return None
            _, nearest_index = min(valid, key=lambda item: item[0])
            b = remaining.pop(nearest_index)
            if not Vertex.IsCoincident(a, b, tolerance=tolerance, silent=True):
                connector = Edge.ByStartVertexEndVertex(a, b, tolerance=tolerance, silent=True)
                if not Topology.IsInstance(connector, "Edge"):
                    if not silent:
                        print("Wire.Close - Error: Could not construct a closing edge. Returning None.")
                    return None
                connectors.append(connector)

        result = Wire.ByEdges(
            (Wire.Edges(wire, silent=True) or []) + connectors,
            orient=False,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            if not silent:
                print("Wire.Close - Error: Could not construct the closed wire. Returning None.")
            return None
        dictionary = Topology.Dictionary(wire, silent=True)
        if dictionary:
            result = Topology.SetDictionary(result, dictionary, silent=True)
        return result



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
    def _CornerVerticesByAngle(wire, cornerType: str = "convex", angTolerance: float = 0.01, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns convex or concave junction vertices of a planar closed manifold wire.

        Curved edges are supported. Junction classification uses the local endpoint
        tangents returned by :meth:`Wire.InteriorAngles`, not endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        cornerType : str , optional
            Either "convex" or "concave". Default is "convex".
        angTolerance : float , optional
            Angular tolerance in degrees around 180 degrees. Default is 0.01.
        mantissa : int , optional
            Number of decimal places used for angle results. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The matching junction vertices in traversal order.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        corner_type = str(cornerType).lower()
        if corner_type not in ("convex", "concave"):
            if not silent:
                print("Wire._CornerVerticesByAngle - Error: cornerType must be 'convex' or 'concave'. Returning None.")
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        angles = Wire.InteriorAngles(wire, tolerance=tolerance, mantissa=mantissa, silent=True)
        if not isinstance(edges, list) or not isinstance(angles, list) or len(edges) != len(angles):
            return None
        vertices = [Edge.StartVertex(edge, silent=True) for edge in edges]
        result = []
        for vertex, angle in zip(vertices, angles):
            if corner_type == "convex" and float(angle) < 180.0 - float(angTolerance):
                result.append(vertex)
            elif corner_type == "concave" and float(angle) > 180.0 + float(angTolerance):
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
    def Cycles(wire, maxVertices: int = 4, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns simple closed circuits found within the input wire.

        Cycle detection operates on endpoint connectivity, but each returned circuit is
        built from the original edge geometry. Curved edges are reused or orientation-
        reversed using :meth:`Edge.Reverse`; they are never replaced by endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        maxVertices : int , optional
            Maximum number of distinct junction vertices in a returned cycle. Default is 4.
        transferDictionaries : bool , optional
            If set to True, source edge dictionaries are explicitly retained on reversed
            cycle edges. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of closed cycle wires.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Cycles - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            maxVertices = max(1, int(maxVertices))
        except Exception:
            if not silent:
                print("Wire.Cycles - Error: The input maxVertices parameter is invalid. Returning None.")
            return None

        edges = Wire.Edges(wire, silent=True) or []
        if not edges:
            return []

        representatives = []
        endpoints = []
        adjacency = {}

        def node_index(vertex):
            for i, representative in enumerate(representatives):
                if Vertex.IsCoincident(vertex, representative, tolerance=tolerance, silent=True):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1

        for index, edge in enumerate(edges):
            a = node_index(Edge.StartVertex(edge, silent=True))
            b = node_index(Edge.EndVertex(edge, silent=True))
            endpoints.append((a, b))
            adjacency.setdefault(a, []).append(index)
            adjacency.setdefault(b, []).append(index)

        found = {}

        def record(path):
            key = tuple(sorted(item[0] for item in path))
            if key not in found:
                found[key] = list(path)

        def walk(start_node, current_node, path_nodes, path_edges, used_edges):
            if len(path_nodes) > maxVertices:
                return
            for edge_index in adjacency.get(current_node, []):
                if edge_index in used_edges:
                    continue
                a, b = endpoints[edge_index]
                next_node = b if a == current_node else a
                step = (edge_index, current_node, next_node)
                if next_node == start_node:
                    if len(path_edges) >= 1 or a == b:
                        record(path_edges + [step])
                    continue
                if next_node in path_nodes or len(path_nodes) >= maxVertices:
                    continue
                walk(
                    start_node,
                    next_node,
                    path_nodes + [next_node],
                    path_edges + [step],
                    used_edges | {edge_index},
                )

        for start_node in range(len(representatives)):
            walk(start_node, start_node, [start_node], [], set())

        result = []
        for path in found.values():
            oriented_edges = []
            valid = True
            for edge_index, from_node, to_node in path:
                source = edges[edge_index]
                a, b = endpoints[edge_index]
                if a == from_node and b == to_node:
                    oriented = source
                elif b == from_node and a == to_node:
                    oriented = Edge.Reverse(source, tolerance=tolerance, silent=True)
                else:
                    valid = False
                    break
                if not Topology.IsInstance(oriented, "Edge"):
                    valid = False
                    break
                if transferDictionaries:
                    dictionary = Topology.Dictionary(source, silent=True)
                    if dictionary:
                        oriented = Topology.SetDictionary(oriented, dictionary, silent=True)
                oriented_edges.append(oriented)
            if not valid:
                continue
            cycle = Wire.ByEdges(oriented_edges, orient=False, tolerance=tolerance, silent=True)
            if Topology.IsInstance(cycle, "Wire") and Wire.IsClosed(cycle, tolerance=tolerance, silent=True):
                result.append(cycle)
        return result


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
    def Ellipse(origin=None,
                inputMode: int = 1,
                width: float = 2.0,
                length: float = 1.0,
                focalLength: float = 0.866025,
                eccentricity: float = 0.866025,
                majorAxisLength: float = 1.0,
                minorAxisLength: float = 0.5,
                sides: int = 32,
                fromAngle: float = 0.0,
                toAngle: float = 360.0,
                close: bool = True,
                direction: list = [0, 0, 1],
                placement: str = "center",
                polyline: bool = True,
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Creates an elliptical Wire.

        When ``polyline`` is False, the ellipse is composed of exact rational
        quadratic NURBS conic Edges. When ``polyline`` is True, the historical
        straight-edge approximation is returned.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            Placement origin. Default is the global origin.
        inputMode : int , optional
            Ellipse definition mode: 1 width/length, 2 focalLength/eccentricity,
            3 focalLength/minorAxisLength, or 4 majorAxisLength/minorAxisLength.
            Default is 1.
        width : float , optional
            Width used by input mode 1. Default is 2.0.
        length : float , optional
            Length used by input mode 1. Default is 1.0.
        focalLength : float , optional
            Focal length used by modes 2 and 3. Default is 0.866025.
        eccentricity : float , optional
            Eccentricity used by mode 2. Default is 0.866025.
        majorAxisLength : float , optional
            Semi-major-axis value used by mode 4, preserving historical semantics.
            Default is 1.0.
        minorAxisLength : float , optional
            Semi-minor-axis value used by modes 3 and 4. Default is 0.5.
        sides : int , optional
            Number of exact conic Edges, or straight segments in polyline mode.
            Default is 32.
        fromAngle : float , optional
            Beginning of the requested angular range in degrees. Default is 0.
        toAngle : float , optional
            End of the requested angular range in degrees. Default is 360.
        close : bool , optional
            For a partial ellipse, if True add a straight closing chord. Default is True.
        direction : list , optional
            Ellipse-plane normal. Default is [0, 0, 1].
        placement : str , optional
            "center" or "lowerleft". Default is "center".
        polyline : bool , optional
            If True, create the historical straight-edge approximation. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created ellipse.
        """
        ellipse_all = Wire.EllipseAll(
            origin=origin,
            inputMode=inputMode,
            width=width,
            length=length,
            focalLength=focalLength,
            eccentricity=eccentricity,
            majorAxisLength=majorAxisLength,
            minorAxisLength=minorAxisLength,
            sides=sides,
            fromAngle=fromAngle,
            toAngle=toAngle,
            close=close,
            direction=direction,
            placement=placement,
            polyline=polyline,
            tolerance=tolerance,
            silent=silent,
        )
        if ellipse_all is None:
            if not silent:
                print("Wire.Ellipse - Error: Could not create an ellipse. Returning None.")
            return None
        return ellipse_all["ellipse"]

    @staticmethod
    def EllipseAll(
        origin=None,
        inputMode: int = 1,
        width: float = 2.0,
        length: float = 1.0,
        focalLength: float = 0.866025,
        eccentricity: float = 0.866025,
        majorAxisLength: float = 1.0,
        minorAxisLength: float = 0.5,
        sides: int = 32,
        fromAngle: float = 0.0,
        toAngle: float = 360.0,
        close: bool = True,
        direction: list = [0, 0, 1],
        placement: str = "center",
        polyline: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates an ellipse and returns its geometry and derived parameters.

        When polyline is True, the ellipse is constructed using straight edges,
        preserving the historical TopologicPy behaviour. When polyline is False,
        the ellipse is constructed from exact rational quadratic conic edges.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. Default is None, which
            places the ellipse at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. Default is 1.
            The options are:
            1. Width and Length.
            2. Focal Length and Eccentricity.
            3. Focal Length and Minor Axis Length.
            4. Major Axis Length and Minor Axis Length.
        width : float , optional
            The width of the ellipse. Used when inputMode is 1. Default is 2.0.
        length : float , optional
            The length of the ellipse. Used when inputMode is 1. Default is 1.0.
        focalLength : float , optional
            The focal length. Used when inputMode is 2 or 3. Default is 0.866025.
        eccentricity : float , optional
            The eccentricity. Used when inputMode is 2. Default is 0.866025.
        majorAxisLength : float , optional
            The semi-major axis length. Used when inputMode is 4. Default is 1.0.
        minorAxisLength : float , optional
            The semi-minor axis length. Used when inputMode is 3 or 4.
            Default is 0.5.
        sides : int , optional
            If polyline is True, the number of straight edges. If polyline is
            False, the number of exact conic edge segments. Default is 32.
        fromAngle : float , optional
            The starting angle in degrees. Default is 0.
        toAngle : float , optional
            The ending angle in degrees. Default is 360.
        close : bool , optional
            If True, a partial ellipse is closed with a straight chord.
            Default is True.
        direction : list , optional
            The normal direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The placement of the origin. Valid options are "center" and
            "lowerleft". Default is "center".
        polyline : bool , optional
            If True, creates the historical straight-edge approximation.
            If False, creates exact conic edges. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        dict
            A dictionary containing:
            - "ellipse": The created ellipse.
            - "foci": The two focal vertices.
            - "a": The semi-major axis length.
            - "b": The semi-minor axis length.
            - "c": The focal length.
            - "e": The eccentricity.
            - "w": The width.
            - "l": The length.

        """
        import math

        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Wire.EllipseAll - Error: Could not create a valid origin vertex. Returning None.")
            return None

        try:
            inputMode = int(inputMode)
        except Exception:
            if not silent:
                print("Wire.EllipseAll - Error: The input inputMode parameter is not a valid integer. Returning None.")
            return None

        if inputMode not in [1, 2, 3, 4]:
            if not silent:
                print("Wire.EllipseAll - Error: The input inputMode parameter must be 1, 2, 3, or 4. Returning None.")
            return None

        if not isinstance(placement, str):
            if not silent:
                print("Wire.EllipseAll - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        if placement not in ["center", "lowerleft"]:
            if not silent:
                print("Wire.EllipseAll - Error: The input placement parameter is not recognized. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Wire.EllipseAll - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            ]

            width = abs(float(width))
            length = abs(float(length))
            focalLength = abs(float(focalLength))
            eccentricity = abs(float(eccentricity))
            majorAxisLength = abs(float(majorAxisLength))
            minorAxisLength = abs(float(minorAxisLength))

            sides = int(math.floor(abs(float(sides))))

            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            tolerance = float(tolerance)

        except Exception:
            if not silent:
                print("Wire.EllipseAll - Error: One or more input parameters are invalid. Returning None.")
            return None

        if tolerance <= 0:
            if not silent:
                print("Wire.EllipseAll - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            if not silent:
                print("Wire.EllipseAll - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        minimum_sides = 3 if polyline else 1

        if sides < minimum_sides:
            if not silent:
                print(
                    "Wire.EllipseAll - Error: The input sides parameter is too small. "
                    "Returning None."
                )
            return None

        # ------------------------------------------------------------------
        # Derive ellipse parameters.
        # ------------------------------------------------------------------

        if inputMode == 1:
            if width <= tolerance or length <= tolerance:
                return None

            w = width
            l = length
            a = width * 0.5
            b = length * 0.5

            c = math.sqrt(abs(a * a - b * b))
            e = c / max(a, b)

        elif inputMode == 2:
            if focalLength <= tolerance or eccentricity <= tolerance:
                return None

            if eccentricity >= 1.0:
                return None

            c = focalLength
            e = eccentricity

            a = c / e
            b_squared = a * a - c * c

            if b_squared <= tolerance * tolerance:
                return None

            b = math.sqrt(b_squared)

            w = 2.0 * a
            l = 2.0 * b

        elif inputMode == 3:
            if focalLength <= tolerance or minorAxisLength <= tolerance:
                return None

            c = focalLength
            b = minorAxisLength
            a = math.sqrt(b * b + c * c)

            e = c / a

            w = 2.0 * a
            l = 2.0 * b

        else:
            if majorAxisLength <= tolerance or minorAxisLength <= tolerance:
                return None

            a = majorAxisLength
            b = minorAxisLength

            c = math.sqrt(abs(a * a - b * b))
            e = c / max(a, b)

            w = 2.0 * a
            l = 2.0 * b

        # ------------------------------------------------------------------
        # Angular range.
        # ------------------------------------------------------------------

        while toAngle < fromAngle:
            toAngle += 360.0

        angle_range = toAngle - fromAngle

        if angle_range <= tolerance:
            return None

        if angle_range > 360.0 + tolerance:
            return None

        full_ellipse = abs(angle_range - 360.0) <= tolerance

        # ------------------------------------------------------------------
        # Historical polyline construction.
        # ------------------------------------------------------------------

        if polyline:
            vertices = []

            for i in range(sides + 1):
                angle = math.radians(
                    fromAngle
                    + angle_range * float(i) / float(sides)
                )

                vertices.append(
                    Vertex.ByCoordinates(
                        math.sin(angle) * a + Vertex.X(origin),
                        math.cos(angle) * b + Vertex.Y(origin),
                        Vertex.Z(origin),
                    )
                )

            base_wire = Wire.ByVertices(
                vertices[::-1],
                close=False if full_ellipse else close,
                tolerance=tolerance,
                silent=True,
            )

        # ------------------------------------------------------------------
        # Exact rational conic construction.
        # ------------------------------------------------------------------

        else:
            # Historical ellipse convention starts at +Y and proceeds
            # counter-clockwise. _ConicEdge uses conventional +X angular
            # coordinates, so convert the parameterization.
            phi_start = 90.0 - toAngle

            edges = []

            for i in range(sides):
                angle_a = (
                    phi_start
                    + angle_range * float(i) / float(sides)
                )

                angle_b = (
                    phi_start
                    + angle_range * float(i + 1) / float(sides)
                )

                edge = Wire._ConicEdge(
                    origin,
                    [a, 0.0, 0.0],
                    [0.0, b, 0.0],
                    angle_a,
                    angle_b,
                    tolerance=tolerance,
                    silent=True,
                )

                if not Topology.IsInstance(edge, "Edge"):
                    if not silent:
                        print("Wire.EllipseAll - Error: Could not create an exact conic edge. Returning None.")
                    return None

                edges.append(edge)

            if not full_ellipse and close:
                chord = Edge.ByStartVertexEndVertex(
                    Edge.EndVertex(
                        edges[-1],
                        silent=True,
                    ),
                    Edge.StartVertex(
                        edges[0],
                        silent=True,
                    ),
                    tolerance=tolerance,
                    silent=True,
                )

                if Topology.IsInstance(chord, "Edge"):
                    edges.append(chord)

            base_wire = Wire.ByEdges(
                edges,
                orient=True,
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(base_wire, "Wire"):
            if not silent:
                print("Wire.EllipseAll - Error: Could not create the ellipse. Returning None.")
            return None

        # ------------------------------------------------------------------
        # Placement.
        # ------------------------------------------------------------------

        if placement == "lowerleft":
            base_wire = Topology.Translate(
                base_wire,
                a,
                b,
                0,
            )

        if direction != [0, 0, 1]:
            base_wire = Topology.Orient(
                base_wire,
                origin=origin,
                dirA=[0, 0, 1],
                dirB=direction,
            )

        # ------------------------------------------------------------------
        # Foci.
        # ------------------------------------------------------------------

        # Preserve the historical TopologicPy convention of placing the foci
        # on the local X axis.
        focus1 = Vertex.ByCoordinates(
            c + Vertex.X(origin),
            Vertex.Y(origin),
            Vertex.Z(origin),
        )

        focus2 = Vertex.ByCoordinates(
            -c + Vertex.X(origin),
            Vertex.Y(origin),
            Vertex.Z(origin),
        )

        foci = Cluster.ByTopologies(
            [focus1, focus2]
        )

        if placement == "lowerleft":
            foci = Topology.Translate(
                foci,
                a,
                b,
                0,
            )

        if direction != [0, 0, 1]:
            foci = Topology.Orient(
                foci,
                origin=origin,
                dirA=[0, 0, 1],
                dirB=direction,
            )

        return {
            "ellipse": base_wire,
            "foci": foci,
            "a": a,
            "b": b,
            "c": c,
            "e": e,
            "w": w,
            "l": l,
        }

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
        Fillets the corners of a polyline wire. Curved input wires are rejected by this vertex-based implementation.

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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Fillet - Error: This implementation supports polyline wires only. Returning None.")
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
                     polyline: bool = True,
                     tolerance: float = 0.0001,
                     silent: bool = False):
        """
        Creates a golden-rectangle spiral from quarter-circle arcs.

        In curved mode each golden-rectangle subdivision contributes one exact
        circular quarter-arc Edge. In polyline mode the historical faceted
        approximation is retained and ``sides`` controls its total straight-edge
        segmentation.

        Parameters
        ----------
        width : float , optional
            Long side of the outer golden rectangle. Default is 1.0.
        maxIterations : int , optional
            Number of recursive golden-square subdivisions and, in curved mode,
            number of exact arc Edges. Default is 10.
        clockwise : bool , optional
            If True, mirror the canonical spiral to obtain clockwise progression.
            Default is False.
        sides : int , optional
            Total straight-edge count in polyline mode. Ignored in curved mode.
            Default is 96.
        origin : topologic_core.Vertex , optional
            Placement origin. Default is the global origin.
        placement : str , optional
            One of "center", "lowerleft", "lowerright", "upperleft", or
            "upperright". Default is "center".
        direction : list , optional
            Spiral-plane normal. Default is [0, 0, 1].
        mantissa : int , optional
            Decimal precision used by the legacy golden-square construction.
            Default is 6.
        polyline : bool , optional
            If True, create the historical straight-edge approximation.
            Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created golden spiral.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster

        try:
            width = float(width)
            maxIterations = int(maxIterations)
            sides = int(sides)
            mantissa = int(mantissa)
            tolerance = float(tolerance)
        except Exception:
            return None
        if width <= 0.0 or maxIterations <= 0 or tolerance <= 0.0:
            return None
        if polyline and sides < maxIterations:
            if not silent:
                print("Wire.GoldenSpiral - Error: In polyline mode, sides must be at least maxIterations. Returning None.")
            return None
        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "lowerright", "upperleft", "upperright"]:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            return None

        def rnd(value):
            return round(float(value), mantissa)

        def square_corners(sx, sy, size):
            return (
                Vertex.ByCoordinates(rnd(sx), rnd(sy), 0.0),
                Vertex.ByCoordinates(rnd(sx + size), rnd(sy), 0.0),
                Vertex.ByCoordinates(rnd(sx + size), rnd(sy + size), 0.0),
                Vertex.ByCoordinates(rnd(sx), rnd(sy + size), 0.0),
            )

        def angle_from(center, point):
            return math.atan2(Vertex.Y(point) - Vertex.Y(center), Vertex.X(point) - Vertex.X(center))

        def normalize_angle(angle):
            while angle <= -math.pi:
                angle += 2.0 * math.pi
            while angle > math.pi:
                angle -= 2.0 * math.pi
            return angle

        def arc_edges(center, p_start, p_end, segment_count=1):
            radius = math.sqrt(
                (Vertex.X(p_start) - Vertex.X(center))**2
                + (Vertex.Y(p_start) - Vertex.Y(center))**2
            )
            if radius <= tolerance:
                return []
            angle0 = angle_from(center, p_start)
            target = angle_from(center, p_end)
            cand_a = angle0 + math.pi / 2.0
            cand_b = angle0 - math.pi / 2.0
            angle1 = cand_a if abs(normalize_angle(cand_a - target)) <= abs(normalize_angle(cand_b - target)) else cand_b

            if not polyline:
                if angle1 > angle0:
                    arc = Edge.Arc(
                        origin=center,
                        radius=radius,
                        fromAngle=math.degrees(angle0),
                        toAngle=math.degrees(angle1),
                        direction=[0, 0, 1],
                        placement="center",
                        tolerance=tolerance,
                        silent=True,
                    )
                else:
                    arc = Edge.Arc(
                        origin=center,
                        radius=radius,
                        fromAngle=math.degrees(angle1),
                        toAngle=math.degrees(angle0),
                        direction=[0, 0, 1],
                        placement="center",
                        tolerance=tolerance,
                        silent=True,
                    )
                    if Topology.IsInstance(arc, "Edge"):
                        arc = Edge.Reverse(arc, tolerance=tolerance, silent=True)
                return [arc] if Topology.IsInstance(arc, "Edge") else []

            segment_count = max(1, int(segment_count))
            if segment_count == 1:
                edge = Edge.ByStartVertexEndVertex(p_start, p_end, tolerance=tolerance, silent=True)
                return [edge] if Topology.IsInstance(edge, "Edge") else []
            cx, cy = Vertex.X(center), Vertex.Y(center)
            points = []
            for i in range(segment_count + 1):
                fraction = float(i) / float(segment_count)
                angle = angle0 + fraction * (angle1 - angle0)
                points.append(Vertex.ByCoordinates(
                    rnd(cx + radius * math.cos(angle)),
                    rnd(cy + radius * math.sin(angle)),
                    0.0,
                ))
            result = []
            for a, b in zip(points[:-1], points[1:]):
                edge = Edge.ByStartVertexEndVertex(a, b, tolerance=tolerance, silent=True)
                if Topology.IsInstance(edge, "Edge"):
                    result.append(edge)
            return result

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        W0 = 1.0
        H0 = 1.0 / phi
        x0 = -W0 * 0.5
        y0 = -H0 * 0.5
        center_vertex = Vertex.ByCoordinates(0.0, 0.0, 0.0)
        side_cycle = ["left", "bottom", "right", "top"]
        rx, ry, rW, rH = x0, y0, W0, H0
        squares = []
        for i in range(maxIterations):
            if rW <= tolerance or rH <= tolerance:
                break
            side = side_cycle[i % 4]
            if rW >= rH:
                size = rH
                if side == "right":
                    sx, sy = rx + (rW - size), ry
                    rW -= size
                else:
                    sx, sy = rx, ry
                    rx += size
                    rW -= size
            else:
                size = rW
                if side == "top":
                    sx, sy = rx, ry + (rH - size)
                    rH -= size
                else:
                    sx, sy = rx, ry
                    ry += size
                    rH -= size
            squares.append((sx, sy, size, side))
        if not squares:
            return None

        if polyline:
            weights = [max(tolerance, size) for _, _, size, _ in squares]
            weight_sum = sum(weights) or 1.0
            segment_counts = [max(1, int(round(sides * weight / weight_sum))) for weight in weights]
            current = sum(segment_counts)
            while current > sides:
                index = max(range(len(segment_counts)), key=lambda i: segment_counts[i])
                if segment_counts[index] <= 1:
                    break
                segment_counts[index] -= 1
                current -= 1
            while current < sides:
                index = max(range(len(segment_counts)), key=lambda i: weights[i])
                segment_counts[index] += 1
                current += 1
        else:
            segment_counts = [1] * len(squares)

        spiral_edges = []
        last_end = None
        epsilon_join = 10.0 ** (-mantissa)
        for (sx, sy, size, side), segment_count in zip(squares, segment_counts):
            bl, br, tr, tl = square_corners(sx, sy, size)
            if side == "left":
                p0, p1, center = tl, br, tr
            elif side == "bottom":
                p0, p1, center = bl, tr, tl
            elif side == "right":
                p0, p1, center = br, tl, bl
            else:
                p0, p1, center = tr, bl, br
            if last_end is not None:
                d0 = abs(Vertex.X(p0) - Vertex.X(last_end)) + abs(Vertex.Y(p0) - Vertex.Y(last_end))
                d1 = abs(Vertex.X(p1) - Vertex.X(last_end)) + abs(Vertex.Y(p1) - Vertex.Y(last_end))
                if d0 > epsilon_join and d1 <= epsilon_join:
                    p0, p1 = p1, p0
            new_edges = arc_edges(center, p0, p1, segment_count)
            if new_edges:
                last_end = Edge.EndVertex(new_edges[-1], silent=True)
                spiral_edges.extend(new_edges)

        spiral = Wire.ByEdges(spiral_edges, orient=True, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(spiral, "Wire"):
            spiral = Topology.SelfMerge(Cluster.ByTopologies(spiral_edges), tolerance=tolerance)
        if not Topology.IsInstance(spiral, "Wire"):
            return None

        if clockwise:
            spiral = Topology.Scale(spiral, center_vertex, 1.0, -1.0, 1.0)
        spiral = Topology.Scale(spiral, center_vertex, width, width, 1.0)
        W = width
        H = width / phi
        references = {
            "center": (0.0, 0.0),
            "lowerleft": (-W * 0.5, -H * 0.5),
            "lowerright": (W * 0.5, -H * 0.5),
            "upperleft": (-W * 0.5, H * 0.5),
            "upperright": (W * 0.5, H * 0.5),
        }
        refx, refy = references[placement]
        spiral = Topology.Translate(spiral, -refx, -refy, 0.0)
        return Topology.Orient(spiral, origin, [0, 0, 1], direction)



    @staticmethod
    def InteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6, silent: bool = False) -> list:
        """
        Returns local interior corner angles of a planar, closed, manifold wire.

        The input wire may contain straight or curved edges. At each junction the angle
        is evaluated from the actual endpoint tangents of the two incident edges. Smooth
        tangent-continuous junctions therefore return 180 degrees. Curve geometry is never
        replaced by endpoint chords.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        mantissa : int , optional
            Number of decimal places to round the returned angles to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Interior angles in traversal order, one angle per edge junction.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        if not Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None.")
            return None

        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(edges, list) or len(edges) < 1:
            return None
        normal = Wire.Normal(wire, outputType="xyz", mantissa=None, tolerance=tolerance, silent=True)
        if not isinstance(normal, list) or len(normal) != 3:
            if not silent:
                print("Wire.InteriorAngles - Error: The input wire is not planar. Returning None.")
            return None

        def dot(a, b): return sum(a[i] * b[i] for i in range(3))
        def cross(a, b):
            return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

        # Determine whether the traversal is clockwise or counter-clockwise relative
        # to the canonical plane normal using a polyline sampling of the real curves.
        sampled_points = []
        for edge in edges:
            for j in range(8):
                vertex = Edge.VertexByParameter(edge, u=j / 8.0, tolerance=tolerance, silent=True)
                if Topology.IsInstance(vertex, "Vertex"):
                    coords = Vertex.Coordinates(vertex, mantissa=None)
                    sampled_points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        end_vertex = Edge.EndVertex(edges[-1], silent=True)
        if Topology.IsInstance(end_vertex, "Vertex"):
            coords = Vertex.Coordinates(end_vertex, mantissa=None)
            sampled_points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        area_vector = [0.0, 0.0, 0.0]
        for a, b in zip(sampled_points[:-1], sampled_points[1:]):
            c = cross(a, b)
            area_vector = [area_vector[i] + c[i] for i in range(3)]
        orientation = 1.0 if dot(area_vector, normal) >= 0.0 else -1.0

        angles = []
        count = len(edges)
        for i in range(count):
            incoming = edges[i - 1]
            outgoing = edges[i]
            tangent_in = Edge.TangentAtParameter(incoming, u=1.0, mantissa=None, tolerance=tolerance, silent=True)
            tangent_out = Edge.TangentAtParameter(outgoing, u=0.0, mantissa=None, tolerance=tolerance, silent=True)
            if not isinstance(tangent_in, (list, tuple)) or not isinstance(tangent_out, (list, tuple)):
                return None
            a = [float(value) for value in tangent_in[:3]]
            b = [float(value) for value in tangent_out[:3]]
            turn = math.degrees(math.atan2(dot(normal, cross(a, b)), max(-1.0, min(1.0, dot(a, b)))))
            angle = 180.0 - orientation * turn
            while angle < 0.0:
                angle += 360.0
            while angle > 360.0:
                angle -= 360.0
            angles.append(angle if mantissa is None else round(angle, mantissa))
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
        Creates *n* intermediate polyline wires by vertex interpolation. Curved input wires are rejected rather than reduced to junction vertices.

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
        if any(not Wire.IsPolyline(wire, tolerance=tolerance, silent=True) for wire in wires if Topology.IsInstance(wire, "Wire")):
            if not silent:
                print("Wire.Interpolate - Error: This implementation supports polyline wires only. Returning None.")
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
        Returns the inverse traversal orientation of the input wire.

        This is a convenience wrapper around :meth:`Wire.Reverse` and therefore
        preserves arbitrary curved constituent edges.

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
            The inverted wire, or None on failure.
        """
        return Wire.Reverse(
            wire,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=silent,
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
    def IsPolyline(wire, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input wire is composed entirely of geometrically linear edges.
        Returns False otherwise.

        A wire is considered a polyline if every edge in the wire is geometrically
        linear within the specified tolerance. The method examines the actual geometry
        of each edge using `Edge.IsLinear`, rather than assuming that an edge is linear
        simply because it has a start and end vertex.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance used to determine if the constituent edges are
            geometrically linear. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if all edges of the input wire are geometrically linear.
            False if one or more edges are curved.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.IsPolyline - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Wire.IsPolyline - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if tolerance <= 0:
            if not silent:
                print("Wire.IsPolyline - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        edges = Topology.Edges(wire)

        if not isinstance(edges, list) or len(edges) < 1:
            if not silent:
                print("Wire.IsPolyline - Error: Could not retrieve any edges from the input wire. Returning None.")
            return None

        for edge in edges:
            if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
                return False

        return True
    
    @staticmethod
    def IsSimilar(wireA, wireB, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False) -> bool:
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
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
        if not Wire.IsPolyline(wireA, tolerance=tolerance, silent=True) or not Wire.IsPolyline(wireB, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.IsSimilar - Error: This similarity definition supports polyline wires only. Returning None.")
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
        Miters the corners of a polyline wire. Curved input wires are rejected by this vertex-based implementation.

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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Miter - Error: This implementation supports polyline wires only. Returning None.")
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
        Returns a deterministic unit normal to the plane containing the complete wire geometry.

        Actual edge curves are sampled and their tangents are checked against the candidate
        plane, so a wire is not considered planar merely because its junction vertices are
        coplanar. The sign is canonicalized by making the dominant component positive.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        outputType : str , optional
            Any subset or permutation of "xyz". Default is "xyz".
        mantissa : int , optional
            Number of decimal places to round the result to. Use None for full precision.
            Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The requested components of the unit normal, or None if the complete wire
            geometry is not planar.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Normal - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not isinstance(outputType, str):
            if not silent:
                print("Wire.Normal - Error: The input outputType parameter is not a valid string. Returning None.")
            return None
        axes = outputType.lower()
        if not axes or any(axis not in "xyz" for axis in axes):
            if not silent:
                print("Wire.Normal - Error: The input outputType parameter contains invalid axes. Returning None.")
            return None
        try:
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tol = 0.0001

        edges = Wire.Edges(wire, silent=True) or []
        if not edges:
            return None

        points = []
        sample_parameters = [i / 16.0 for i in range(17)]
        for edge in edges:
            for u in sample_parameters:
                vertex = Edge.VertexByParameter(edge, u=u, tolerance=tol, silent=True)
                if not Topology.IsInstance(vertex, "Vertex"):
                    continue
                coordinates = Vertex.Coordinates(vertex, mantissa=None)
                if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 3:
                    continue
                point = [float(coordinates[0]), float(coordinates[1]), float(coordinates[2])]
                if not any(math.dist(point, existing) <= tol for existing in points):
                    points.append(point)

        if len(points) < 3:
            if not silent:
                print("Wire.Normal - Error: The wire geometry does not define a unique plane. Returning None.")
            return None

        def sub(a, b): return [a[i] - b[i] for i in range(3)]
        def dot(a, b): return sum(a[i] * b[i] for i in range(3))
        def cross(a, b):
            return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        def magnitude(v): return math.sqrt(dot(v, v))

        origin = None
        normal = None
        for i in range(len(points) - 2):
            for j in range(i + 1, len(points) - 1):
                a = sub(points[j], points[i])
                if magnitude(a) <= tol:
                    continue
                for k in range(j + 1, len(points)):
                    b = sub(points[k], points[i])
                    n = cross(a, b)
                    mag = magnitude(n)
                    if mag > tol:
                        origin = points[i]
                        normal = [value / mag for value in n]
                        break
                if normal is not None:
                    break
            if normal is not None:
                break
        if normal is None:
            if not silent:
                print("Wire.Normal - Error: The wire geometry is collinear and does not define a unique plane. Returning None.")
            return None

        for point in points:
            if abs(dot(sub(point, origin), normal)) > tol:
                if not silent:
                    print("Wire.Normal - Error: The complete wire geometry is not planar. Returning None.")
                return None

        tangent_tol = max(tol, 1.0e-9)
        for edge in edges:
            for u in (0.125, 0.25, 0.5, 0.75, 0.875):
                tangent = Edge.TangentAtParameter(edge, u=u, mantissa=None, tolerance=tol, silent=True)
                if isinstance(tangent, (list, tuple)) and len(tangent) >= 3:
                    if abs(dot([float(tangent[0]), float(tangent[1]), float(tangent[2])], normal)) > tangent_tol:
                        if not silent:
                            print("Wire.Normal - Error: The complete wire geometry is not planar. Returning None.")
                        return None

        dominant = max(range(3), key=lambda index: abs(normal[index]))
        if normal[dominant] < 0.0:
            normal = [-value for value in normal]
        values = normal if mantissa is None else [round(value, mantissa) for value in normal]
        mapping = {"x": values[0], "y": values[1], "z": values[2]}
        return [mapping[axis] for axis in axes]
    

    @staticmethod
    def OrientEdges(wire, vertexA, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the input manifold wire oriented head-to-tail from the requested start vertex.

        Actual constituent edge geometry is retained. Any necessary edge reversal is
        performed through :meth:`Edge.Reverse`; curved edges are never rebuilt from endpoints.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertexA : topologic_core.Vertex
            Desired traversal start vertex.
        transferDictionaries : bool , optional
            If set to True, wire and edge dictionaries are explicitly retained. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The oriented wire, or None if a unique traversal cannot be established.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertexA, "Vertex"):
            if not silent:
                print("Wire.OrientEdges - Error: One or more input parameters are invalid. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            if not silent:
                print("Wire.OrientEdges - Error: The input wire is non-manifold. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, startVertex=vertexA, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list):
            if not silent:
                print("Wire.OrientEdges - Error: Could not orient all edges without altering their geometry. Returning None.")
            return None
        result = Wire.ByEdges(
            ordered,
            orient=False,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            return None
        if transferDictionaries:
            dictionary = Topology.Dictionary(wire, silent=True)
            if dictionary:
                result = Topology.SetDictionary(result, dictionary, silent=True)
        return result


    @staticmethod
    def Planarize(wire, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a planarized projection of the complete input wire geometry.

        A best-fit plane is estimated from points sampled on the actual constituent
        curves. On a backend with native curve projection the complete curves are
        projected. On a backend without native curve projection, curved wires are
        rejected rather than replaced by endpoint chords; polyline wires use the
        exact historical vertex projection pathway.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        origin : topologic_core.Vertex , optional
            Origin of the target plane. If None, the wire centroid is used. Default is None.
        mantissa : int , optional
            Number of decimal places used by the plane fitter. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The planarized wire, or None if curve-preserving planarization is unavailable.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Planarize - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            return None

        samples = []
        for edge in Wire.Edges(wire, silent=True) or []:
            for i in range(9):
                vertex = Edge.VertexByParameter(edge, u=i / 8.0, tolerance=tolerance, silent=True)
                if Topology.IsInstance(vertex, "Vertex"):
                    samples.append(vertex)
        if len(samples) < 3:
            if not silent:
                print("Wire.Planarize - Error: Insufficient geometry to determine a target plane. Returning None.")
            return None
        equation = Vertex.PlaneEquation(samples, mantissa=mantissa, tolerance=tolerance, silent=True)
        if not isinstance(equation, dict):
            if not silent:
                print("Wire.Planarize - Error: Could not determine a best-fit plane. Returning None.")
            return None
        receiver = Face.RectangleByPlaneEquation(origin=origin, equation=equation, tolerance=tolerance)
        if not Topology.IsInstance(receiver, "Face"):
            return None
        return Wire.Project(
            wire,
            receiver,
            direction=None,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=silent,
        )


    @staticmethod
    def Project(wire, face, direction: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Projects the complete input wire onto a face along a direction.

        Native curve projection is preferred and preserves arbitrary constituent curves.
        If the active backend lacks native curve projection, the fallback is available
        only for polyline wires; a curved wire is never projected by replacing each curve
        with its endpoint chord.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        face : topologic_core.Face
            The receiving face.
        direction : list , optional
            Projection direction. If None, the reverse face normal is used. Default is None.
        mantissa : int , optional
            Number of decimal places used for fallback projected coordinates. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The projected wire, or None when a curve-preserving projection is unavailable.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Wire.Project - Error: One or more input topology parameters are invalid. Returning None.")
            return None
        if direction is None or (isinstance(direction, (list, tuple)) and len(direction) == 0):
            normal = Face.Normal(face, outputType="xyz", mantissa=None)
            if not isinstance(normal, (list, tuple)) or len(normal) != 3:
                return None
            direction = [-float(value) for value in normal]
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Wire.Project - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            direction = [float(value) for value in direction]
            magnitude = math.sqrt(sum(value * value for value in direction))
        except Exception:
            magnitude = 0.0
        if magnitude <= max(abs(float(tolerance)), 1.0e-12):
            if not silent:
                print("Wire.Project - Error: The input direction vector has zero magnitude. Returning None.")
            return None
        direction = [value / magnitude for value in direction]

        large_face = Topology.Scale(face, Topology.CenterOfMass(face), 500, 500, 500)
        if not Topology.IsInstance(large_face, "Face"):
            return None

        if Wire._UseNativeWireBackend():
            try:
                projected = Core.WireUtility.Project(wire, large_face, direction, tolerance)
            except Exception:
                projected = None
            if Topology.IsInstance(projected, "Wire"):
                return projected
            if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
                if not silent:
                    print("Wire.Project - Error: Native curve projection failed; refusing to replace curved edges by chords. Returning None.")
                return None
        elif not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Project - Error: The active backend cannot project curved wires without altering their geometry. Returning None.")
            return None

        projected_edges = []
        for edge in Wire._OrderedEdges(wire, tolerance=tolerance, silent=True) or Wire.Edges(wire, silent=True) or []:
            start = Vertex.Project(Edge.StartVertex(edge, silent=True), large_face, direction=direction, mantissa=mantissa, tolerance=tolerance, silent=True)
            end = Vertex.Project(Edge.EndVertex(edge, silent=True), large_face, direction=direction, mantissa=mantissa, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                return None
            projected_edge = Edge.ByStartVertexEndVertex(start, end, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(projected_edge, "Edge"):
                return None
            projected_edges.append(projected_edge)
        return Wire.ByEdges(projected_edges, orient=True, tolerance=tolerance, silent=silent)

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
        Removes redundant junctions between adjacent geometrically linear collinear edges.

        Curved edges are always preserved unchanged. Only adjacent edges whose
        actual geometries are linear and whose directions are collinear within the
        specified angular tolerance may be merged. If all edges of an open wire
        collapse into one geometrically linear edge, that Edge is returned directly.

        Parameters
        ----------
        wire : topologic_core.Wire or topologic_core.Cluster
            The input wire, or a cluster containing wires.
        angTolerance : float , optional
            Angular tolerance in degrees. Default is 0.1.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology
            The simplified Wire, Edge, or aggregate topology.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        try:
            tolerance = float(tolerance)
            angTolerance = abs(float(angTolerance))
        except Exception:
            return None
        if not math.isfinite(tolerance) or not math.isfinite(angTolerance) or tolerance <= 0.0:
            return None

        if Topology.IsInstance(wire, "Cluster"):
            wires = Topology.Wires(wire, silent=True) or []
            processed = [Wire.RemoveCollinearEdges(candidate, angTolerance=angTolerance, tolerance=tolerance, silent=True) for candidate in wires]
            processed = [candidate for candidate in processed if candidate is not None]
            if not processed:
                return None
            if len(processed) == 1:
                return processed[0]
            return Topology.SelfMerge(Cluster.ByTopologies(processed, silent=True), tolerance=tolerance)

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.RemoveCollinearEdges - Error: The input topology is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            pieces = Wire.Split(wire, tolerance=tolerance, silent=True) or []
            processed = []
            for piece in pieces:
                if Topology.IsInstance(piece, "Wire"):
                    result = Wire.RemoveCollinearEdges(piece, angTolerance=angTolerance, tolerance=tolerance, silent=True)
                    if result is not None:
                        processed.append(result)
                elif Topology.IsInstance(piece, "Edge"):
                    processed.append(piece)
            if not processed:
                return wire
            if len(processed) == 1:
                return processed[0]
            return Topology.SelfMerge(Cluster.ByTopologies(processed, silent=True), tolerance=tolerance)

        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(edges, list) or not edges:
            return wire
        if len(edges) == 1:
            return wire
        closed = Wire.IsClosed(wire, tolerance=tolerance, silent=True)

        def can_merge(edgeA, edgeB):
            if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
                return False
            if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
                return False
            angle = Edge.Angle(edgeA, edgeB, mantissa=15, bracket=True, tolerance=tolerance, silent=True)
            return angle is not None and abs(float(angle)) <= angTolerance

        if closed:
            break_index = None
            for i in range(len(edges)):
                if not can_merge(edges[i - 1], edges[i]):
                    break_index = i
                    break
            if break_index is not None:
                edges = edges[break_index:] + edges[:break_index]

        merged_edges = []
        current = edges[0]
        for next_edge in edges[1:]:
            if can_merge(current, next_edge):
                candidate = Edge.ByStartVertexEndVertex(
                    Edge.StartVertex(current, silent=True),
                    Edge.EndVertex(next_edge, silent=True),
                    tolerance=tolerance,
                    silent=True,
                )
                if Topology.IsInstance(candidate, "Edge"):
                    current = candidate
                    continue
            merged_edges.append(current)
            current = next_edge
        merged_edges.append(current)

        if len(merged_edges) == 1:
            result = merged_edges[0]
            dictionary = Topology.Dictionary(wire, silent=True)
            if dictionary:
                result = Topology.SetDictionary(result, dictionary, silent=True)
            return result

        result = Wire.ByEdges(merged_edges, orient=True, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Wire"):
            return wire
        dictionary = Topology.Dictionary(wire, silent=True)
        if dictionary:
            result = Topology.SetDictionary(result, dictionary, silent=True)
        return result



    @staticmethod
    def Representation(wire, normalize: bool = True, rotate: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns an alternating edge-length and local-interior-angle representation of a closed wire.

        Edge lengths are true curve lengths and interior angles are tangent-based, so the
        representation remains well-defined for curvilinear closed wires. It is a compact
        descriptor, not a complete geometric-equivalence test.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input closed manifold wire.
        normalize : bool , optional
            If True, edge lengths are divided by the shortest edge length. Default is True.
        rotate : bool , optional
            If True, the pair sequence is cyclically rotated so the shortest edge is first.
            Default is True.
        mantissa : int , optional
            Number of decimal places to round returned values to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Alternating [length, angle, length, angle, ...] values.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire") or not Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Representation - Error: The input wire must be a valid closed wire. Returning None.")
            return None
        if not Wire.IsManifold(wire, silent=True, tolerance=tolerance):
            return None
        edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        angles = Wire.InteriorAngles(wire, tolerance=tolerance, mantissa=None, silent=True)
        if not isinstance(edges, list) or not isinstance(angles, list) or len(edges) != len(angles):
            return None
        lengths = [Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True) for edge in edges]
        if any(length is None for length in lengths):
            return None
        lengths = [float(length) for length in lengths]
        if normalize:
            minimum = min(lengths)
            if minimum <= tolerance:
                return None
            lengths = [length / minimum for length in lengths]
        pairs = list(zip(lengths, angles))
        if rotate and pairs:
            index = min(range(len(pairs)), key=lambda i: pairs[i][0])
            pairs = pairs[index:] + pairs[:index]
        result = []
        for length, angle in pairs:
            result.extend([round(float(length), mantissa), round(float(angle), mantissa)])
        return result


    @staticmethod
    def Reverse(wire, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the input wire with traversal direction reversed while preserving edge geometry.

        The ordered constituent edges are reversed in order and each actual edge is
        orientation-reversed with :meth:`Edge.Reverse`. Curves are never reconstructed
        from their endpoints.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        transferDictionaries : bool , optional
            If set to True, wire and edge dictionaries are explicitly retained. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The reversed wire, or None if reversal cannot preserve the input geometry.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Reverse - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list):
            if not silent:
                print("Wire.Reverse - Error: The input wire has no unique traversal direction. Returning None.")
            return None
        reversed_edges = []
        for source in reversed(ordered):
            edge = Edge.Reverse(source, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(edge, "Edge"):
                if not silent:
                    print("Wire.Reverse - Error: An input edge could not be reversed without altering its geometry. Returning None.")
                return None
            if transferDictionaries:
                dictionary = Topology.Dictionary(source, silent=True)
                if dictionary:
                    edge = Topology.SetDictionary(edge, dictionary, silent=True)
            reversed_edges.append(edge)
        result = Wire.ByEdges(
            reversed_edges,
            orient=False,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(result, "Wire"):
            return None
        if transferDictionaries:
            dictionary = Topology.Dictionary(wire, silent=True)
            if dictionary:
                result = Topology.SetDictionary(result, dictionary, silent=True)
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
        Creates a ribbon from a polyline wire. Curved input wires are rejected because the current ribbon construction depends on line offsets.

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

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Ribbon - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Ribbon - Error: This implementation supports polyline wires only. Returning None.")
            return None

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
        Simplifies a polyline wire using a vertex-based simplification algorithm. Curved input wires are rejected rather than linearized.
        
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
        if not Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.Simplify - Error: This implementation supports polyline wires only. Returning None.")
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
    def Spiral(origin=None,
               radiusA: float = 0.05,
               radiusB: float = 0.5,
               height: float = 1,
               turns: int = 10,
               sides: int = 36,
               clockwise: bool = False,
               reverse: bool = False,
               direction: list = [0, 0, 1],
               placement: str = "center",
               polyline: bool = True,
               tolerance: float = 0.0001,
               silent: bool = False):
        """
        Creates an Archimedean spatial spiral between two radii.

        Curved mode approximates the exact analytic spiral with cubic Bezier/NURBS
        Edges using endpoint positions and analytic tangents. ``sides`` is the
        number of curved Edges per turn. Polyline mode preserves the historical
        straight-edge construction.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            Placement origin. Default is the global origin.
        radiusA : float , optional
            First radius. Historical behaviour orders the two radii so the spiral
            starts at the larger value. Default is 0.05.
        radiusB : float , optional
            Second radius. Default is 0.5.
        height : float , optional
            Total axial height. Default is 1.
        turns : int , optional
            Number of complete turns. Default is 10.
        sides : int , optional
            Number of curved Edges per turn, or straight segments per turn in
            polyline mode. Default is 36.
        clockwise : bool , optional
            If True, reverse the rotational sense. Default is False.
        reverse : bool , optional
            If True, axial height decreases from ``height`` to 0; otherwise it
            increases from 0 to ``height``. Default is False.
        direction : list , optional
            Spiral axis direction. Default is [0, 0, 1].
        placement : str , optional
            One of "center", "lowerleft", "upperleft", "lowerright", or
            "upperright". Default is "center".
        polyline : bool , optional
            If True, create straight segments. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created spiral.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            radiusA = float(radiusA)
            radiusB = float(radiusB)
            height = float(height)
            turns = int(turns)
            sides = int(sides)
            tolerance = float(tolerance)
        except Exception:
            return None
        if radiusA <= 0.0 or radiusB <= 0.0 or abs(radiusA - radiusB) <= tolerance:
            return None
        if radiusB > radiusA:
            radiusA, radiusB = radiusB, radiusA
        if turns <= 0 or sides < 3 or tolerance <= 0.0:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            return None

        total_angle = 2.0 * math.pi * float(turns)
        radial_rate = (radiusB - radiusA) / total_angle
        cw = -1.0 if clockwise else 1.0

        def point(parameter):
            radius = radiusA + radial_rate * parameter
            u = parameter / total_angle
            z = height * (1.0 - u) if reverse else height * u
            return [
                cw * radius * math.cos(parameter),
                radius * math.sin(parameter),
                z,
            ]

        def derivative(parameter):
            radius = radiusA + radial_rate * parameter
            dz = -height / total_angle if reverse else height / total_angle
            return [
                cw * (radial_rate * math.cos(parameter) - radius * math.sin(parameter)),
                radial_rate * math.sin(parameter) + radius * math.cos(parameter),
                dz,
            ]

        segment_count = sides * turns
        boundaries = [total_angle * float(i) / float(segment_count) for i in range(segment_count + 1)]
        sampled = [point(parameter) for parameter in boundaries]
        x_values = [p[0] for p in sampled]
        y_values = [p[1] for p in sampled]

        if polyline:
            vertices = [Vertex.ByCoordinates(*coordinates) for coordinates in sampled]
            base_wire = Wire.ByVertices(vertices, close=False, tolerance=tolerance, silent=True)
        else:
            edges = []
            for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
                dt = t1 - t0
                p0 = point(t0)
                p3 = point(t1)
                d0 = derivative(t0)
                d1 = derivative(t1)
                p1 = [p0[i] + d0[i] * dt / 3.0 for i in range(3)]
                p2 = [p3[i] - d1[i] * dt / 3.0 for i in range(3)]
                control_points = [Vertex.ByCoordinates(*coords) for coords in [p0, p1, p2, p3]]
                edge = Edge.ByNurbsParameters(
                    controlPoints=control_points,
                    weights=[1.0, 1.0, 1.0, 1.0],
                    knots=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    isRational=False,
                    isPeriodic=False,
                    degree=3,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(edge, "Edge"):
                    if not silent:
                        print("Wire.Spiral - Error: Could not create a curved spiral segment. Returning None.")
                    return None
                edges.append(edge)
            base_wire = Wire.ByEdges(edges, orient=True, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(base_wire, "Wire"):
            return None
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if placement == "center":
            base_wire = Topology.Translate(base_wire, 0, 0, -height * 0.5)
        elif placement == "lowerleft":
            base_wire = Topology.Translate(base_wire, -x_min, -y_min, 0)
        elif placement == "upperleft":
            base_wire = Topology.Translate(base_wire, -x_min, -y_max, 0)
        elif placement == "lowerright":
            base_wire = Topology.Translate(base_wire, -x_max, -y_min, 0)
        elif placement == "upperright":
            base_wire = Topology.Translate(base_wire, -x_max, -y_max, 0)
        if direction != [0, 0, 1]:
            base_wire = Topology.Orient(base_wire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return base_wire



    @staticmethod
    def Split(wire, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Splits a branching wire at vertices whose topological degree is greater than two.

        Each returned run reuses the original edge geometry. Edges that must be traversed
        in the opposite direction are reversed with :meth:`Edge.Reverse`; curved edges are
        never rebuilt from their endpoints.

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
            Maximal simple runs. A one-edge run is returned as an Edge for compatibility;
            longer runs are returned as Wires.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Split - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        edges = Wire.Edges(wire, silent=True) or []
        if not edges:
            return None

        representatives = []
        endpoints = []
        adjacency = {}
        def node_index(vertex):
            for i, representative in enumerate(representatives):
                if Vertex.IsCoincident(vertex, representative, tolerance=tolerance, silent=True):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1
        for index, edge in enumerate(edges):
            a = node_index(Edge.StartVertex(edge, silent=True))
            b = node_index(Edge.EndVertex(edge, silent=True))
            endpoints.append((a, b))
            adjacency.setdefault(a, []).append(index)
            adjacency.setdefault(b, []).append(index)
        if all(len(indices) <= 2 for indices in adjacency.values()):
            return [wire]

        used = set()
        runs = []
        for seed_index in range(len(edges)):
            if seed_index in used:
                continue
            a, b = endpoints[seed_index]
            if len(adjacency[a]) != 2:
                current_node = a
            elif len(adjacency[b]) != 2:
                current_node = b
            else:
                current_node = a
            run = []
            edge_index = seed_index
            while edge_index is not None and edge_index not in used:
                source = edges[edge_index]
                a, b = endpoints[edge_index]
                if a == current_node:
                    oriented = source
                    next_node = b
                elif b == current_node:
                    oriented = Edge.Reverse(source, tolerance=tolerance, silent=True)
                    next_node = a
                else:
                    break
                if not Topology.IsInstance(oriented, "Edge"):
                    if not silent:
                        print("Wire.Split - Error: An edge could not be reoriented without altering its geometry. Returning None.")
                    return None
                run.append(oriented)
                used.add(edge_index)
                if len(adjacency.get(next_node, [])) != 2:
                    break
                candidates = [idx for idx in adjacency[next_node] if idx not in used]
                if not candidates:
                    break
                current_node = next_node
                edge_index = candidates[0]
            if run:
                runs.append(run)

        result = []
        for run in runs:
            if len(run) == 1:
                result.append(run[0])
            else:
                run_wire = Wire.ByEdges(run, orient=False, tolerance=tolerance, silent=True)
                if Topology.IsInstance(run_wire, "Wire"):
                    result.append(run_wire)
        return result if result else [wire]
    

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
    def Squircle(origin=None,
                 radius: float = 0.5,
                 sides: int = 121,
                 a: float = 2.0,
                 b: float = 2.0,
                 direction: list = [0, 0, 1],
                 placement: str = "center",
                 angTolerance: float = 0.1,
                 polyline: bool = True,
                 tolerance: float = 0.0001,
                 silent: bool = False):
        """
        Creates a squircle/superellipse Wire.

        The analytic parameterization is evaluated exactly at every segment
        boundary. Curved mode joins those boundaries with smooth cubic NURBS
        approximations using local analytic-curve tangent directions. Polyline
        mode preserves the historical straight-edge construction.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            Placement origin. Default is the global origin.
        radius : float , optional
            Squircle radius/half-size. Default is 0.5.
        sides : int , optional
            Number of curved NURBS Edges, or straight segments in polyline mode.
            Default is 121.
        a : float , optional
            X exponent-control factor. ``a=1`` gives the circular exponent.
            Larger values produce squarer forms. Default is 2.0.
        b : float , optional
            Y exponent-control factor. Default is 2.0.
        direction : list , optional
            Squircle-plane normal. Default is [0, 0, 1].
        placement : str , optional
            One of "center", "lowerleft", "upperleft", "lowerright", or
            "upperright". Default is "center".
        angTolerance : float , optional
            Angular cleanup tolerance used only in polyline mode. Default is 0.1.
        polyline : bool , optional
            If True, create the historical straight-edge approximation. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The created squircle.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            radius = abs(float(radius))
            sides = int(sides)
            a = float(a)
            b = float(b)
            tolerance = float(tolerance)
        except Exception:
            return None
        if radius <= tolerance or a <= 0.0 or b <= 0.0 or sides < 4 or tolerance <= 0.0:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if math.sqrt(sum(value * value for value in direction)) <= tolerance:
            return None

        if abs(a - 1.0) <= 1.0e-12 and abs(b - 1.0) <= 1.0e-12:
            return Wire.Circle(
                origin=origin,
                radius=radius,
                sides=sides,
                direction=direction,
                placement=placement,
                polyline=polyline,
                tolerance=tolerance,
                silent=silent,
            )

        def point(parameter):
            cosine = math.cos(parameter)
            sine = math.sin(parameter)
            x = math.copysign(abs(cosine) ** (1.0 / a), cosine) * radius
            y = math.copysign(abs(sine) ** (1.0 / b), sine) * radius
            return [x, y, 0.0]

        if polyline:
            vertices = [Vertex.ByCoordinates(*point(2.0 * math.pi * float(i) / float(sides - 1))) for i in range(sides)]
            base_wire = Wire.ByVertices(vertices, close=True, tolerance=tolerance, silent=True)
            if Topology.IsInstance(base_wire, "Wire"):
                base_wire = Wire.RemoveCollinearEdges(base_wire, angTolerance=angTolerance, tolerance=tolerance, silent=True)
                if Topology.IsInstance(base_wire, "Wire"):
                    simplified = Wire.Simplify(base_wire, tolerance=tolerance, silent=True)
                    if simplified is not None:
                        base_wire = simplified
        else:
            parameters = [2.0 * math.pi * float(i) / float(sides) for i in range(sides + 1)]
            edges = []
            h = 2.0 * math.pi / float(sides) * 1.0e-3

            def tangent_direction(parameter):
                before = point(parameter - h)
                after = point(parameter + h)
                vector = [after[i] - before[i] for i in range(3)]
                magnitude = math.sqrt(sum(value * value for value in vector))
                if magnitude <= 1.0e-15:
                    return [0.0, 0.0, 0.0]
                return [value / magnitude for value in vector]

            for t0, t1 in zip(parameters[:-1], parameters[1:]):
                p0 = point(t0)
                p3 = point(t1)
                chord = math.sqrt(sum((p3[i] - p0[i])**2 for i in range(3)))
                if chord <= tolerance:
                    continue
                tangent0 = tangent_direction(t0)
                tangent1 = tangent_direction(t1)
                handle = chord / 3.0
                p1 = [p0[i] + tangent0[i] * handle for i in range(3)]
                p2 = [p3[i] - tangent1[i] * handle for i in range(3)]
                control_points = [Vertex.ByCoordinates(*coords) for coords in [p0, p1, p2, p3]]
                edge = Edge.ByNurbsParameters(
                    controlPoints=control_points,
                    weights=[1.0, 1.0, 1.0, 1.0],
                    knots=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    isRational=False,
                    isPeriodic=False,
                    degree=3,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(edge, "Edge"):
                    return None
                edges.append(edge)
            base_wire = Wire.ByEdges(edges, orient=True, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(base_wire, "Wire"):
            return None
        if placement == "lowerleft":
            base_wire = Topology.Translate(base_wire, radius, radius, 0)
        elif placement == "upperleft":
            base_wire = Topology.Translate(base_wire, radius, -radius, 0)
        elif placement == "lowerright":
            base_wire = Topology.Translate(base_wire, -radius, radius, 0)
        elif placement == "upperright":
            base_wire = Topology.Translate(base_wire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            base_wire = Topology.Orient(base_wire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return base_wire


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
        Returns the oriented start and end vertices of a simple open wire.

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
            Two vertices [start, end], or None for a closed/non-manifold wire.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list) or not ordered:
            return None
        start = Edge.StartVertex(ordered[0], silent=True)
        end = Edge.EndVertex(ordered[-1], silent=True)
        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            return None
        if Vertex.IsCoincident(start, end, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.StartEndVertices - Error: The input wire is closed. Returning None.")
            return None
        return [start, end]
    

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
        Returns curvilinear distance along a simple wire between an origin and a vertex.

        Distances are accumulated from the actual lengths of constituent curves. Local
        positions on curved edges are measured by trimming/evaluating the actual curve,
        never by Euclidean endpoint-chord distance.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            Target vertex lying on the wire.
        origin : topologic_core.Vertex , optional
            Distance origin. If None, the traversal start is used. Default is None.
        mantissa : int , optional
            Number of decimal places to round the result to. Use None for full precision.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            Absolute curvilinear distance, or None if either point is not on the wire.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Wire.VertexDistance - Error: One or more input parameters are invalid. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list) or not ordered:
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Edge.StartVertex(ordered[0], silent=True)
        d_vertex = Wire._DistanceFromStart(wire, vertex, tolerance=tolerance, silent=True)
        d_origin = Wire._DistanceFromStart(wire, origin, tolerance=tolerance, silent=True)
        if d_vertex is None or d_origin is None:
            if not silent:
                print("Wire.VertexDistance - Error: The target vertex or origin does not lie on the input wire. Returning None.")
            return None
        value = abs(float(d_vertex) - float(d_origin))
        return value if mantissa is None else round(value, mantissa)



    @staticmethod
    def VertexByDistance(wire, distance: float = 0.0, origin=None, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex at a signed curvilinear distance along an open manifold wire.

        Actual constituent-edge lengths are used. Closed wires are rejected because
        this method preserves the historical requirement for a unique start/end
        traversal direction.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input open manifold wire.
        distance : float , optional
            Signed curvilinear distance. Default is 0.
        origin : topologic_core.Vertex , optional
            Origin on the wire. If None, the wire start is used. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The evaluated vertex, or None if the request is invalid.
        """
        import math
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.VertexByDistance - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            distance = float(distance)
            tolerance = float(tolerance)
        except Exception:
            return None
        if not math.isfinite(distance) or not math.isfinite(tolerance) or tolerance <= 0.0:
            return None
        if not Wire.IsManifold(wire, tolerance=tolerance, silent=True):
            return None
        if Wire.IsClosed(wire, tolerance=tolerance, silent=True):
            if not silent:
                print("Wire.VertexByDistance - Error: The input wire parameter is closed. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if not isinstance(ordered, list) or not ordered:
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tolerance, silent=True)
        if total is None or float(total) <= tolerance:
            return None
        total = float(total)
        start = Edge.StartVertex(ordered[0], silent=True)
        end = Edge.EndVertex(ordered[-1], silent=True)
        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            return None
        if abs(distance) <= tolerance:
            return start
        if abs(distance - total) <= tolerance:
            return end
        if not Topology.IsInstance(origin, "Vertex"):
            origin = start
        origin_distance = Wire._DistanceFromStart(wire, origin, tolerance=tolerance, silent=True)
        if origin_distance is None:
            return None
        if Vertex.IsCoincident(origin, end, tolerance=tolerance, silent=True):
            target = total - distance
        else:
            target = float(origin_distance) + distance
        if target < -tolerance or target > total + tolerance:
            return None
        target = max(0.0, min(total, target))
        return Wire._VertexAtDistanceFromStart(wire, target, tolerance=tolerance, silent=silent)

    



    @staticmethod
    def ParameterAtVertex(wire, vertex, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the global normalized arc-length parameter of a vertex on a simple wire.

        The parameter is based on true curvilinear distance along each constituent edge,
        including arcs and splines. For a closed wire the chosen traversal seam is 0.0.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            A vertex lying on the wire.
        mantissa : int , optional
            Number of decimal places to round the result to. Use None for full precision.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            Normalized global arc-length parameter in [0, 1], or None on failure.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire") or not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Wire.ParameterAtVertex - Error: One or more input parameters are invalid. Returning None.")
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tolerance, silent=True)
        distance = Wire._DistanceFromStart(wire, vertex, tolerance=tolerance, silent=True)
        if total is None or float(total) <= tolerance or distance is None:
            if not silent:
                print("Wire.ParameterAtVertex - Error: The input vertex does not lie on a valid simple wire. Returning None.")
            return None
        value = max(0.0, min(1.0, float(distance) / float(total)))
        return value if mantissa is None else round(value, mantissa)


    @staticmethod
    def VertexByParameter(wire, u: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex at a global normalized arc-length parameter on a simple wire.

        Global parameterization is by true accumulated curve length, not by proportional
        interpolation of each edge's native curve parameter.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        u : float , optional
            Normalized global arc-length parameter in [0, 1]. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The evaluated vertex, or None on failure.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.VertexByParameter - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        try:
            u = float(u)
        except Exception:
            if not silent:
                print("Wire.VertexByParameter - Error: The input u parameter is not numerical. Returning None.")
            return None
        if u < 0.0 or u > 1.0:
            if not silent:
                print("Wire.VertexByParameter - Error: The input u parameter must be in [0, 1]. Returning None.")
            return None
        total = Wire.Length(wire, mantissa=None, tolerance=tolerance, silent=True)
        if total is None or float(total) <= tolerance:
            return None
        return Wire._VertexAtDistanceFromStart(
            wire,
            float(u) * float(total),
            tolerance=tolerance,
            silent=silent,
        )


    @staticmethod
    def Vertices(wire, silent: bool = False, tolerance: float = 0.0001) -> list:
        """
        Returns wire junction vertices without duplicate coincident endpoint wrappers.

        For a simple manifold wire, vertices are returned in traversal order. For a
        branching wire, coordinate-unique endpoint vertices are returned in backend order.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        tolerance : float , optional
            The desired tolerance used for endpoint uniqueness. Default is 0.0001.

        Returns
        -------
        list
            The wire vertices.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Vertices - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        ordered = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
        if isinstance(ordered, list) and ordered:
            result = [Edge.StartVertex(ordered[0], silent=True)]
            result.extend(Edge.EndVertex(edge, silent=True) for edge in ordered)
            if len(result) > 1 and Vertex.IsCoincident(result[0], result[-1], tolerance=tolerance, silent=True):
                result.pop()
            return [vertex for vertex in result if Topology.IsInstance(vertex, "Vertex")]

        raw = []
        try:
            Core.InstanceCall(wire, "Vertices", None, raw)
        except Exception:
            raw = []
        result = []
        for vertex in raw:
            if not Topology.IsInstance(vertex, "Vertex"):
                continue
            if not any(Vertex.IsCoincident(vertex, existing, tolerance=tolerance, silent=True) for existing in result):
                result.append(vertex)
        return result

