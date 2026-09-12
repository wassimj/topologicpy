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
            The number of decimal places to round the result to. Default is 6.
        bracket : bool
            If set to True, the returned angle is bracketed between 0 and 180. Default is False.

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
    def Arc(
        origin=None,
        radius: float = 0.5,
        fromAngle: float = 0.0,
        toAngle: float = 180.0,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single open circular arc Edge.

        The arc is constructed as an exact curve rather than as a polyline
        approximation. Angles are measured in degrees counter-clockwise from the
        positive local X-axis when viewed along the positive local Z-axis.

        The input direction defines the normal of the plane containing the arc.
        The resulting topology is always a single open Edge. A complete 360-degree
        circle cannot be created using this method.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the arc. If None, the global origin is used.
            The interpretation of the origin depends on the input placement
            parameter. Default is None.
        radius : float , optional
            The radius of the arc. Default is 0.5.
        fromAngle : float , optional
            The angle in degrees at which the arc starts. Default is 0.0.
        toAngle : float , optional
            The angle in degrees at which the arc ends. If this value is less than
            fromAngle, 360 degrees are added until a positive counter-clockwise
            sweep is obtained. The resulting sweep must be greater than zero and
            less than 360 degrees. Default is 180.0.
        direction : list , optional
            The vector representing the normal to the plane of the arc.
            Default is [0, 0, 1].
        placement : str , optional
            The placement of the input origin relative to the arc. The options are
            "center", "start", and "end". If set to "center", the centre of the
            underlying circle is placed at the origin. If set to "start", the start
            vertex of the arc is placed at the origin. If set to "end", the end
            vertex of the arc is placed at the origin. It is case insensitive.
            Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created open circular arc.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        # Validate tolerance.
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Arc - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Arc - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # Validate origin.
        if origin is None:
            origin = Vertex.Origin()
        elif not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Arc - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None

        # Validate radius.
        try:
            radius = abs(float(radius))
        except Exception:
            if not silent:
                print("Edge.Arc - Error: The input radius parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(radius) or radius <= tolerance:
            if not silent:
                print("Edge.Arc - Error: The input radius parameter must be greater than the input tolerance. Returning None.")
            return None

        # Validate angles.
        try:
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
        except Exception:
            if not silent:
                print("Edge.Arc - Error: The input fromAngle or toAngle parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(fromAngle) or not math.isfinite(toAngle):
            if not silent:
                print("Edge.Arc - Error: The input fromAngle and toAngle parameters must be finite numbers. Returning None.")
            return None

        while toAngle < fromAngle:
            toAngle += 360.0

        sweep = toAngle - fromAngle

        if sweep <= 1.0e-12:
            if not silent:
                print("Edge.Arc - Error: The angular sweep must be greater than zero. Returning None.")
            return None

        if sweep >= 360.0 - 1.0e-12:
            if not silent:
                print("Edge.Arc - Error: The angular sweep must be less than 360 degrees. Returning None.")
            return None

        # An open Edge must have distinguishable start and end vertices.
        chord_length = 2.0 * radius * abs(
            math.sin(math.radians(sweep) * 0.5)
        )

        if chord_length <= tolerance:
            if not silent:
                print("Edge.Arc - Error: The arc start and end vertices are closer than the input tolerance. Returning None.")
            return None

        # Validate direction.
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Arc - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            dx = float(direction[0])
            dy = float(direction[1])
            dz = float(direction[2])
        except Exception:
            if not silent:
                print("Edge.Arc - Error: The input direction parameter is not numerical. Returning None.")
            return None

        if not all(math.isfinite(value) for value in [dx, dy, dz]):
            if not silent:
                print("Edge.Arc - Error: The input direction parameter must contain finite numbers. Returning None.")
            return None

        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)

        if magnitude <= tolerance:
            if not silent:
                print("Edge.Arc - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        direction = [
            dx / magnitude,
            dy / magnitude,
            dz / magnitude,
        ]

        # Validate placement.
        if not isinstance(placement, str):
            if not silent:
                print("Edge.Arc - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        if placement not in ["center", "start", "end"]:
            if not silent:
                print("Edge.Arc - Error: The input placement string is not one of center, start, or end. Returning None.")
            return None

        # Prefer a native backend implementation.
        arc = None

        try:
            if Core.HasAttribute("EdgeUtility", "Arc"):
                arc = Core.EdgeUtility.Arc(
                    radius,
                    fromAngle,
                    toAngle,
                    tolerance,
                )
        except Exception:
            arc = None

        # TopologicCore currently exposes ByNurbsCurve but not Arc.
        if not Topology.IsInstance(arc, "Edge"):
            arc = Edge._ArcByNurbs(
                radius=radius,
                fromAngle=fromAngle,
                toAngle=toAngle,
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(arc, "Edge"):
            if not silent:
                print("Edge.Arc - Error: Could not create the circular arc. Returning None.")
            return None

        # Select the canonical placement anchor.
        if placement == "center":
            source_origin = Vertex.Origin()
        elif placement == "start":
            source_origin = Edge.StartVertex(arc, silent=True)
        else:
            source_origin = Edge.EndVertex(arc, silent=True)

        if not Topology.IsInstance(source_origin, "Vertex"):
            if not silent:
                print("Edge.Arc - Error: Could not determine the placement origin of the arc. Returning None.")
            return None

        # Orient and place in one affine transformation. This preserves the curve.
        arc = Topology.OrientAndPlace(
            arc,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(arc, "Edge"):
            if not silent:
                print("Edge.Arc - Error: Could not orient and place the circular arc. Returning None.")
            return None

        return arc

    @staticmethod
    def ArcByVertices(startVertex, middleVertex, endVertex, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates one exact circular arc Edge through three input vertices.

        Parameters
        ----------
        startVertex : topologic_core.Vertex
            The start vertex of the arc.
        middleVertex : topologic_core.Vertex
            A vertex that lies on the desired arc between start and end.
        endVertex : topologic_core.Vertex
            The end vertex of the arc.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The exact circular arc Edge, or None if the three vertices are invalid or collinear.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not all(Topology.IsInstance(v, "Vertex") for v in [startVertex, middleVertex, endVertex]):
            if not silent:
                print("Edge.ArcByVertices - Error: One or more input vertices are invalid. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            return None

        def xyz(v):
            return [float(x) for x in Vertex.Coordinates(v, mantissa=None)]

        def sub(a, b):
            return [a[i] - b[i] for i in range(3)]

        def add(a, b):
            return [a[i] + b[i] for i in range(3)]

        def mul(a, s):
            return [a[i] * s for i in range(3)]

        def dot(a, b):
            return sum(a[i] * b[i] for i in range(3))

        def cross(a, b):
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        def mag(a):
            return math.sqrt(dot(a, a))

        A, B, C = xyz(startVertex), xyz(middleVertex), xyz(endVertex)
        u = sub(B, A)
        v = sub(C, A)
        w = cross(u, v)
        w2 = dot(w, w)
        if w2 <= tolerance * tolerance:
            if not silent:
                print("Edge.ArcByVertices - Error: The three input vertices are collinear. Returning None.")
            return None

        u2 = dot(u, u)
        v2 = dot(v, v)
        offset = mul(add(mul(cross(v, w), u2), mul(cross(w, u), v2)), 1.0 / (2.0 * w2))
        center = add(A, offset)
        radius_vec = sub(A, center)
        radius = mag(radius_vec)
        if radius <= tolerance:
            return None

        normal = mul(w, 1.0 / mag(w))
        axis_x = mul(radius_vec, 1.0 / radius)
        axis_y = cross(normal, axis_x)
        axis_y_mag = mag(axis_y)
        if axis_y_mag <= tolerance:
            return None
        axis_y = mul(axis_y, 1.0 / axis_y_mag)

        def angle_of(point, y_axis):
            radial = sub(point, center)
            a = math.atan2(dot(radial, y_axis), dot(radial, axis_x))
            if a < 0.0:
                a += 2.0 * math.pi
            return a

        middle_angle = angle_of(B, axis_y)
        end_angle = angle_of(C, axis_y)
        if not (1.0e-12 < middle_angle < end_angle - 1.0e-12):
            axis_y = mul(axis_y, -1.0)
            middle_angle = angle_of(B, axis_y)
            end_angle = angle_of(C, axis_y)
        if not (1.0e-12 < middle_angle < end_angle - 1.0e-12):
            if not silent:
                print("Edge.ArcByVertices - Error: Could not determine the circular sweep through the middle vertex. Returning None.")
            return None

        sweep = end_angle
        span_count = max(1, int(math.ceil(math.degrees(sweep) / 90.0)))
        span_angle = sweep / float(span_count)
        control_points = []
        weights = []

        def point(angle, scale=1.0):
            c = math.cos(angle) * scale
            s = math.sin(angle) * scale
            return Vertex.ByCoordinates(
                center[0] + radius * (axis_x[0] * c + axis_y[0] * s),
                center[1] + radius * (axis_x[1] * c + axis_y[1] * s),
                center[2] + radius * (axis_x[2] * c + axis_y[2] * s),
            )

        for i in range(span_count):
            a0 = i * span_angle
            a1 = (i + 1) * span_angle
            am = 0.5 * (a0 + a1)
            weight = math.cos(0.5 * (a1 - a0))
            if weight <= 0.0:
                return None
            p0 = point(a0)
            p1 = point(am, 1.0 / weight)
            p2 = point(a1)
            if i == 0:
                control_points.append(p0)
                weights.append(1.0)
            control_points.extend([p1, p2])
            weights.extend([weight, 1.0])

        knots = [0.0, 0.0, 0.0]
        for i in range(1, span_count):
            k = float(i) / float(span_count)
            knots.extend([k, k])
        knots.extend([1.0, 1.0, 1.0])

        return Edge.ByNurbsParameters(
            controlPoints=control_points,
            weights=weights,
            knots=knots,
            isRational=True,
            isPeriodic=False,
            degree=2,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def _ArcByNurbs(
        radius: float = 0.5,
        fromAngle: float = 0.0,
        toAngle: float = 180.0,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates an exact open circular arc as a rational quadratic NURBS.

        This internal fallback is used when the active backend does not expose a
        native circular-arc constructor. The arc is created in the XY plane,
        centred at the global origin. Public :meth:`Edge.Arc` subsequently handles
        placement and orientation.

        Parameters
        ----------
        radius : float , optional
            The radius of the arc. Default is 0.5.
        fromAngle : float , optional
            The start angle of the arc in degrees. Default is 0.0.
        toAngle : float , optional
            The end angle of the arc in degrees. Default is 180.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created open circular arc, or None if it cannot be created.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        try:
            radius = abs(float(radius))
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Edge._ArcByNurbs - Error: One or more input parameters are invalid. Returning None.")
            return None

        if not all(math.isfinite(value) for value in [radius, fromAngle, toAngle, tolerance]):
            if not silent:
                print("Edge._ArcByNurbs - Error: One or more input parameters are not finite. Returning None.")
            return None

        if tolerance <= 0.0 or radius <= tolerance:
            if not silent:
                print("Edge._ArcByNurbs - Error: The input radius must be greater than the input tolerance. Returning None.")
            return None

        while toAngle < fromAngle:
            toAngle += 360.0

        sweep = toAngle - fromAngle
        if sweep <= 1.0e-12 or sweep >= 360.0 - 1.0e-12:
            if not silent:
                print("Edge._ArcByNurbs - Error: The angular sweep must be greater than zero and less than 360 degrees. Returning None.")
            return None

        span_count = max(1, int(math.ceil(sweep / 90.0)))
        span_angle = sweep / float(span_count)
        controlPoints = []
        weights = []

        for i in range(span_count):
            a0 = math.radians(fromAngle + i * span_angle)
            a1 = math.radians(fromAngle + (i + 1) * span_angle)
            am = 0.5 * (a0 + a1)
            weight = math.cos(0.5 * (a1 - a0))
            if weight <= 0.0:
                if not silent:
                    print("Edge._ArcByNurbs - Error: Could not compute a valid rational arc representation. Returning None.")
                return None

            p0 = Vertex.ByCoordinates(radius * math.cos(a0), radius * math.sin(a0), 0.0)
            p1 = Vertex.ByCoordinates((radius / weight) * math.cos(am), (radius / weight) * math.sin(am), 0.0)
            p2 = Vertex.ByCoordinates(radius * math.cos(a1), radius * math.sin(a1), 0.0)

            if i == 0:
                controlPoints.append(p0)
                weights.append(1.0)
            controlPoints.append(p1)
            weights.append(weight)
            controlPoints.append(p2)
            weights.append(1.0)

        knots = [0.0, 0.0, 0.0]
        for i in range(1, span_count):
            knot = float(i) / float(span_count)
            knots.extend([knot, knot])
        knots.extend([1.0, 1.0, 1.0])

        arc = Edge.ByNurbsParameters(
            controlPoints=controlPoints,
            weights=weights,
            knots=knots,
            isRational=True,
            isPeriodic=False,
            degree=2,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(arc, "Edge"):
            if not silent:
                print("Edge._ArcByNurbs - Error: Could not create the circular arc. Returning None.")
            return None
        return arc

    @staticmethod
    def Bezier(
        controlPoints,
        weights=None,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single exact Bezier curve Edge.

        The Bezier curve is represented as a clamped B-spline/NURBS curve.
        If weights are supplied, a rational Bezier curve is created; otherwise
        a non-rational Bezier curve is created. The degree is one less than the
        number of control points.

        Parameters
        ----------
        controlPoints : list
            The control vertices of the Bezier curve. At least two valid
            vertices must be supplied.
        weights : list , optional
            One finite positive weight per control point. If supplied, a
            rational Bezier curve is created. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created Bezier Edge, or None if the curve cannot be created.
        """
        import math
        from topologicpy.Topology import Topology

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Bezier - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Bezier - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not isinstance(controlPoints, (list, tuple)):
            if not silent:
                print("Edge.Bezier - Error: The input controlPoints parameter is not a valid list. Returning None.")
            return None
        controlPoints = list(controlPoints)
        if len(controlPoints) < 2:
            if not silent:
                print("Edge.Bezier - Error: At least two control points are required. Returning None.")
            return None
        if not all(Topology.IsInstance(vertex, "Vertex") for vertex in controlPoints):
            if not silent:
                print("Edge.Bezier - Error: One or more control points are not valid vertices. Returning None.")
            return None

        degree = len(controlPoints) - 1
        is_rational = weights is not None
        if weights is None:
            weights = [1.0] * len(controlPoints)
        else:
            if not isinstance(weights, (list, tuple)):
                if not silent:
                    print("Edge.Bezier - Error: The input weights parameter is not a valid list. Returning None.")
                return None
            try:
                weights = [float(value) for value in weights]
            except Exception:
                if not silent:
                    print("Edge.Bezier - Error: One or more weights are not numerical. Returning None.")
                return None
            if len(weights) != len(controlPoints):
                if not silent:
                    print("Edge.Bezier - Error: The number of weights must equal the number of control points. Returning None.")
                return None
            if any(not math.isfinite(value) or value <= 0.0 for value in weights):
                if not silent:
                    print("Edge.Bezier - Error: All weights must be finite positive numbers. Returning None.")
                return None

        # A degree-p Bezier is exactly a clamped B-spline with only the two end
        # knots, each repeated p+1 times.
        knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        edge = Edge.ByNurbsParameters(
            controlPoints=controlPoints,
            weights=weights,
            knots=knots,
            isRational=is_rational,
            isPeriodic=False,
            degree=degree,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Bezier - Error: Could not create the Bezier edge. Returning None.")
            return None
        return edge

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
            The desired length of the bisecting edge. Default is 1.0.
        placement : int , optional
            The desired placement of the bisecting edge.
            If set to 0, the bisecting edge centroid will be placed at the end vertex of the first edge.
            If set to 1, the bisecting edge start vertex will be placed at the end vertex of the first edge.
            If set to 2, the bisecting edge end vertex will be placed at the end vertex of the first edge.
            If set to any number other than 0, 1, or 2, the bisecting edge centroid will be placed at the end vertex of the first edge. Default is 0.
        tolerance : float , optional
            The desired tolerance to decide if an Edge can be created. Default is 0.0001.
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
            The desired origin of the edge. If set to None, the centroid of the face is chosen as the origin of the edge. Default is None.
        length : float , optional
            The desired length of the edge. Default is 1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
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
    def ByNurbsParameters(controlPoints,
                          weights=None,
                          knots=None,
                          isRational: bool = False,
                          isPeriodic: bool = False,
                          degree: int = 3,
                          tolerance: float = 0.0001,
                          silent: bool = False):
        """
        Creates an edge from exact NURBS/B-spline parameters.

        Parameters
        ----------
        controlPoints : list
            The control vertices (poles) of the curve.
        weights : list , optional
            One positive weight per control point. If None, all weights are 1.0.
        knots : list , optional
            Expanded nondecreasing knot vector. Repeated knots are repeated in
            the list. If None, a uniform expanded knot vector is generated.
        isRational : bool , optional
            If True, construct a rational NURBS curve. Default is False.
        isPeriodic : bool , optional
            If True, request a periodic B-spline/NURBS curve. Default is False.
        degree : int , optional
            Curve degree. Default is 3.
        tolerance : float , optional
            Geometric tolerance used for input validation. Default is 0.0001.
        silent : bool , optional
            If True, suppress diagnostics. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created edge, or None on failure.
        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.ByNurbsParameters - Error: The input tolerance is invalid. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.ByNurbsParameters - Error: The input tolerance must be greater than zero. Returning None.")
            return None

        try:
            controlPoints = Helper.Flatten(controlPoints)
        except Exception:
            controlPoints = controlPoints if isinstance(controlPoints, list) else []
        controlPoints = [v for v in controlPoints if Topology.IsInstance(v, "Vertex")]
        if len(controlPoints) < 2:
            if not silent:
                print("Edge.ByNurbsParameters - Error: Fewer than two valid control points were supplied. Returning None.")
            return None

        try:
            degree = int(degree)
        except Exception:
            if not silent:
                print("Edge.ByNurbsParameters - Error: The input degree is invalid. Returning None.")
            return None
        if degree < 1 or degree >= len(controlPoints):
            if not silent:
                print("Edge.ByNurbsParameters - Error: Degree must be at least 1 and smaller than the number of control points. Returning None.")
            return None

        isRational = bool(isRational)
        isPeriodic = bool(isPeriodic)

        if weights is None:
            weights = [1.0] * len(controlPoints)
        try:
            weights = [float(value) for value in weights]
        except Exception:
            weights = []
        if (
            len(weights) != len(controlPoints)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
        ):
            if not silent:
                print("Edge.ByNurbsParameters - Error: The weights must contain one finite positive value per control point. Returning None.")
            return None
        if not isRational:
            weights = [1.0] * len(controlPoints)

        if knots is None:
            if isPeriodic:
                # A simple uniform periodic OCCT B-spline with n poles uses n+1
                # unique knots of multiplicity 1.
                knots = [float(i) for i in range(len(controlPoints) + 1)]
            else:
                interior = len(controlPoints) - degree - 1
                knots = [0.0] * (degree + 1)
                if interior > 0:
                    knots += [float(i) / float(interior + 1) for i in range(1, interior + 1)]
                knots += [1.0] * (degree + 1)
        try:
            knots = [float(value) for value in knots]
        except Exception:
            knots = []

        if any(not math.isfinite(value) for value in knots):
            if not silent:
                print("Edge.ByNurbsParameters - Error: The knot vector contains a non-finite value. Returning None.")
            return None
        if any(knots[i] > knots[i + 1] for i in range(len(knots) - 1)):
            if not silent:
                print("Edge.ByNurbsParameters - Error: The knot vector is not nondecreasing. Returning None.")
            return None
        if len(knots) < 2 or abs(knots[-1] - knots[0]) <= 1.0e-15:
            if not silent:
                print("Edge.ByNurbsParameters - Error: The knot vector has zero parameter range. Returning None.")
            return None

        # Validate expanded knot multiplicities against the OCCT B-spline rules.
        unique_knots = []
        multiplicities = []
        for value in knots:
            if unique_knots and value == unique_knots[-1]:
                multiplicities[-1] += 1
            else:
                unique_knots.append(value)
                multiplicities.append(1)
        if isPeriodic:
            valid_knots = (
                multiplicities[0] == multiplicities[-1]
                and all(1 <= m <= degree for m in multiplicities)
                and sum(multiplicities) - multiplicities[0] == len(controlPoints)
            )
        else:
            valid_knots = (
                sum(multiplicities) == len(controlPoints) + degree + 1
                and all(1 <= m <= degree for m in multiplicities[1:-1])
                and 1 <= multiplicities[0] <= degree + 1
                and 1 <= multiplicities[-1] <= degree + 1
            )
        if not valid_knots:
            if not silent:
                print("Edge.ByNurbsParameters - Error: The knot multiplicities are incompatible with the control points, degree, and periodicity. Returning None.")
            return None

        edge = None
        try:
            if Core.HasAttribute("EdgeUtility", "ByNurbsCurve"):
                edge = Core.EdgeUtility.ByNurbsCurve(
                    controlPoints,
                    knots,
                    weights,
                    degree,
                    isPeriodic,
                    isRational,
                )
        except Exception:
            edge = None

        if not Topology.IsInstance(edge, "Edge"):
            try:
                if Core.HasAttribute("Edge", "ByNurbsParameters"):
                    edge = Core.Edge.ByNurbsParameters(
                        controlPoints,
                        weights,
                        knots,
                        isRational,
                        isPeriodic,
                        degree,
                    )
            except Exception:
                edge = None

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ByNurbsParameters - Error: The active backend could not construct the NURBS edge. Returning None.")
            return None
        return edge

    @staticmethod
    def ByCurve(controlPoints,
                degree: int = 3,
                isPeriodic: bool = False,
                tolerance: float = 0.0001,
                silent: bool = False):
        """Creates a non-rational B-spline edge using the input vertices as control points.

        This is a convenience wrapper around :meth:`Edge.ByNurbsParameters`.
        The curve is not a polyline: it remains one topological Edge backed by
        one continuous B-spline curve.
        """
        return Edge.ByNurbsParameters(
            controlPoints=controlPoints,
            weights=None,
            knots=None,
            isRational=False,
            isPeriodic=isPeriodic,
            degree=degree,
            tolerance=tolerance,
            silent=silent,
        )

    def ByOffset2D(edge, offset: float = 1.0, tolerance: float = 0.0001):
        """
        Creates an edge offset from the input edge in the XY plane.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)

        x1, y1, _ = Vertex.Coordinates(sv)
        x2, y2, _ = Vertex.Coordinates(ev)

        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length < tolerance:
            return None

        # Perpendicular vector to the left
        nx = -dy / length
        ny = dx / length

        ox = nx * offset
        oy = ny * offset

        new_sv = Vertex.ByCoordinates(x1 + ox, y1 + oy, 0)
        new_ev = Vertex.ByCoordinates(x2 + ox, y2 + oy, 0)

        return Edge.ByVertices(new_sv, new_ev)
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
            The desired tolerance to decide if an Edge can be created. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
            edge = Core.Edge.ByStartVertexEndVertex(vertexA, vertexB)
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
            The desired direction vector of the edge. Default is [0,0,1] (pointing up in the Z direction)
        length: float , optional
            The desired length of edge. Default is 1.0.
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

        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input vertex parameter is not a valid vertex. Returning None.")
            return None
        
        if length <= tolerance:
            if not silent:
                print("Edge.ByOriginDirectionLength - Error: The input edge parameter must not be less than the input tolerance parameter. Returning None.")
            return None

        endVertex = Topology.TranslateByDirectionDistance(origin, direction=direction[:3], distance=length)
        edge = Edge.ByVertices(origin, endVertex, tolerance=tolerance, silent=silent)
        return edge

    @staticmethod
    def ByVertices(*vertices, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that connects the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices. The first item is considered the start vertex and the last item is considered the end vertex.
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
        import inspect

        vertexList = list(vertices)
        vertexList = Helper.Flatten(vertexList)
        vertexList = [v for v in vertexList if Topology.IsInstance(v, "vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Edge.ByVertices - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if len(vertexList) == 1:
            if not silent:
                print("Edge.ByVertices - Warning: The input vertices parameter contains only one vertex. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
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
            The desired tolerance to decide if an edge can be created. Default is 0.0001.

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
    def Circle(
        origin=None,
        radius: float = 0.5,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single closed circular Edge.

        The circle is constructed as an exact closed curve rather than as a
        polyline approximation. It is initially defined in a local XY plane and
        oriented such that its positive local Z-axis aligns with the input
        direction.

        The seam of the closed Edge is located on the positive local X-axis.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the circle. If None, the global origin is
            used. The interpretation of this origin depends on the input
            placement parameter. Default is None.
        radius : float , optional
            The radius of the circle. Default is 0.5.
        direction : list , optional
            The vector representing the normal to the plane of the circle.
            Default is [0, 0, 1].
        placement : str , optional
            The placement of the input origin relative to the circle. The options
            are "center", "lowerleft", "upperleft", "lowerright", and
            "upperright". These correspond to the centre or a corner of the
            circle's local bounding square. It is case insensitive.
            Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created closed circular Edge.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        # Validate tolerance.
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Circle - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Circle - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # Validate origin.
        if origin is None:
            origin = Vertex.Origin()

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Circle - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None

        # Validate radius.
        try:
            radius = abs(float(radius))
        except Exception:
            if not silent:
                print("Edge.Circle - Error: The input radius parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(radius) or radius <= tolerance:
            if not silent:
                print("Edge.Circle - Error: The input radius parameter must be greater than the input tolerance. Returning None.")
            return None

        # Validate direction.
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Circle - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            dx = float(direction[0])
            dy = float(direction[1])
            dz = float(direction[2])
        except Exception:
            if not silent:
                print("Edge.Circle - Error: The input direction parameter is not numerical. Returning None.")
            return None

        if not all(math.isfinite(value) for value in [dx, dy, dz]):
            if not silent:
                print("Edge.Circle - Error: The input direction parameter must contain finite numbers. Returning None.")
            return None

        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)

        if magnitude <= tolerance:
            if not silent:
                print("Edge.Circle - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        direction = [
            dx / magnitude,
            dy / magnitude,
            dz / magnitude,
        ]

        # Validate placement.
        if not isinstance(placement, str):
            if not silent:
                print("Edge.Circle - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        valid_placements = [
            "center",
            "lowerleft",
            "upperleft",
            "lowerright",
            "upperright",
        ]

        if placement not in valid_placements:
            if not silent:
                print("Edge.Circle - Error: The input placement parameter is not a recognized string. Returning None.")
            return None

        # Determine the offset from the input placement origin to the centre of
        # the circle in the canonical local XY plane.
        if placement == "center":
            offset = [0.0, 0.0, 0.0]
        elif placement == "lowerleft":
            offset = [radius, radius, 0.0]
        elif placement == "upperleft":
            offset = [radius, -radius, 0.0]
        elif placement == "lowerright":
            offset = [-radius, radius, 0.0]
        else:  # upperright
            offset = [-radius, -radius, 0.0]

        # Compute the rotation from the canonical +Z normal to the requested
        # circle normal. This lets us construct the circle directly in its final
        # orientation rather than transforming the resulting Edge afterwards.
        matrix = Vector.TransformationMatrix(
            [0, 0, 1],
            direction,
        )

        if matrix is None:
            if not silent:
                print("Edge.Circle - Error: Could not determine the circle orientation. Returning None.")
            return None

        # Rotate the local centre offset.
        ox = (
            matrix[0][0] * offset[0] +
            matrix[0][1] * offset[1] +
            matrix[0][2] * offset[2]
        )
        oy = (
            matrix[1][0] * offset[0] +
            matrix[1][1] * offset[1] +
            matrix[1][2] * offset[2]
        )
        oz = (
            matrix[2][0] * offset[0] +
            matrix[2][1] * offset[1] +
            matrix[2][2] * offset[2]
        )

        px, py, pz = Vertex.Coordinates(origin, mantissa=None)

        center = Vertex.ByCoordinates(
            px + ox,
            py + oy,
            pz + oz,
        )

        if not Topology.IsInstance(center, "Vertex"):
            if not silent:
                print("Edge.Circle - Error: Could not determine the centre of the circle. Returning None.")
            return None

        # The canonical local +X direction is transformed by the same rotation.
        # This provides a stable location for the seam/parameter zero.
        xAxis = [
            matrix[0][0],
            matrix[1][0],
            matrix[2][0],
        ]

        try:
            if not Core.HasAttribute("EdgeUtility", "ByCircle"):
                if not silent:
                    print("Edge.Circle - Error: The active backend does not support circular edges. Returning None.")
                return None

            circle = Core.EdgeUtility.ByCircle(
                center,
                radius,
                xAxis[0],
                xAxis[1],
                xAxis[2],
                direction[0],
                direction[1],
                direction[2],
            )

        except Exception:
            circle = None

        if not Topology.IsInstance(circle, "Edge"):
            if not silent:
                print("Edge.Circle - Error: Could not create the circular edge. Returning None.")
            return None

        return circle

    @staticmethod
    def Ellipse(
        origin=None,
        inputMode: int = 1,
        width: float = 2.0,
        length: float = 1.0,
        focalLength: float = 0.866025,
        eccentricity: float = 0.866025,
        majorAxisLength: float = 1.0,
        minorAxisLength: float = 0.5,
        fromAngle: float = 0.0,
        toAngle: float = 360.0,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates one exact elliptical Edge over the requested angular interval.

        A complete 360-degree ellipse is returned as one closed rational quadratic
        NURBS Edge. A partial ellipse is returned as one open exact rational quadratic
        NURBS Edge. Angles are measured counter-clockwise from the positive local
        X-axis.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin. Default is the global origin.
        inputMode : int , optional
            1 = width/length, 2 = focalLength/eccentricity,
            3 = focalLength/minorAxisLength, 4 = majorAxisLength/minorAxisLength.
        width : float , optional
            Full local X width for mode 1. Default is 2.0.
        length : float , optional
            Full local Y length for mode 1. Default is 1.0.
        focalLength : float , optional
            Focal length for modes 2 and 3. Default is 0.866025.
        eccentricity : float , optional
            Eccentricity for mode 2. Default is 0.866025.
        majorAxisLength : float , optional
            Historical semi-axis input for mode 4. Default is 1.0.
        minorAxisLength : float , optional
            Historical semi-axis input for modes 3 and 4. Default is 0.5.
        fromAngle : float , optional
            Start angle in degrees. Default is 0.
        toAngle : float , optional
            End angle in degrees. Values below fromAngle are advanced by 360 degrees.
            The sweep must be greater than zero and no greater than 360 degrees.
        direction : list , optional
            Ellipse-plane normal. Default is [0, 0, 1].
        placement : str , optional
            "center" or "lowerleft". Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The exact elliptical Edge.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        try:
            tolerance = float(tolerance)
            inputMode = int(inputMode)
            width = abs(float(width))
            length = abs(float(length))
            focalLength = abs(float(focalLength))
            eccentricity = abs(float(eccentricity))
            majorAxisLength = abs(float(majorAxisLength))
            minorAxisLength = abs(float(minorAxisLength))
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
        except Exception:
            if not silent:
                print("Edge.Ellipse - Error: One or more input parameters are invalid. Returning None.")
            return None

        if not all(math.isfinite(v) for v in [tolerance, width, length, focalLength, eccentricity, majorAxisLength, minorAxisLength, fromAngle, toAngle]):
            return None
        if tolerance <= 0.0:
            return None

        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Ellipse - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None

        if inputMode == 1:
            if width <= tolerance or length <= tolerance:
                return None
            a, b = 0.5 * width, 0.5 * length
        elif inputMode == 2:
            if focalLength <= tolerance or eccentricity <= 0.0 or eccentricity >= 1.0:
                return None
            a = focalLength / eccentricity
            b2 = a * a - focalLength * focalLength
            if b2 <= tolerance * tolerance:
                return None
            b = math.sqrt(b2)
        elif inputMode == 3:
            if focalLength <= tolerance or minorAxisLength <= tolerance:
                return None
            b = minorAxisLength
            a = math.sqrt(b * b + focalLength * focalLength)
        elif inputMode == 4:
            if majorAxisLength <= tolerance or minorAxisLength <= tolerance:
                return None
            a, b = majorAxisLength, minorAxisLength
        else:
            return None

        while toAngle < fromAngle:
            toAngle += 360.0
        sweep = toAngle - fromAngle
        if sweep <= 1.0e-12 or sweep > 360.0 + 1.0e-9:
            if not silent:
                print("Edge.Ellipse - Error: The angular sweep must be greater than zero and no greater than 360 degrees. Returning None.")
            return None
        if abs(sweep - 360.0) <= 1.0e-9:
            sweep = 360.0
            toAngle = fromAngle + 360.0

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(v) for v in direction]
        except Exception:
            return None
        magnitude = math.sqrt(sum(v * v for v in direction))
        if magnitude <= tolerance:
            return None
        direction = [v / magnitude for v in direction]

        placement = str(placement).lower()
        if placement not in ["center", "lowerleft"]:
            return None

        span_count = max(1, int(math.ceil(sweep / 90.0)))
        span_angle = sweep / float(span_count)
        control_points = []
        weights = []

        def point(angle_radians, scale=1.0):
            return Vertex.ByCoordinates(
                a * math.cos(angle_radians) * scale,
                b * math.sin(angle_radians) * scale,
                0.0,
            )

        for i in range(span_count):
            a0 = math.radians(fromAngle + i * span_angle)
            a1 = math.radians(fromAngle + (i + 1) * span_angle)
            am = 0.5 * (a0 + a1)
            weight = math.cos(0.5 * (a1 - a0))
            if weight <= 0.0:
                return None
            p0 = point(a0)
            p1 = point(am, 1.0 / weight)
            p2 = point(a1)
            if i == 0:
                control_points.append(p0)
                weights.append(1.0)
            control_points.extend([p1, p2])
            weights.extend([weight, 1.0])

        knots = [0.0, 0.0, 0.0]
        for i in range(1, span_count):
            k = float(i) / float(span_count)
            knots.extend([k, k])
        knots.extend([1.0, 1.0, 1.0])

        ellipse = Edge.ByNurbsParameters(
            controlPoints=control_points,
            weights=weights,
            knots=knots,
            isRational=True,
            isPeriodic=False,
            degree=2,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(ellipse, "Edge"):
            return None

        source_origin = Vertex.Origin() if placement == "center" else Vertex.ByCoordinates(-a, -b, 0.0)
        ellipse = Topology.OrientAndPlace(
            ellipse,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )
        return ellipse if Topology.IsInstance(ellipse, "Edge") else None


    @staticmethod
    def Connection(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the shortest straight Edge connecting the two input Edges.

        When the active backend provides a native connection operation, closest
        points are computed from the complete edge geometries. This is important
        for curved Edges because the closest points need not be endpoint vertices.
        For backends without a native connection operation, the historical
        closest-endpoint construction is retained as a compatibility fallback.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        tolerance : float , optional
            The desired tolerance. If the minimum separation is less than or equal
            to this value, None is returned. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The shortest connecting Edge, or None if the input Edges intersect,
            touch within tolerance, or a connection cannot be created.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.Connection - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.Connection - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Connection - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Connection - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # Prefer the native backend because it sees the complete underlying
        # curve geometry rather than only the topological endpoints. If that
        # operation is available, a None result is meaningful (the Edges touch
        # or intersect within tolerance) and must not be replaced by the fallback.
        try:
            if Core.HasAttribute("EdgeUtility", "Connection"):
                result = Core.EdgeUtility.Connection(edgeA, edgeB, tolerance)
                if Topology.IsInstance(result, "Edge"):
                    return result
                return None
        except Exception:
            pass

        # Compatibility fallback for backends that do not expose Connection.
        sva = Edge.StartVertex(edgeA, silent=True)
        eva = Edge.EndVertex(edgeA, silent=True)
        svb = Edge.StartVertex(edgeB, silent=True)
        evb = Edge.EndVertex(edgeB, silent=True)
        vertices = [sva, eva, svb, evb]
        if not all(Topology.IsInstance(vertex, "Vertex") for vertex in vertices):
            if not silent:
                print("Edge.Connection - Error: Could not determine the input edge vertices. Returning None.")
            return None

        pairs = [[sva, svb], [sva, evb], [eva, svb], [eva, evb]]
        distances = [Vertex.Distance(pair[0], pair[1]) for pair in pairs]
        pairs = Helper.Sort(pairs, distances)
        closest_pair = pairs[0]
        return_edge = Edge.ByVertices(closest_pair, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(return_edge, "Edge"):
            if not silent:
                print("Edge.Connection - Warning: Could not connect the two edges. Returning None.")
            return None
        return return_edge

    
    @staticmethod
    def Direction(edge, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the unit chord direction of the input edge.

        For a curved edge, the direction is defined by the vector from its oriented start vertex to its oriented
        end vertex. It is therefore a global chord direction rather than a local curve tangent. A closed edge has
        a degenerate chord and therefore has no global chord direction.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. If None, full
            available precision is returned. Default is 6.
        tolerance : float , optional
            The desired tolerance used to detect a degenerate chord. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The unit chord direction of the input edge, or None if the chord is degenerate.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Direction - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Direction - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Direction - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None
        if mantissa is not None:
            try:
                mantissa = int(mantissa)
            except Exception:
                if not silent:
                    print("Edge.Direction - Error: The input mantissa parameter is not a valid integer. Returning None.")
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
        result = [dx / magnitude, dy / magnitude, dz / magnitude]
        return result if mantissa is None else [round(value, mantissa) for value in result]
    
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
        import inspect

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print(f"Edge.EndVertex - Error: The input edge parameter {edge} is not a valid topologic edge. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        vert = None
        try:
            #vert = edge.EndVertex() # H to Core
            vert = Core.InstanceCall(edge, "EndVertex")
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
            The number of decimal places to round the result to. Default is 6.

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
        Extends a geometrically linear edge by the input distance.

        This method is intended only for geometrically linear edges. Curved edges
        are not modified and will cause the method to return None.

        If bothSides is True, half of the input distance is added to each end of
        the edge. Otherwise, the end vertex is extended unless reverse is True, in
        which case the start vertex is extended. The dictionary of the input edge
        is transferred to the returned edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The total distance by which to extend the edge. Default is 1.0.
        bothSides : bool , optional
            If set to True, the extension is distributed equally between both ends.
            Default is True.
        reverse : bool , optional
            If bothSides is False and reverse is True, extend the start vertex.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The extended edge, or None if the operation cannot be completed.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Extend - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Extend - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Extend - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None
        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Extend - Error: The input edge is curved. This method only supports geometrically linear edges. Returning None.")
            return None
        try:
            distance = abs(float(distance))
        except Exception:
            if not silent:
                print("Edge.Extend - Error: The input distance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(distance):
            if not silent:
                print("Edge.Extend - Error: The input distance parameter must be finite. Returning None.")
            return None
        if distance <= tolerance:
            return edge

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        direction = Edge.Direction(edge, mantissa=None, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex") or direction is None:
            if not silent:
                print("Edge.Extend - Error: Could not determine valid edge endpoints or direction. Returning None.")
            return None

        sx, sy, sz = Vertex.Coordinates(start, mantissa=None)
        ex, ey, ez = Vertex.Coordinates(end, mantissa=None)
        dx, dy, dz = direction

        if bothSides:
            d = 0.5 * distance
            new_start = Vertex.ByCoordinates(sx - dx*d, sy - dy*d, sz - dz*d)
            new_end = Vertex.ByCoordinates(ex + dx*d, ey + dy*d, ez + dz*d)
        elif reverse:
            new_start = Vertex.ByCoordinates(sx - dx*distance, sy - dy*distance, sz - dz*distance)
            new_end = end
        else:
            new_start = start
            new_end = Vertex.ByCoordinates(ex + dx*distance, ey + dy*distance, ez + dz*distance)

        result = Edge.ByStartVertexEndVertex(new_start, new_end, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Extend - Error: Could not create the extended edge. Returning None.")
            return None
        try:
            dictionary = Topology.Dictionary(edge, silent=True)
            updated = Topology.SetDictionary(result, dictionary, silent=True)
            if Topology.IsInstance(updated, "Edge"):
                result = updated
        except Exception:
            pass
        return result

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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
    def GoldenSpiral(
        width: float = 1.0,
        maxIterations: int = 10,
        clockwise: bool = False,
        origin=None,
        placement: str = "center",
        direction: list = [0, 0, 1],
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates one exact golden-rectangle spiral Edge.

        The geometry is the traditional golden-rectangle construction composed of
        quarter-circle arcs. All quarter-circle spans are stored in one rational
        quadratic NURBS Edge, so the result is a single topological Edge and each
        circular span is geometrically exact.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        try:
            width = float(width)
            maxIterations = int(maxIterations)
            mantissa = int(mantissa)
            tolerance = float(tolerance)
        except Exception:
            return None
        if width <= tolerance or maxIterations <= 0 or tolerance <= 0.0:
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
            direction = [float(v) for v in direction]
        except Exception:
            return None
        if math.sqrt(sum(v * v for v in direction)) <= tolerance:
            return None

        def rnd(v):
            return round(float(v), mantissa)

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        W0, H0 = 1.0, 1.0 / phi
        rx, ry, rW, rH = -0.5 * W0, -0.5 * H0, W0, H0
        side_cycle = ["left", "bottom", "right", "top"]
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

        def P(x, y):
            return [rnd(x), rnd(y), 0.0]

        spans = []
        last_end = None
        join_tol = max(tolerance, 10.0 ** (-max(1, mantissa)))
        for sx, sy, size, side in squares:
            bl = P(sx, sy)
            br = P(sx + size, sy)
            tr = P(sx + size, sy + size)
            tl = P(sx, sy + size)
            if side == "left":
                p0, p2, center = tl, br, tr
            elif side == "bottom":
                p0, p2, center = bl, tr, tl
            elif side == "right":
                p0, p2, center = br, tl, bl
            else:
                p0, p2, center = tr, bl, br

            if last_end is not None:
                d0 = math.sqrt(sum((p0[j] - last_end[j]) ** 2 for j in range(3)))
                d2 = math.sqrt(sum((p2[j] - last_end[j]) ** 2 for j in range(3)))
                if d2 < d0:
                    p0, p2 = p2, p0
                p0 = list(last_end)

            a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
            target = math.atan2(p2[1] - center[1], p2[0] - center[0])
            candidates = [a0 + math.pi / 2.0, a0 - math.pi / 2.0]
            def angle_error(a):
                d = a - target
                while d <= -math.pi:
                    d += 2.0 * math.pi
                while d > math.pi:
                    d -= 2.0 * math.pi
                return abs(d)
            a1 = min(candidates, key=angle_error)
            am = 0.5 * (a0 + a1)
            weight = math.cos(0.5 * (a1 - a0))
            radius = math.sqrt((p0[0] - center[0]) ** 2 + (p0[1] - center[1]) ** 2)
            if radius <= tolerance or weight <= 0.0:
                return None
            p1 = [
                center[0] + radius * math.cos(am) / weight,
                center[1] + radius * math.sin(am) / weight,
                0.0,
            ]
            spans.append((p0, p1, p2, weight))
            last_end = p2

        control_points = []
        weights = []
        for i, (p0, p1, p2, weight) in enumerate(spans):
            if i == 0:
                control_points.append(Vertex.ByCoordinates(*p0))
                weights.append(1.0)
            control_points.extend([Vertex.ByCoordinates(*p1), Vertex.ByCoordinates(*p2)])
            weights.extend([weight, 1.0])

        count = len(spans)
        knots = [0.0, 0.0, 0.0]
        for i in range(1, count):
            k = float(i) / float(count)
            knots.extend([k, k])
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
        if not Topology.IsInstance(edge, "Edge"):
            return None

        local_origin = Vertex.Origin()
        if clockwise:
            edge = Topology.Scale(edge, local_origin, 1.0, -1.0, 1.0)
        edge = Topology.Scale(edge, local_origin, width, width, 1.0)
        if not Topology.IsInstance(edge, "Edge"):
            return None

        W, H = width, width / phi
        refs = {
            "center": [0.0, 0.0, 0.0],
            "lowerleft": [-0.5 * W, -0.5 * H, 0.0],
            "lowerright": [0.5 * W, -0.5 * H, 0.0],
            "upperleft": [-0.5 * W, 0.5 * H, 0.0],
            "upperright": [0.5 * W, 0.5 * H, 0.0],
        }
        source_origin = Vertex.ByCoordinates(*refs[placement])
        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )
        return edge if Topology.IsInstance(edge, "Edge") else None

    @staticmethod
    def Helix(
        origin=None,
        radius: float = 0.5,
        height: float = 1.0,
        turns: float = 1.0,
        sides: int = 16,
        clockwise: bool = False,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single smooth helical Edge.

        The helix is represented as one cubic B-spline Edge composed internally
        of tangent-matched cubic Bezier spans. The curve closely approximates the
        analytic circular helix while remaining one topological Edge.

        Unlike circles, ellipses, parabolas, and hyperbolas, a circular helix does
        not have an exact finite NURBS representation.

        In its canonical orientation, the helix has its axis along the positive
        Z-axis and starts at (radius, 0, 0).

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin. If None, the global origin is used.
            Default is None.
        radius : float , optional
            The radius of the helix. Default is 0.5.
        height : float , optional
            The total height of the helix measured along its axis.
            Default is 1.0.
        turns : float , optional
            The number of complete turns. Fractional turns are permitted.
            Default is 1.0.
        sides : int , optional
            The number of cubic curve spans per complete turn. Increasing this
            value improves geometric accuracy without increasing the number of
            topological Edges. Default is 16.
        clockwise : bool , optional
            If set to True, the helix rotates clockwise when viewed along its
            positive axis. Otherwise, it rotates counter-clockwise.
            Default is False.
        direction : list , optional
            The direction of the helix axis. Default is [0, 0, 1].
        placement : str , optional
            Specifies which canonical location is placed at the input origin.
            Valid options are "center", "base", "bottom", "start", and "end".
            "base" and "bottom" refer to the centre of the base circle.
            Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created helical Edge, or None if it cannot be created.

        """
        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Helix - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None

        try:
            radius = abs(float(radius))
            height = abs(float(height))
            turns = abs(float(turns))
            sides = int(math.floor(abs(float(sides))))
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Helix - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        if not all(
            math.isfinite(value)
            for value in [
                radius,
                height,
                turns,
                tolerance,
            ]
        ):
            if not silent:
                print("Edge.Helix - Error: One or more numerical parameters are not finite. Returning None.")
            return None

        if tolerance <= 0.0:
            if not silent:
                print("Edge.Helix - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if radius <= tolerance:
            if not silent:
                print("Edge.Helix - Error: The radius must be greater than the input tolerance. Returning None.")
            return None

        if height <= tolerance:
            if not silent:
                print("Edge.Helix - Error: The height must be greater than the input tolerance. Returning None.")
            return None

        if turns <= 0.0:
            if not silent:
                print("Edge.Helix - Error: The number of turns must be greater than zero. Returning None.")
            return None

        if sides < 4:
            if not silent:
                print("Edge.Helix - Error: The sides parameter must be at least 4. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Helix - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            ]
        except Exception:
            if not silent:
                print("Edge.Helix - Error: The input direction parameter is not numerical. Returning None.")
            return None

        magnitude = math.sqrt(
            sum(value * value for value in direction)
        )

        if not math.isfinite(magnitude) or magnitude <= tolerance:
            if not silent:
                print("Edge.Helix - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        direction = [
            value / magnitude
            for value in direction
        ]

        if not isinstance(placement, str):
            if not silent:
                print("Edge.Helix - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        if placement == "bottom":
            placement = "base"

        if placement not in [
            "center",
            "base",
            "start",
            "end",
        ]:
            if not silent:
                print("Edge.Helix - Error: The placement must be center, base, bottom, start, or end. Returning None.")
            return None

        orientation = -1.0 if bool(clockwise) else 1.0

        theta_total = (
            orientation
            * 2.0
            * math.pi
            * turns
        )

        # Number of cubic spans. "sides" controls approximation resolution,
        # not topological segmentation.
        span_count = max(
            1,
            int(
                math.ceil(
                    turns * sides
                )
            ),
        )

        dz_dtheta = height / theta_total

        control_points = []

        for i in range(span_count):
            theta0 = (
                theta_total
                * float(i)
                / float(span_count)
            )

            theta1 = (
                theta_total
                * float(i + 1)
                / float(span_count)
            )

            delta = theta1 - theta0

            z0 = (
                height
                * float(i)
                / float(span_count)
            )

            z1 = (
                height
                * float(i + 1)
                / float(span_count)
            )

            p0 = [
                radius * math.cos(theta0),
                radius * math.sin(theta0),
                z0,
            ]

            p3 = [
                radius * math.cos(theta1),
                radius * math.sin(theta1),
                z1,
            ]

            tangent0 = [
                -radius * math.sin(theta0),
                radius * math.cos(theta0),
                dz_dtheta,
            ]

            tangent1 = [
                -radius * math.sin(theta1),
                radius * math.cos(theta1),
                dz_dtheta,
            ]

            p1 = [
                p0[j] + delta * tangent0[j] / 3.0
                for j in range(3)
            ]

            p2 = [
                p3[j] - delta * tangent1[j] / 3.0
                for j in range(3)
            ]

            if i == 0:
                control_points.append(
                    Vertex.ByCoordinates(*p0)
                )

            control_points.append(
                Vertex.ByCoordinates(*p1)
            )

            control_points.append(
                Vertex.ByCoordinates(*p2)
            )

            control_points.append(
                Vertex.ByCoordinates(*p3)
            )

        # Piecewise cubic Bezier representation as one B-spline Edge.
        #
        # Each internal knot has multiplicity 3, producing independent cubic
        # Bezier spans while preserving the single topological Edge.
        knots = [0.0] * 4

        for i in range(1, span_count):
            knot = (
                float(i)
                / float(span_count)
            )

            knots.extend([
                knot,
                knot,
                knot,
            ])

        knots.extend([1.0] * 4)

        edge = Edge.ByNurbsParameters(
            controlPoints=control_points,
            weights=None,
            knots=knots,
            isRational=False,
            isPeriodic=False,
            degree=3,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Helix - Error: Could not create the helical edge. Returning None.")
            return None

        if placement == "center":
            source_origin = Vertex.ByCoordinates(
                0.0,
                0.0,
                height * 0.5,
            )

        elif placement == "base":
            source_origin = Vertex.Origin()

        elif placement == "start":
            source_origin = Edge.StartVertex(
                edge,
                silent=True,
            )

        else:
            source_origin = Edge.EndVertex(
                edge,
                silent=True,
            )

        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Helix - Error: Could not orient and place the helix. Returning None.")
            return None

        return edge

    def Hyperbola(
        origin=None,
        a: float = 1.0,
        b: float = 0.5,
        fromParameter: float = -1.0,
        toParameter: float = 1.0,
        branch: str = "right",
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single exact hyperbolic Edge.

        In its canonical orientation, the hyperbola lies in the XY plane and is
        defined by:

            x^2 / a^2 - y^2 / b^2 = 1

        The right branch is parametrically defined by:

            x = a*cosh(u)
            y = b*sinh(u)

        and the left branch is its reflection across the Y-axis.

        The resulting curve is represented exactly as a rational quadratic
        Bezier/NURBS curve.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin. If None, the global origin is used.
            Default is None.
        a : float , optional
            The semi-transverse axis length. Default is 1.0.
        b : float , optional
            The semi-conjugate axis length. Default is 0.5.
        fromParameter : float , optional
            The starting hyperbolic parameter. Default is -1.0.
        toParameter : float , optional
            The ending hyperbolic parameter. Default is 1.0.
        branch : str , optional
            The branch of the hyperbola. Valid options are "right" and "left".
            Default is "right".
        direction : list , optional
            The normal vector of the plane containing the hyperbola.
            Default is [0, 0, 1].
        placement : str , optional
            Specifies which canonical location is placed at the input origin.
            Valid options are "center", "vertex", "start", and "end".
            Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created exact hyperbolic Edge, or None if it cannot be created.

        """
        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Hyperbola - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None

        try:
            a = abs(float(a))
            b = abs(float(b))
            fromParameter = float(fromParameter)
            toParameter = float(toParameter)
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Hyperbola - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        if not all(
            math.isfinite(value)
            for value in [
                a,
                b,
                fromParameter,
                toParameter,
                tolerance,
            ]
        ):
            if not silent:
                print("Edge.Hyperbola - Error: One or more numerical parameters are not finite. Returning None.")
            return None

        if tolerance <= 0.0:
            if not silent:
                print("Edge.Hyperbola - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if a <= tolerance or b <= tolerance:
            if not silent:
                print("Edge.Hyperbola - Error: The a and b parameters must be greater than the input tolerance. Returning None.")
            return None

        if abs(toParameter - fromParameter) <= 1.0e-12:
            if not silent:
                print("Edge.Hyperbola - Error: The fromParameter and toParameter values must be different. Returning None.")
            return None

        if not isinstance(branch, str):
            if not silent:
                print("Edge.Hyperbola - Error: The input branch parameter is not a valid string. Returning None.")
            return None

        branch = branch.lower()

        if branch not in ["right", "left"]:
            if not silent:
                print("Edge.Hyperbola - Error: The branch must be right or left. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Hyperbola - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            ]
        except Exception:
            if not silent:
                print("Edge.Hyperbola - Error: The input direction parameter is not numerical. Returning None.")
            return None

        magnitude = math.sqrt(
            sum(value * value for value in direction)
        )

        if not math.isfinite(magnitude) or magnitude <= tolerance:
            if not silent:
                print("Edge.Hyperbola - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        direction = [
            value / magnitude
            for value in direction
        ]

        if not isinstance(placement, str):
            if not silent:
                print("Edge.Hyperbola - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        if placement not in [
            "center",
            "vertex",
            "start",
            "end",
        ]:
            if not silent:
                print("Edge.Hyperbola - Error: The placement must be center, vertex, start, or end. Returning None.")
            return None

        sign = 1.0 if branch == "right" else -1.0

        # Convert the natural hyperbolic parameter u to the rational conic
        # parameter t = tanh(u / 2).
        t0 = math.tanh(0.5 * fromParameter)
        t1 = math.tanh(0.5 * toParameter)

        w0 = 1.0 - t0 * t0
        w1 = 1.0 - t0 * t1
        w2 = 1.0 - t1 * t1

        if min(w0, w1, w2) <= 1.0e-14:
            if not silent:
                print("Edge.Hyperbola - Error: The requested parameter range is too large for a numerically stable finite hyperbola segment. Returning None.")
            return None

        # Homogeneous quadratic Bezier control points for the rational
        # parameterization:
        #
        # x = a * (1 + t^2) / (1 - t^2)
        # y = 2*b*t / (1 - t^2)
        #
        # reflected in X for the left branch.
        p0 = Vertex.ByCoordinates(
            sign * a * (1.0 + t0 * t0) / w0,
            2.0 * b * t0 / w0,
            0.0,
        )

        p1 = Vertex.ByCoordinates(
            sign * a * (1.0 + t0 * t1) / w1,
            b * (t0 + t1) / w1,
            0.0,
        )

        p2 = Vertex.ByCoordinates(
            sign * a * (1.0 + t1 * t1) / w2,
            2.0 * b * t1 / w2,
            0.0,
        )

        edge = Edge.Bezier(
            [p0, p1, p2],
            weights=[w0, w1, w2],
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Hyperbola - Error: Could not create the hyperbolic edge. Returning None.")
            return None

        if placement == "center":
            source_origin = Vertex.Origin()

        elif placement == "vertex":
            source_origin = Vertex.ByCoordinates(
                sign * a,
                0.0,
                0.0,
            )

        elif placement == "start":
            source_origin = Edge.StartVertex(
                edge,
                silent=True,
            )

        else:
            source_origin = Edge.EndVertex(
                edge,
                silent=True,
            )

        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Hyperbola - Error: Could not orient and place the hyperbola. Returning None.")
            return None

        return edge

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
            If set to True, the edge must be strictly identical to the one found in the list. Otherwise, a distance comparison is used. Default is False.
        tolerance : float , optional
            The tolerance for computing if the input edge is identical to an edge from the list. Default is 0.0001.

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
            If set to True, error and warning messages are suppressed. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

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
    def IsClosed(edge, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input edge is closed. Returns False otherwise.

        A closed edge has no distinct topological start and end boundary. Native
        backend closure detection is preferred when available. If the active
        backend does not expose such a query, closure is determined from the start
        and end vertices.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance used by the fallback closure test. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the input edge is closed. False otherwise.

        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.IsClosed - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.IsClosed - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.IsClosed - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # Prefer native backend topology.
        try:
            if Core.HasAttribute("EdgeUtility", "IsClosed"):
                try:
                    result = Core.EdgeUtility.IsClosed(edge, tolerance)
                except TypeError:
                    result = Core.EdgeUtility.IsClosed(edge)
                if isinstance(result, bool):
                    return result
        except Exception:
            pass

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)

        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            if not silent:
                print("Edge.IsClosed - Error: Could not determine the start or end vertex of the input edge. Returning None.")
            return None

        try:
            if Topology.IsSame(start, end, silent=True):
                return True
        except Exception:
            pass

        # Some backends may return separate wrappers for the same geometric
        # boundary vertex, so use coincidence as a conservative fallback.
        return bool(Vertex.IsCoincident(start, end, tolerance=tolerance, silent=True))

    @staticmethod
    def IsCollinear(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001):
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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def IsLinear(edge, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input edge is geometrically linear. Returns False otherwise.

        An edge is considered linear if its actual geometry follows one straight
        segment within the specified tolerance. Native backend classification is
        preferred. If unavailable, the method compares the exact edge length with
        the Euclidean distance between its endpoints.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance used to determine if the edge is geometrically
            linear. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if the input edge is geometrically linear. False otherwise.

        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.IsLinear - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.IsLinear - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.IsLinear - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # Prefer a backend-native geometric classification when available.
        try:
            if Core.HasAttribute("EdgeUtility", "IsLinear"):
                try:
                    result = Core.EdgeUtility.IsLinear(edge, tolerance)
                except TypeError:
                    result = Core.EdgeUtility.IsLinear(edge)
                if isinstance(result, bool):
                    return result
        except Exception:
            pass

        # Backend-neutral fallback: a non-degenerate rectifiable curve is one
        # straight segment iff its curve length equals its endpoint chord length.
        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            return False

        a = Vertex.Coordinates(start, mantissa=None)
        b = Vertex.Coordinates(end, mantissa=None)
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 3 or len(b) < 3:
            return False

        chord_length = math.sqrt(sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)))
        if chord_length <= tolerance:
            return False

        curve_length = None
        try:
            try:
                curve_length = Core.EdgeUtility.Length(edge, tolerance)
            except TypeError:
                curve_length = Core.EdgeUtility.Length(edge)
        except Exception:
            curve_length = None

        if curve_length is None:
            return False
        try:
            curve_length = float(curve_length)
        except Exception:
            return False
        if not math.isfinite(curve_length):
            return False

        return bool(abs(curve_length - chord_length) <= tolerance)

    @staticmethod
    def _IsLinear(edge, tolerance: float = 0.0001) -> bool:
        """Returns True when the actual geometry is one straight segment.

        Native backend classification is preferred. Otherwise, the method uses
        the global geometric invariant that a curve is a straight segment only
        when its actual curve length equals the Euclidean distance between its
        endpoints within tolerance. This avoids classifying sampled-but-curved or
        backtracking collinear geometry as linear.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            return False
        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001

        try:
            if Core.HasAttribute("EdgeUtility", "IsLinear"):
                try:
                    result = Core.EdgeUtility.IsLinear(edge, tolerance)
                except TypeError:
                    result = Core.EdgeUtility.IsLinear(edge)
                if isinstance(result, bool):
                    return result
        except Exception:
            pass

        start_vertex = Edge.StartVertex(edge, silent=True)
        end_vertex = Edge.EndVertex(edge, silent=True)
        if not Topology.IsInstance(start_vertex, "Vertex") or not Topology.IsInstance(end_vertex, "Vertex"):
            return False
        a = Vertex.Coordinates(start_vertex, mantissa=None)
        b = Vertex.Coordinates(end_vertex, mantissa=None)
        chord_length = math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        if chord_length <= tolerance:
            return False

        curve_length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
        if curve_length is None or not math.isfinite(float(curve_length)):
            return False
        return bool(abs(float(curve_length) - chord_length) <= tolerance)

    @staticmethod
    def IsParallel(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Return True if the two input edges are parallel. Returns False otherwise.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

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
    def Length(edge, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the geometric length of the input edge.

        The length is evaluated by the active backend from the actual edge curve,
        not from the start-to-end chord.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. If None, full
            available precision is returned. Default is 6.
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
            tolerance = float(tolerance)
        except Exception:
            tolerance = 0.0001
        try:
            length = Core.EdgeUtility.Length(edge, tolerance)
        except TypeError:
            try:
                length = Core.EdgeUtility.Length(edge)
            except Exception:
                length = None
        except Exception:
            length = None
        if not isinstance(length, (int, float)) or not math.isfinite(float(length)):
            if not silent:
                print("Edge.Length - Error: Could not compute the length of the input edge. Returning None.")
            return None
        value = float(length)
        return value if mantissa is None else round(value, int(mantissa))

    @staticmethod
    def Line(origin= None, length: float = 1, direction: list = [1,0,0], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates a straight edge (line) using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. Default is None which results in the edge being placed at (0, 0, 0).
        length : float , optional
            The desired length of the edge. Default is 1.0.
        direction : list , optional
            The desired direction (vector) of the edge. Default is [1,0,0] (along the X-axis).
        placement : str , optional
            The desired placement of the edge. The options are:
            1. "center" which places the center of the edge at the origin.
            2. "start" which places the start of the edge at the origin.
            3. "end" which places the end of the edge at the origin.
            The default is "center". It is case insensitive.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
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
    def NormalAtParameter(edge, u: float = 0.5, angle: float = 0.0, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """Returns a deterministic unit normal vector to the actual edge curve.

        For a genuinely curved edge, the principal normal supplied by the active
        backend is used when defined. For a straight edge, or at zero curvature,
        the historical TopologicPy transverse-normal convention is used: an XY
        edge receives its left-hand in-plane normal. ``angle`` rotates the normal
        about the local tangent.
        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.NormalAtParameter - Error: The input edge is invalid. Returning None.")
            return None
        try:
            u = max(0.0, min(1.0, float(u)))
            angle = float(angle)
        except Exception:
            if not silent:
                print("Edge.NormalAtParameter - Error: The input u or angle parameter is not numerical. Returning None.")
            return None

        tangent = Edge.TangentAtParameter(edge, u=u, mantissa=None, tolerance=tolerance, silent=True)
        if tangent is None:
            return None
        tx, ty, tz = [float(value) for value in tangent[:3]]

        normal = None
        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            try:
                if Core.HasAttribute("EdgeUtility", "NormalAtParameter"):
                    normal = Core.EdgeUtility.NormalAtParameter(edge, u)
            except Exception:
                normal = None
            if normal is None:
                du = max(1.0e-6, min(1.0e-3, abs(float(tolerance))*10.0))
                ta = Edge.TangentAtParameter(edge, u=max(0.0, u-du), mantissa=None, tolerance=tolerance, silent=True)
                tb = Edge.TangentAtParameter(edge, u=min(1.0, u+du), mantissa=None, tolerance=tolerance, silent=True)
                if ta is not None and tb is not None:
                    normal = [float(tb[i])-float(ta[i]) for i in range(3)]

        try:
            magnitude = math.sqrt(sum(float(value)*float(value) for value in normal[:3])) if normal is not None else 0.0
        except Exception:
            magnitude = 0.0
        if magnitude <= max(abs(float(tolerance)), 1.0e-12):
            if abs(tx) <= tolerance and abs(ty) <= tolerance:
                normal = [1.0, 0.0, 0.0]
            elif abs(tz) <= tolerance:
                normal = [-ty, tx, 0.0]
            else:
                normal = [ty, -tx, 0.0]

        try:
            nx, ny, nz = [float(value) for value in normal[:3]]
            magnitude = math.sqrt(nx*nx + ny*ny + nz*nz)
            if magnitude <= max(abs(float(tolerance)), 1.0e-12):
                return None
            nx, ny, nz = nx/magnitude, ny/magnitude, nz/magnitude
            if abs(angle) > tolerance:
                radians = math.radians(angle)
                c = math.cos(radians)
                s = math.sin(radians)
                dot = tx*nx + ty*ny + tz*nz
                nx, ny, nz = (
                    nx*c + (ty*nz-tz*ny)*s + tx*dot*(1-c),
                    ny*c + (tz*nx-tx*nz)*s + ty*dot*(1-c),
                    nz*c + (tx*ny-ty*nx)*s + tz*dot*(1-c),
                )
            result = [nx, ny, nz]
            return result if mantissa is None else [round(value, mantissa) for value in result]
        except Exception:
            return None

    def Normal(edge, angle: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """Returns a deterministic unit normal at the midpoint of the actual edge curve."""
        return Edge.NormalAtParameter(edge, u=0.5, angle=angle, tolerance=tolerance, silent=silent)

    @staticmethod
    def NormalEdge(edge, length: float = 1.0, u: float = 0.5, angle: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """Creates a straight edge along the local normal of the actual input curve at parameter ``u``."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(edge, "Edge"):
            if not silent: print("Edge.NormalEdge - Error: The input edge is invalid. Returning None.")
            return None
        if not isinstance(length,(int,float)) or float(length) <= tolerance:
            if not silent: print("Edge.NormalEdge - Error: The length must be greater than tolerance. Returning None.")
            return None
        origin=Edge.VertexByParameter(edge,u=u,tolerance=tolerance,silent=True)
        normal=Edge.NormalAtParameter(edge,u=u,angle=angle,mantissa=None,tolerance=tolerance,silent=True)
        if not Topology.IsInstance(origin,"Vertex") or normal is None: return None
        x,y,z=Vertex.Coordinates(origin,mantissa=None)
        end=Vertex.ByCoordinates(x+normal[0]*length,y+normal[1]*length,z+normal[2]*length)
        return Edge.ByStartVertexEndVertex(origin,end,tolerance=tolerance,silent=silent)

    @staticmethod
    def Normalize(edge, useEndVertex: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a geometrically linear edge of unit length.

        This method is intended only for geometrically linear edges. Curved edges
        are not modified and will cause the method to return None.

        By default, the start vertex of the input edge remains fixed. If
        useEndVertex is True, the end vertex remains fixed instead.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        useEndVertex : bool , optional
            If set to False, the start vertex remains fixed. If set to True, the
            end vertex remains fixed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            A unit-length edge, or None if the input edge is curved or the
            operation cannot be completed.

        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Normalize - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Normalize - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Normalize - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Normalize - Error: The input edge is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        return Edge.SetLength(
            edge,
            length=1.0,
            bothSides=False,
            reverse=bool(useEndVertex),
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Parabola(
        origin=None,
        focalLength: float = 0.5,
        fromParameter: float = -1.0,
        toParameter: float = 1.0,
        direction: list = [0, 0, 1],
        placement: str = "vertex",
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a single exact parabolic Edge.

        In its canonical orientation, the parabola lies in the XY plane, has its
        vertex at the global origin, opens in the positive Y direction, and is
        defined parametrically by:

            x = 2*f*t
            y = f*t^2

        where f is the focal length and t is the curve parameter.

        The resulting parabola is represented exactly as a quadratic Bezier curve.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the parabola. If None, the global origin is
            used. Default is None.
        focalLength : float , optional
            The focal length of the parabola. Default is 0.5.
        fromParameter : float , optional
            The starting parametric value. Default is -1.0.
        toParameter : float , optional
            The ending parametric value. Default is 1.0.
        direction : list , optional
            The normal vector of the plane containing the parabola.
            Default is [0, 0, 1].
        placement : str , optional
            Specifies which canonical point is placed at the input origin.
            The options are "vertex", "start", and "end". Default is "vertex".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created exact parabolic Edge, or None if it cannot be created.

        """
        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Edge.Parabola - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None

        try:
            focalLength = abs(float(focalLength))
            fromParameter = float(fromParameter)
            toParameter = float(toParameter)
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Parabola - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        if not all(
            math.isfinite(value)
            for value in [
                focalLength,
                fromParameter,
                toParameter,
                tolerance,
            ]
        ):
            if not silent:
                print("Edge.Parabola - Error: One or more numerical parameters are not finite. Returning None.")
            return None

        if tolerance <= 0.0:
            if not silent:
                print("Edge.Parabola - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if focalLength <= tolerance:
            if not silent:
                print("Edge.Parabola - Error: The focal length must be greater than the input tolerance. Returning None.")
            return None

        if abs(toParameter - fromParameter) <= 1.0e-12:
            if not silent:
                print("Edge.Parabola - Error: The fromParameter and toParameter values must be different. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("Edge.Parabola - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            ]
        except Exception:
            if not silent:
                print("Edge.Parabola - Error: The input direction parameter is not numerical. Returning None.")
            return None

        magnitude = math.sqrt(
            sum(value * value for value in direction)
        )

        if not math.isfinite(magnitude) or magnitude <= tolerance:
            if not silent:
                print("Edge.Parabola - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        direction = [
            value / magnitude
            for value in direction
        ]

        if not isinstance(placement, str):
            if not silent:
                print("Edge.Parabola - Error: The input placement parameter is not a valid string. Returning None.")
            return None

        placement = placement.lower()

        if placement not in ["vertex", "start", "end"]:
            if not silent:
                print("Edge.Parabola - Error: The placement must be vertex, start, or end. Returning None.")
            return None

        t0 = fromParameter
        t1 = toParameter
        f = focalLength

        # Exact quadratic Bezier representation of:
        #
        # x = 2*f*t
        # y = f*t^2
        #
        # over t0 <= t <= t1.
        p0 = Vertex.ByCoordinates(
            2.0 * f * t0,
            f * t0 * t0,
            0.0,
        )

        p1 = Vertex.ByCoordinates(
            f * (t0 + t1),
            f * t0 * t1,
            0.0,
        )

        p2 = Vertex.ByCoordinates(
            2.0 * f * t1,
            f * t1 * t1,
            0.0,
        )

        edge = Edge.Bezier(
            [p0, p1, p2],
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Parabola - Error: Could not create the parabolic edge. Returning None.")
            return None

        if placement == "vertex":
            source_origin = Vertex.Origin()
        elif placement == "start":
            source_origin = Edge.StartVertex(
                edge,
                silent=True,
            )
        else:
            source_origin = Edge.EndVertex(
                edge,
                silent=True,
            )

        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Parabola - Error: Could not orient and place the parabola. Returning None.")
            return None

        return edge

    def ParameterAtVertex(edge, vertex, mantissa: int = 6, silent: bool = False, tolerance: float = 0.0001) -> float:
        """
        Returns the normalized curve parameter of a vertex lying on the input edge.

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
            The desired tolerance used to determine whether the vertex lies on the edge. Default is 0.0001.

        Returns
        -------
        float
            The normalized parameter in the range [0, 1], or None if the vertex is not on the edge.

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

        value = None
        try:
            if Core.HasAttribute("EdgeUtility", "ParameterAtVertex"):
                value = Core.EdgeUtility.ParameterAtVertex(edge, vertex)
            elif Core.HasAttribute("EdgeUtility", "ParameterAtPoint"):
                try:
                    value = Core.EdgeUtility.ParameterAtPoint(edge, vertex, tolerance)
                except TypeError:
                    value = Core.EdgeUtility.ParameterAtPoint(edge, vertex)
        except Exception:
            value = None

        if value is None:
            return None
        try:
            value = float(value)
            return value if mantissa is None else round(value, mantissa)
        except Exception:
            return None


    @staticmethod
    def Quadrance(edge, mantissa: int = 6) -> float:
        """
        Returns the quadrance of the input edge. See: https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        float
            The quadrance of the input edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            print("Edge.Quadrance - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        sv = Edge.StartVertex(edge)
        ev = Edge.EndVertex(edge)

        return Vertex.Quadrance(sv, ev, mantissa = mantissa)


    @staticmethod
    def Reverse(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates an edge with the reverse orientation of the input edge.

        The geometry of the input edge is preserved. Native backend reversal is
        preferred. Endpoint reconstruction is permitted only for a geometrically
        linear edge when no native reversal exists. The input dictionary is retained.

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
            The reversed edge, or None if reversal cannot be performed without
            altering its geometry.
        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Reverse - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Reverse - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Reverse - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        result = None
        try:
            if Core.HasAttribute("Edge", "Reverse"):
                try:
                    result = Core.Edge.Reverse(edge, tolerance=tolerance, silent=True)
                except TypeError:
                    result = Core.Edge.Reverse(edge)
        except Exception:
            result = None

        # Generic kernel-native orientation reversal. This is exposed by
        # TopologicCore and preserves the underlying OCCT/NURBS curve exactly.
        if not Topology.IsInstance(result, "Edge"):
            try:
                candidate = Core.InstanceCall(edge, "Reversed")
                if Topology.IsInstance(candidate, "Edge"):
                    result = candidate
            except Exception:
                pass

        # Some backends expose Reversed directly on their topology objects even
        # when it is not routed through a backend namespace.
        if not Topology.IsInstance(result, "Edge"):
            try:
                candidate = edge.Reversed()
                if Topology.IsInstance(candidate, "Edge"):
                    result = candidate
            except Exception:
                pass

        # A backend-native parameter trim is another exact reversal route.
        if not Topology.IsInstance(result, "Edge"):
            try:
                result = Edge.TrimByParameters(
                    edge,
                    uA=1.0,
                    uB=0.0,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                result = None

        if not Topology.IsInstance(result, "Edge") and Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            result = Edge.ByStartVertexEndVertex(
                Edge.EndVertex(edge, silent=True),
                Edge.StartVertex(edge, silent=True),
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Reverse - Error: The active backend could not reverse the input edge without altering its geometry. Returning None.")
            return None
        try:
            updated = Topology.SetDictionary(result, Topology.Dictionary(edge, silent=True), silent=True)
            if Topology.IsInstance(updated, "Edge"):
                result = updated
        except Exception:
            pass
        return result
    
    @staticmethod
    def SetLength(edge, length: float = 1.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a geometrically linear edge with the requested length.

        This method is intended only for geometrically linear edges. Curved edges
        are not modified and will cause the method to return None. The input edge
        dictionary is transferred to the returned edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        length : float , optional
            The desired length of the returned edge. Default is 1.0.
        bothSides : bool , optional
            If True, preserve the midpoint. Default is True.
        reverse : bool , optional
            If bothSides is False and reverse is True, preserve the end vertex;
            otherwise preserve the start vertex. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The resized edge, or None if the operation cannot be completed.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.SetLength - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.SetLength - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.SetLength - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None
        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.SetLength - Error: The input edge is curved. This method only supports geometrically linear edges. Returning None.")
            return None
        try:
            length = float(length)
        except Exception:
            if not silent:
                print("Edge.SetLength - Error: The input length parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(length) or length <= tolerance:
            if not silent:
                print("Edge.SetLength - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        direction = Edge.Direction(edge, mantissa=None, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex") or direction is None:
            return None
        sx, sy, sz = Vertex.Coordinates(start, mantissa=None)
        ex, ey, ez = Vertex.Coordinates(end, mantissa=None)
        dx, dy, dz = direction

        if bothSides:
            mx, my, mz = 0.5*(sx+ex), 0.5*(sy+ey), 0.5*(sz+ez)
            h = 0.5*length
            new_start = Vertex.ByCoordinates(mx-dx*h, my-dy*h, mz-dz*h)
            new_end = Vertex.ByCoordinates(mx+dx*h, my+dy*h, mz+dz*h)
        elif reverse:
            new_end = end
            new_start = Vertex.ByCoordinates(ex-dx*length, ey-dy*length, ez-dz*length)
        else:
            new_start = start
            new_end = Vertex.ByCoordinates(sx+dx*length, sy+dy*length, sz+dz*length)

        result = Edge.ByStartVertexEndVertex(new_start, new_end, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.SetLength - Error: Could not create the resized edge. Returning None.")
            return None
        try:
            updated = Topology.SetDictionary(result, Topology.Dictionary(edge, silent=True), silent=True)
            if Topology.IsInstance(updated, "Edge"):
                result = updated
        except Exception:
            pass
        return result

    @staticmethod
    def Spread(edgeA, edgeB, mantissa: int = 6, bracket: bool = False) -> float:
        """
        Returns the spread between the two input edges.

        Spread is the rational trigonometry equivalent of angle and is defined as:
            spread = sin^2(theta)

        - spread = 0   : edges are parallel
        - spread = 1   : edges are perpendicular
        - 0 <= spread <= 1
        - No trigonometric functions are used

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        bracket : bool
            If set to True, the spread is bracketed to represent the acute case
            (i.e. invariant under edge reversal). Default is False.

        Returns
        -------
        float
            The spread between the two input edges.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(edgeA, "Edge"):
            print("Edge.Spread - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            print("Edge.Spread - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        # Direction vectors
        u = Edge.Direction(edgeA, mantissa=15)
        v = Edge.Direction(edgeB, mantissa=15)

        return Vector.Spread(u, v, mantissa = mantissa, bracket = bracket)

    @staticmethod
    def Spiral(
        origin=None,
        radiusA: float = 0.05,
        radiusB: float = 0.5,
        height: float = 1.0,
        turns: int = 10,
        clockwise: bool = False,
        reverse: bool = False,
        direction: list = [0, 0, 1],
        placement: str = "center",
        segmentsPerTurn: int = 12,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates one smooth cubic B-spline Edge approximating an Archimedean spatial spiral.

        ``segmentsPerTurn`` controls the internal cubic approximation only. It does not
        create multiple topological Edges; the result is always one Edge.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            radiusA = float(radiusA)
            radiusB = float(radiusB)
            height = float(height)
            turns = int(turns)
            segmentsPerTurn = int(segmentsPerTurn)
            tolerance = float(tolerance)
        except Exception:
            return None
        if radiusA <= 0.0 or radiusB <= 0.0 or abs(radiusA - radiusB) <= tolerance:
            return None
        if radiusB > radiusA:
            radiusA, radiusB = radiusB, radiusA
        if turns <= 0 or segmentsPerTurn < 4 or tolerance <= 0.0:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(v) for v in direction]
        except Exception:
            return None
        if math.sqrt(sum(v * v for v in direction)) <= tolerance:
            return None

        total_angle = 2.0 * math.pi * float(turns)
        radial_rate = (radiusB - radiusA) / total_angle
        cw = -1.0 if clockwise else 1.0

        def point(t):
            radius = radiusA + radial_rate * t
            u = t / total_angle
            z = height * (1.0 - u) if reverse else height * u
            return [cw * radius * math.cos(t), radius * math.sin(t), z]

        def derivative(t):
            radius = radiusA + radial_rate * t
            dz = -height / total_angle if reverse else height / total_angle
            return [
                cw * (radial_rate * math.cos(t) - radius * math.sin(t)),
                radial_rate * math.sin(t) + radius * math.cos(t),
                dz,
            ]

        span_count = segmentsPerTurn * turns
        boundaries = [total_angle * i / float(span_count) for i in range(span_count + 1)]
        controls = []
        sampled = []
        for i, (t0, t1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            dt = t1 - t0
            p0, p3 = point(t0), point(t1)
            d0, d1 = derivative(t0), derivative(t1)
            p1 = [p0[j] + d0[j] * dt / 3.0 for j in range(3)]
            p2 = [p3[j] - d1[j] * dt / 3.0 for j in range(3)]
            if i == 0:
                controls.append(Vertex.ByCoordinates(*p0))
            controls.extend([Vertex.ByCoordinates(*p1), Vertex.ByCoordinates(*p2), Vertex.ByCoordinates(*p3)])
            sampled.append(p0)
        sampled.append(point(boundaries[-1]))

        knots = [0.0] * 4
        for i in range(1, span_count):
            k = float(i) / float(span_count)
            knots.extend([k, k, k])
        knots.extend([1.0] * 4)
        edge = Edge.ByNurbsParameters(
            controlPoints=controls,
            weights=[1.0] * len(controls),
            knots=knots,
            isRational=False,
            isPeriodic=False,
            degree=3,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(edge, "Edge"):
            return None

        xs = [p[0] for p in sampled]
        ys = [p[1] for p in sampled]
        refs = {
            "center": [0.0, 0.0, 0.5 * height],
            "lowerleft": [min(xs), min(ys), 0.0],
            "upperleft": [min(xs), max(ys), 0.0],
            "lowerright": [max(xs), min(ys), 0.0],
            "upperright": [max(xs), max(ys), 0.0],
        }
        source_origin = Vertex.ByCoordinates(*refs[placement])
        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )
        return edge if Topology.IsInstance(edge, "Edge") else None

    @staticmethod
    def Squircle(
        origin=None,
        radius: float = 0.5,
        a: float = 2.0,
        b: float = 2.0,
        direction: list = [0, 0, 1],
        placement: str = "center",
        segments: int = 32,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates one closed cubic B-spline Edge approximating a squircle/superellipse.

        ``segments`` controls the internal cubic approximation only. The result is
        always one topological Edge. When ``a == b == 1`` an exact circular Edge is
        returned instead.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if origin is None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            radius = abs(float(radius))
            a = float(a)
            b = float(b)
            segments = int(segments)
            tolerance = float(tolerance)
        except Exception:
            return None
        if radius <= tolerance or a <= 0.0 or b <= 0.0 or segments < 8 or tolerance <= 0.0:
            return None
        placement = str(placement).lower()
        if placement not in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            direction = [float(v) for v in direction]
        except Exception:
            return None
        if math.sqrt(sum(v * v for v in direction)) <= tolerance:
            return None

        if abs(a - 1.0) <= 1.0e-12 and abs(b - 1.0) <= 1.0e-12:
            return Edge.Circle(
                origin=origin,
                radius=radius,
                direction=direction,
                placement=placement,
                tolerance=tolerance,
                silent=silent,
            )

        def point(t):
            c, s = math.cos(t), math.sin(t)
            return [
                math.copysign(abs(c) ** (1.0 / a), c) * radius,
                math.copysign(abs(s) ** (1.0 / b), s) * radius,
                0.0,
            ]

        h = 2.0 * math.pi / float(segments) * 1.0e-3
        def tangent(t):
            p0, p1 = point(t - h), point(t + h)
            v = [p1[i] - p0[i] for i in range(3)]
            m = math.sqrt(sum(x * x for x in v))
            if m <= 1.0e-15:
                return [0.0, 0.0, 0.0]
            return [x / m for x in v]

        controls = []
        for i in range(segments):
            t0 = 2.0 * math.pi * i / float(segments)
            t1 = 2.0 * math.pi * (i + 1) / float(segments)
            p0, p3 = point(t0), point(t1)
            chord = math.sqrt(sum((p3[j] - p0[j]) ** 2 for j in range(3)))
            if chord <= tolerance:
                return None
            d0, d1 = tangent(t0), tangent(t1)
            handle = chord / 3.0
            p1 = [p0[j] + d0[j] * handle for j in range(3)]
            p2 = [p3[j] - d1[j] * handle for j in range(3)]
            if i == 0:
                controls.append(Vertex.ByCoordinates(*p0))
            controls.extend([Vertex.ByCoordinates(*p1), Vertex.ByCoordinates(*p2), Vertex.ByCoordinates(*p3)])

        knots = [0.0] * 4
        for i in range(1, segments):
            k = float(i) / float(segments)
            knots.extend([k, k, k])
        knots.extend([1.0] * 4)
        edge = Edge.ByNurbsParameters(
            controlPoints=controls,
            weights=[1.0] * len(controls),
            knots=knots,
            isRational=False,
            isPeriodic=False,
            degree=3,
            tolerance=tolerance,
            silent=True,
        )
        if not Topology.IsInstance(edge, "Edge"):
            return None

        refs = {
            "center": [0.0, 0.0, 0.0],
            "lowerleft": [-radius, -radius, 0.0],
            "upperleft": [-radius, radius, 0.0],
            "lowerright": [radius, -radius, 0.0],
            "upperright": [radius, radius, 0.0],
        }
        source_origin = Vertex.ByCoordinates(*refs[placement])
        edge = Topology.OrientAndPlace(
            edge,
            originA=source_origin,
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            tolerance=tolerance,
            silent=True,
        )
        return edge if Topology.IsInstance(edge, "Edge") else None

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
        import inspect

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print(f"Edge.StartVertex - Error: The input edge parameter {edge} is not a valid topologic edge. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        vert = None
        try:
            # vert = edge.StartVertex() # H to Core
            vert = Core.InstanceCall(edge, "StartVertex")
        except:
            vert = None
        return vert

    @staticmethod
    def TangentAtParameter(edge, u: float = 0.5, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the unit tangent vector to the actual edge curve at normalized parameter *u*.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        u : float , optional
            The normalized parameter along the edge. Default is 0.5.
        mantissa : int , optional
            The number of decimal places to round the result to. If set to None, no rounding is applied. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The unit tangent vector [x, y, z], or None if it cannot be evaluated.

        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.TangentAtParameter - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            u = max(0.0, min(1.0, float(u)))
        except Exception:
            if not silent:
                print("Edge.TangentAtParameter - Error: The input u parameter is not numerical. Returning None.")
            return None

        tangent = None
        try:
            if Core.HasAttribute("EdgeUtility", "TangentAtParameter"):
                tangent = Core.EdgeUtility.TangentAtParameter(edge, u)
        except Exception:
            tangent = None

        if tangent is None:
            # TopologicCore exposes exact point evaluation but no tangent query.
            # Numerically differentiate the actual edge curve rather than using
            # its endpoint chord, so the fallback remains valid for curves.
            du = max(1.0e-6, min(1.0e-3, tolerance * 10.0))
            ua = max(0.0, u - du)
            ub = min(1.0, u + du)
            if ub <= ua:
                return None
            va = Edge.VertexByParameter(edge, ua, tolerance=tolerance, silent=True)
            vb = Edge.VertexByParameter(edge, ub, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(va, "Vertex") or not Topology.IsInstance(vb, "Vertex"):
                return None
            a = Vertex.Coordinates(va, mantissa=None)
            b = Vertex.Coordinates(vb, mantissa=None)
            tangent = [b[i] - a[i] for i in range(3)]

        try:
            values = [float(v) for v in tangent[:3]]
            magnitude = math.sqrt(sum(v * v for v in values))
            if magnitude <= tolerance:
                return None
            values = [v / magnitude for v in values]
            return values if mantissa is None else [round(v, mantissa) for v in values]
        except Exception:
            return None

    @staticmethod
    def Trim(edge, distance: float = 0.0, bothSides: bool = True, reverse: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Trims the input edge by a geometric distance while preserving its geometry.

        The input distance represents the total amount removed from the edge. If
        bothSides is True, half of the distance is removed from each end. Otherwise,
        the distance is removed from the end of the edge unless reverse is True, in
        which case it is removed from the start.

        For curved edges, distances are measured along the actual curve rather than
        along the chord between its endpoints. The underlying curve geometry is
        preserved. If the active backend cannot trim a curved edge exactly, None is
        returned rather than replacing or approximating the curve.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The total geometric distance to remove from the edge. Negative values
            are treated as positive. Default is 0.0.
        bothSides : bool , optional
            If set to True, half of the input distance is removed from each end of
            the edge. Default is True.
        reverse : bool , optional
            If bothSides is False and reverse is True, the distance is removed from
            the start of the edge instead of the end. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge, or None if the operation cannot be completed without
            altering the curve geometry.

        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Trim - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Trim - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Trim - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            distance = abs(float(distance))
        except Exception:
            if not silent:
                print("Edge.Trim - Error: The input distance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(distance):
            if not silent:
                print("Edge.Trim - Error: The input distance parameter must be a finite number. Returning None.")
            return None

        if distance <= tolerance:
            return edge

        length = Edge.Length(
            edge,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        if length is None or length <= tolerance:
            if not silent:
                print("Edge.Trim - Error: Could not determine a valid length for the input edge. Returning None.")
            return None

        if length - distance <= tolerance:
            if not silent:
                print("Edge.Trim - Error: The input distance leaves an edge shorter than or equal to the input tolerance. Returning None.")
            return None

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)

        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            if not silent:
                print("Edge.Trim - Error: Could not determine the start or end vertex of the input edge. Returning None.")
            return None

        if bothSides:
            trim_distance = 0.5 * distance

            vertexA = Edge.VertexByDistance(
                edge,
                distance=trim_distance,
                origin=start,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

            vertexB = Edge.VertexByDistance(
                edge,
                distance=-trim_distance,
                origin=end,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

        elif reverse:
            vertexA = Edge.VertexByDistance(
                edge,
                distance=distance,
                origin=start,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

            vertexB = end

        else:
            vertexA = start

            vertexB = Edge.VertexByDistance(
                edge,
                distance=-distance,
                origin=end,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

        if not Topology.IsInstance(vertexA, "Vertex") or not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Edge.Trim - Error: Could not determine the trimming vertices. Returning None.")
            return None

        # Linear edges can be reconstructed exactly and avoid an unnecessary
        # parameter lookup.
        if Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            result = Edge.ByStartVertexEndVertex(
                vertexA,
                vertexB,
                tolerance=tolerance,
                silent=silent,
            )
            if Topology.IsInstance(result, "Edge"):
                try:
                    updated = Topology.SetDictionary(result, Topology.Dictionary(edge, silent=True), silent=True)
                    if Topology.IsInstance(updated, "Edge"):
                        result = updated
                except Exception:
                    pass
            return result

        # For curves, obtain the native curve parameters corresponding to the
        # required arc-length positions and trim the actual curve.
        uA = Edge.ParameterAtVertex(
            edge,
            vertexA,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        uB = Edge.ParameterAtVertex(
            edge,
            vertexB,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        if uA is None or uB is None:
            if not silent:
                print("Edge.Trim - Error: Could not determine the curve parameters at the trimming vertices. Returning None.")
            return None

        return Edge.TrimByParameters(
            edge,
            uA=uA,
            uB=uB,
            tolerance=tolerance,
            silent=silent,
        )

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
            If set to True, which segment is preserved is reversed. Otherwise, it is not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
                return Edge.ByVertices([eva, intVertex], tolerance=tolerance, silent=silent)
            else:
                return Edge.ByVertices([sva, intVertex], tolerance=tolerance, silent=silent)
        return edgeA

    @staticmethod
    def TrimByParameters(
        edge,
        uA: float = 0.0,
        uB: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Returns the portion of the input edge between two normalized curve parameters.

        Parameters are normalized to the range [0, 1], where 0 is the start of
        the edge and 1 is the end. If ``uA`` is greater than ``uB``, the returned
        edge is oriented from ``uA`` toward ``uB``.

        The active backend's native trimming operation is preferred so curved
        geometry is preserved exactly. If native trimming is unavailable, a
        geometrically linear edge is reconstructed exactly from its evaluated
        endpoints. Curved edges are never silently converted to line segments or
        approximated by this method.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        uA : float , optional
            The normalized parameter corresponding to the start of the returned
            edge. Default is 0.0.
        uB : float , optional
            The normalized parameter corresponding to the end of the returned
            edge. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge, or None if the operation fails.

        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.TrimByParameters - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Edge.TrimByParameters - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.TrimByParameters - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            uA = float(uA)
            uB = float(uB)
        except Exception:
            if not silent:
                print("Edge.TrimByParameters - Error: The input uA or uB parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(uA) or not math.isfinite(uB):
            if not silent:
                print("Edge.TrimByParameters - Error: The input uA and uB parameters must be finite numbers. Returning None.")
            return None

        if uA < -tolerance or uA > 1.0 + tolerance:
            if not silent:
                print("Edge.TrimByParameters - Error: The input uA parameter must be in the range [0, 1]. Returning None.")
            return None
        if uB < -tolerance or uB > 1.0 + tolerance:
            if not silent:
                print("Edge.TrimByParameters - Error: The input uB parameter must be in the range [0, 1]. Returning None.")
            return None

        uA = max(0.0, min(1.0, uA))
        uB = max(0.0, min(1.0, uB))

        if abs(uB - uA) <= 1.0e-12:
            if not silent:
                print("Edge.TrimByParameters - Error: The input parameters define a zero-length interval. Returning None.")
            return None

        if uA == 0.0 and uB == 1.0:
            return edge

        # Prefer exact backend-native trimming. This also handles a complete
        # reversal (uA=1, uB=0) without reconstructing a curved edge from only
        # its endpoints.
        try:
            if Core.HasAttribute("EdgeUtility", "Trim"):
                result = Core.EdgeUtility.Trim(edge, uA, uB)
                if Topology.IsInstance(result, "Edge"):
                    length = Edge.Length(result, mantissa=12)
                    if length is not None and length > tolerance:
                        return result
        except Exception:
            pass

        # Reconstructing from evaluated endpoints is exact for linear geometry.
        if Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            vertexA = Edge.VertexByParameter(edge, u=uA, tolerance=tolerance, silent=True)
            vertexB = Edge.VertexByParameter(edge, u=uB, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(vertexA, "Vertex") or not Topology.IsInstance(vertexB, "Vertex"):
                if not silent:
                    print("Edge.TrimByParameters - Error: Could not determine the trimmed edge vertices. Returning None.")
                return None
            return Edge.ByStartVertexEndVertex(
                vertexA,
                vertexB,
                tolerance=tolerance,
                silent=silent,
            )

        if not silent:
            print("Edge.TrimByParameters - Error: The active backend could not trim the curved input edge exactly. Returning None.")
        return None

    @staticmethod
    def VertexByDistance(edge, distance: float = 0.0, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """Returns a vertex at signed curvilinear distance from an origin on an edge.

        Native arc-length evaluation is preferred. Linear edges may be extended
        beyond their endpoints. Curved open edges are evaluated only within their
        actual finite domain. Closed edges wrap around their closed path.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            Signed curvilinear distance measured in the orientation of the edge.
            Default is 0.0.
        origin : topologic_core.Vertex , optional
            A vertex lying on the edge from which distance is measured. If None,
            the start vertex is used. Default is None.
        mantissa : int , optional
            The number of decimal places to round returned coordinates to. If None,
            full available precision is returned. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The resulting vertex, or None if the requested position cannot be
            evaluated without extrapolating an open curved edge.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.VertexByDistance - Error: The input edge is invalid. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Edge.StartVertex(edge, silent=True)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        try:
            distance = float(distance)
            tolerance = abs(float(tolerance))
        except Exception:
            return None
        if abs(distance) <= tolerance:
            return origin

        try:
            if Core.HasAttribute("EdgeUtility", "PointAtDistance"):
                vertex = Core.EdgeUtility.PointAtDistance(edge, distance, origin, tolerance)
                if Topology.IsInstance(vertex, "Vertex"):
                    coordinates = Vertex.Coordinates(vertex, mantissa=None)
                    if mantissa is None:
                        return vertex
                    return Vertex.ByCoordinates(*[round(value, mantissa) for value in coordinates])
            elif Core.HasAttribute("EdgeUtility", "VertexAtDistance"):
                vertex = Core.EdgeUtility.VertexAtDistance(edge, distance, origin)
                if Topology.IsInstance(vertex, "Vertex"):
                    coordinates = Vertex.Coordinates(vertex, mantissa=None)
                    if mantissa is None:
                        return vertex
                    return Vertex.ByCoordinates(*[round(value, mantissa) for value in coordinates])
        except Exception:
            pass

        if Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            direction = Edge.Direction(edge, mantissa=None, tolerance=tolerance, silent=True)
            if direction is None:
                return None
            if Edge.ParameterAtVertex(edge, origin, mantissa=None, tolerance=tolerance, silent=True) is None:
                return None
            x, y, z = Vertex.Coordinates(origin, mantissa=None)
            coordinates = [x + direction[0]*distance, y + direction[1]*distance, z + direction[2]*distance]
            if mantissa is not None:
                coordinates = [round(value, mantissa) for value in coordinates]
            return Vertex.ByCoordinates(*coordinates)

        u0 = Edge.ParameterAtVertex(edge, origin, mantissa=None, tolerance=tolerance, silent=True)
        if u0 is None:
            return None
        closed = bool(Edge.IsClosed(edge, tolerance=tolerance, silent=True))
        effective_distance = distance
        if closed:
            total_length = Edge.Length(edge, mantissa=None, tolerance=tolerance, silent=True)
            if total_length is None or total_length <= tolerance:
                return None
            effective_distance = math.fmod(distance, total_length)
            if abs(effective_distance) <= tolerance:
                return origin

        sign = 1.0 if effective_distance > 0.0 else -1.0
        target = abs(effective_distance)

        def _point_coordinates(u):
            vertex = Edge.VertexByParameter(edge, u=u, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(vertex, "Vertex"):
                return None
            coordinates = Vertex.Coordinates(vertex, mantissa=None)
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
                return None
            try:
                return [float(value) for value in coordinates]
            except Exception:
                return None

        def arc_length(a, b):
            """Adaptive polyline integration of curve length over [a,b]."""
            if abs(b-a) <= 1.0e-15:
                return 0.0
            p0 = _point_coordinates(a)
            p1 = _point_coordinates(b)
            if p0 is None or p1 is None:
                return None

            # The integration tolerance is intentionally tighter than the public
            # geometric tolerance because this routine is also used to invert
            # arc length back to a curve parameter.
            length_tol = max(1.0e-10, tolerance * 0.005)
            max_depth = 18

            def recurse(u0, q0, u1, q1, depth):
                um = 0.5 * (u0 + u1)
                qm = _point_coordinates(um)
                if qm is None:
                    return None
                chord = math.dist(q0, q1)
                split = math.dist(q0, qm) + math.dist(qm, q1)
                if depth >= max_depth or abs(split - chord) <= length_tol * max(1.0, split):
                    return split
                left = recurse(u0, q0, um, qm, depth + 1)
                if left is None:
                    return None
                right = recurse(um, qm, u1, q1, depth + 1)
                if right is None:
                    return None
                return left + right

            return recurse(a, p0, b, p1, 0)

        def solve_between(a, b, requested):
            available = arc_length(a, b)
            if available is None or requested > available + tolerance:
                return None
            low, high = 0.0, 1.0
            solve_tol = max(1.0e-9, tolerance * 0.01)
            for _ in range(56):
                fraction = 0.5 * (low + high)
                candidate = a + fraction * (b - a)
                length_now = arc_length(a, candidate)
                if length_now is None:
                    return None
                if abs(length_now - requested) <= solve_tol:
                    return candidate
                if length_now < requested:
                    low = fraction
                else:
                    high = fraction
            return a + 0.5 * (low + high) * (b - a)

        limit = 1.0 if sign > 0.0 else 0.0
        available = arc_length(u0, limit)
        if available is None:
            return None
        if target <= available + tolerance:
            candidate = solve_between(u0, limit, target)
        elif closed:
            remainder = target - available
            wrap_start, wrap_end = (0.0, 1.0) if sign > 0.0 else (1.0, 0.0)
            candidate = solve_between(wrap_start, wrap_end, remainder)
        else:
            return None

        if candidate is None:
            return None
        vertex = Edge.VertexByParameter(edge, u=candidate, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        if mantissa is None:
            return vertex
        coordinates = Vertex.Coordinates(vertex, mantissa=None)
        return Vertex.ByCoordinates(*[round(value, mantissa) for value in coordinates])
    
    @staticmethod
    def VertexByParameter(edge, u: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex at normalized parameter *u* on the actual edge curve.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        u : float , optional
            The normalized parameter along the edge. A parameter of 0 returns the start vertex and 1 returns the end vertex. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex, or None if the parameter cannot be evaluated.

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
        if u < -tolerance or u > 1.0 + tolerance:
            if not silent:
                print("Edge.VertexByParameter - Error: The input u parameter must be in the range [0, 1]. Returning None.")
            return None

        u = max(0.0, min(1.0, u))
        if u == 0.0:
            return Edge.StartVertex(edge, silent=True)
        if u == 1.0:
            return Edge.EndVertex(edge, silent=True)

        vertex = None
        try:
            if Core.HasAttribute("EdgeUtility", "VertexAtParameter"):
                vertex = Core.EdgeUtility.VertexAtParameter(edge, u)
            elif Core.HasAttribute("EdgeUtility", "PointAtParameter"):
                vertex = Core.EdgeUtility.PointAtParameter(edge, u)
        except Exception:
            vertex = None

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Edge.VertexByParameter - Error: The active backend could not evaluate the edge curve. Returning None.")
            return None
        return vertex

    @staticmethod
    def Vertices(edge, silent: bool = False) -> list:
        """
        Returns the list of vertices of the input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print(f"Edge.Vertices - Error: The input edge parameter {edge} is not a valid topologic edge. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        # vertices = []
        # _ = edge.Vertices(None, vertices) # H to Core
        vertices = []
        try:
            _ = Core.InstanceCall(edge, "Vertices", None, vertices)
        except Exception:
            vertices = None
        return vertices
