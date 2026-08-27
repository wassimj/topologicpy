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
    def Align2D(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a 4x4 transformation matrix that aligns one geometrically linear
        edge to another in 2D.

        The transformation translates the centroid of edgeA to the centroid of
        edgeB, uniformly scales edgeA to the length of edgeB, and rotates it about
        the global Z-axis to match the direction of edgeB.

        This method is intended only for geometrically linear edges. Curved edges
        are not represented by their endpoint chords and will cause the method to
        return None.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The source edge.
        edgeB : topologic_core.Edge
            The target edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            The 4x4 transformation matrix, or None if either input edge is curved
            or the transformation cannot be determined.

        """
        import math
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

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Align2D - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Align2D - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Align2D - Error: The input edgeA is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Align2D - Error: The input edgeB is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        lengthA = Edge.Length(
            edgeA,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        lengthB = Edge.Length(
            edgeB,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

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

        directionA = Edge.Direction(
            edgeA,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        directionB = Edge.Direction(
            edgeB,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        if directionA is None or directionB is None:
            if not silent:
                print("Edge.Align2D - Error: Could not compute valid edge directions. Returning None.")
            return None

        xyA = math.sqrt(directionA[0] ** 2 + directionA[1] ** 2)
        xyB = math.sqrt(directionB[0] ** 2 + directionB[1] ** 2)

        if xyA <= tolerance or xyB <= tolerance:
            if not silent:
                print("Edge.Align2D - Error: One or both input edges have a degenerate XY direction. Returning None.")
            return None

        move_to_origin = Matrix.ByTranslation(-x1, -y1, -z1)
        move_to_target = Matrix.ByTranslation(x2, y2, z2)

        scale_factor = lengthB / lengthA
        scaling_matrix = Matrix.ByScaling(
            scale_factor,
            scale_factor,
            1.0,
        )

        angleA = Vector.CompassAngle(directionA, [1, 0, 0])
        angleB = Vector.CompassAngle(directionB, [1, 0, 0])

        if angleA is None or angleB is None:
            if not silent:
                print("Edge.Align2D - Error: Could not compute the required rotation. Returning None.")
            return None

        rotation_matrix = Matrix.ByRotation(
            0,
            0,
            angleB - angleA,
            order="xyz",
        )

        matrix = Matrix.Multiply(
            scaling_matrix,
            move_to_origin,
        )

        matrix = Matrix.Multiply(
            rotation_matrix,
            matrix,
        )

        return Matrix.Multiply(
            move_to_target,
            matrix,
        )

    @staticmethod
    def Angle(edgeA, edgeB, mantissa: int = 6, bracket: bool = False, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the angle in degrees between two geometrically linear edges.

        This method is intended only for geometrically linear edges. An arbitrary
        curved edge does not have a single global direction, so curved edges are
        not represented by their endpoint chords and will cause the method to
        return None.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        bracket : bool , optional
            If set to True, the returned angle is bracketed to the range 0 to
            90 degrees. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        float
            The angle in degrees between the input linear edges, or None if either
            edge is curved or the angle cannot be determined.

        """
        import math
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

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Angle - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Angle - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.Angle - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.Angle - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Angle - Error: The input edgeA is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Angle - Error: The input edgeB is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        directionA = Edge.Direction(
            edgeA,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        directionB = Edge.Direction(
            edgeB,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        if directionA is None or directionB is None:
            if not silent:
                print("Edge.Angle - Error: Could not determine the directions of the input edges. Returning None.")
            return None

        angle = Vector.Angle(directionA, directionB)

        if angle is None:
            if not silent:
                print("Edge.Angle - Error: Could not determine the angle between the input edges. Returning None.")
            return None

        angle = float(angle)

        if bracket and angle > 90.0:
            angle = 180.0 - angle

        return round(angle, mantissa)

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
    def AdjacentEdges(edge, hostTopology, silent: bool = False) -> list:
        """Returns the edges adjacent to the input edge within a host topology.

        Two edges are adjacent when they share a topological end vertex.
        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.AdjacentEdges - Error: The input edge is invalid. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            if not silent:
                print("Edge.AdjacentEdges - Error: The input hostTopology is invalid. Returning None.")
            return None
        result = []
        try:
            Core.InstanceCall(edge, "AdjacentEdges", hostTopology, result)
        except Exception:
            try:
                result = Topology.AdjacentTopologies(edge, hostTopology, topologyType="edge")
            except Exception:
                result = None
        return result

    @staticmethod
    def Bisect(edgeA, edgeB, length: float = 1.0, placement: int = 0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a straight edge that bisects the local angle between two input
        edges sharing an endpoint.

        The input edges may be linear or curved. For curved edges, the local
        tangent at the shared endpoint is used rather than the start-to-end chord.
        Each tangent is oriented away from the shared vertex before the bisecting
        direction is calculated.

        Closed edges are not supported because a closed edge has two local
        directions at its seam and therefore does not define a unique outgoing
        direction for this operation.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        length : float , optional
            The desired length of the bisecting edge. Default is 1.0.
        placement : int , optional
            The desired placement of the bisecting edge. If set to 0, its centroid
            is placed at the shared vertex. If set to 1, its start vertex is placed
            at the shared vertex. If set to 2, its end vertex is placed at the
            shared vertex. Other values are treated as 0. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The created straight bisecting edge, or None if a unique bisector
            cannot be determined.

        """
        import math
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
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Bisect - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Bisect - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            length = float(length)
        except Exception:
            if not silent:
                print("Edge.Bisect - Error: The input length parameter is not numerical. Returning None.")
            return None

        if not math.isfinite(length) or length <= tolerance:
            if not silent:
                print("Edge.Bisect - Error: The input length parameter must be greater than the input tolerance. Returning None.")
            return None

        try:
            placement = int(placement)
        except Exception:
            placement = 0

        if placement not in [0, 1, 2]:
            placement = 0

        a0 = Edge.StartVertex(edgeA, silent=True)
        a1 = Edge.EndVertex(edgeA, silent=True)
        b0 = Edge.StartVertex(edgeB, silent=True)
        b1 = Edge.EndVertex(edgeB, silent=True)

        if not all(Topology.IsInstance(v, "Vertex") for v in [a0, a1, b0, b1]):
            if not silent:
                print("Edge.Bisect - Error: Could not determine the input edge vertices. Returning None.")
            return None

        # A closed edge does not have a unique outgoing direction at its seam.
        if Vertex.IsCoincident(a0, a1, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Bisect - Error: The input edgeA is closed and does not define a unique direction at its seam. Returning None.")
            return None

        if Vertex.IsCoincident(b0, b1, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Bisect - Error: The input edgeB is closed and does not define a unique direction at its seam. Returning None.")
            return None

        shared = None
        a_at_start = None
        b_at_start = None

        if Vertex.IsCoincident(a0, b0, tolerance=tolerance, silent=True):
            shared = a0
            a_at_start = True
            b_at_start = True

        elif Vertex.IsCoincident(a0, b1, tolerance=tolerance, silent=True):
            shared = a0
            a_at_start = True
            b_at_start = False

        elif Vertex.IsCoincident(a1, b0, tolerance=tolerance, silent=True):
            shared = a1
            a_at_start = False
            b_at_start = True

        elif Vertex.IsCoincident(a1, b1, tolerance=tolerance, silent=True):
            shared = a1
            a_at_start = False
            b_at_start = False

        else:
            if not silent:
                print("Edge.Bisect - Error: The input edges do not share an endpoint. Returning None.")
            return None

        def outgoing_tangent(edge, at_start):
            u = 0.0 if at_start else 1.0

            tangent = Edge.TangentAtParameter(
                edge,
                u=u,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

            if not isinstance(tangent, (list, tuple)) or len(tangent) != 3:
                return None

            try:
                tangent = [
                    float(tangent[0]),
                    float(tangent[1]),
                    float(tangent[2]),
                ]
            except Exception:
                return None

            magnitude = math.sqrt(sum(value * value for value in tangent))

            if magnitude <= tolerance:
                return None

            tangent = [value / magnitude for value in tangent]

            # At the end of an oriented Edge the tangent points into the shared
            # vertex. Reverse it so that both vectors point away from the junction.
            if not at_start:
                tangent = [-value for value in tangent]

            return tangent

        directionA = outgoing_tangent(edgeA, a_at_start)
        directionB = outgoing_tangent(edgeB, b_at_start)

        if directionA is None or directionB is None:
            if not silent:
                print("Edge.Bisect - Error: Could not determine valid endpoint tangents for the input edges. Returning None.")
            return None

        bisector = Vector.Bisect(directionA, directionB)

        if not isinstance(bisector, (list, tuple)) or len(bisector) != 3:
            if not silent:
                print("Edge.Bisect - Error: Could not determine a unique bisecting direction. Returning None.")
            return None

        try:
            bisector = [
                float(bisector[0]),
                float(bisector[1]),
                float(bisector[2]),
            ]
        except Exception:
            if not silent:
                print("Edge.Bisect - Error: Could not determine a valid bisecting direction. Returning None.")
            return None

        magnitude = math.sqrt(sum(value * value for value in bisector))

        if magnitude <= tolerance:
            if not silent:
                print("Edge.Bisect - Error: The input edge directions do not define a unique bisector. Returning None.")
            return None

        bisector = [value / magnitude for value in bisector]

        end = Topology.TranslateByDirectionDistance(
            shared,
            direction=bisector,
            distance=length,
        )

        if not Topology.IsInstance(end, "Vertex"):
            if not silent:
                print("Edge.Bisect - Error: Could not determine the end vertex of the bisecting edge. Returning None.")
            return None

        result = Edge.ByStartVertexEndVertex(
            shared,
            end,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: Could not create the bisecting edge. Returning None.")
            return None

        if placement == 0:
            result = Topology.TranslateByDirectionDistance(
                result,
                direction=[-v for v in bisector],
                distance=0.5 * length,
            )

        elif placement == 2:
            result = Topology.TranslateByDirectionDistance(
                result,
                direction=[-v for v in bisector],
                distance=length,
            )

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Bisect - Error: Could not place the bisecting edge. Returning None.")
            return None

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
        Creates a straight edge offset to the left of the input geometrically linear edge in the XY plane.

        This method is intended only for geometrically linear edges. Curved edges
        are not converted to their endpoint chords and will cause the method to
        return None.

        The returned edge lies on Z = 0, matching the historical behaviour of this
        method. Positive offset values move the edge to the left of its oriented
        start-to-end direction when viewed from the positive Z-axis.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        offset : float , optional
            The signed offset distance. Positive values offset to the left of the
            start-to-end direction. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The offset straight edge, or None if the input edge is curved or the
            operation cannot be completed.

        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ByOffset2D - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.ByOffset2D - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.ByOffset2D - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ByOffset2D - Error: The input edge is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        try:
            offset = float(offset)
        except Exception:
            if not silent:
                print("Edge.ByOffset2D - Error: The input offset parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(offset):
            if not silent:
                print("Edge.ByOffset2D - Error: The input offset parameter must be a finite number. Returning None.")
            return None

        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)

        if not Topology.IsInstance(sv, "Vertex") or not Topology.IsInstance(ev, "Vertex"):
            if not silent:
                print("Edge.ByOffset2D - Error: Could not determine the start or end vertex of the input edge. Returning None.")
            return None

        x1, y1, _ = Vertex.Coordinates(sv, mantissa=None)
        x2, y2, _ = Vertex.Coordinates(ev, mantissa=None)

        dx = x2 - x1
        dy = y2 - y1

        length = math.sqrt(dx * dx + dy * dy)

        if length <= tolerance:
            if not silent:
                print("Edge.ByOffset2D - Error: The XY projection of the input edge is degenerate. Returning None.")
            return None

        nx = -dy / length
        ny = dx / length

        new_sv = Vertex.ByCoordinates(
            x1 + nx * offset,
            y1 + ny * offset,
            0.0,
        )

        new_ev = Vertex.ByCoordinates(
            x2 + nx * offset,
            y2 + ny * offset,
            0.0,
        )

        return Edge.ByStartVertexEndVertex(
            new_sv,
            new_ev,
            tolerance=tolerance,
            silent=silent,
        )

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
    def ByNurbsParameters(controlPoints, weights=None, knots=None, isRational: bool = False, isPeriodic: bool = False, degree: int = 3, tolerance: float = 0.0001, silent: bool = False):
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
    def ByCurve(controlPoints, degree: int = 3, isPeriodic: bool = False, tolerance: float = 0.0001, silent: bool = False):
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
    def Connection(edgeA, edgeB, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the shortest straight edge connecting the two input edges.

        The complete geometry of each edge is considered. The input edges may be
        linear or curved.

        A native backend closest-distance operation is preferred when available.
        Otherwise, the actual parameterized curves are evaluated numerically to
        locate their closest points.

        If the edges intersect, touch, overlap, or otherwise have a minimum
        separation less than or equal to the input tolerance, None is returned
        because a non-degenerate connecting edge cannot be created.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The shortest straight edge connecting the two input edges, or None if
            no non-degenerate connection can be created.

        """
        import math
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

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

        # Prefer a native backend closest-points operation.
        try:
            if Core.HasAttribute("EdgeUtility", "Connection"):
                result = Core.EdgeUtility.Connection(
                    edgeA,
                    edgeB,
                    tolerance,
                )

                if Topology.IsInstance(result, "Edge"):
                    return result
        except Exception:
            pass

        # If they intersect or overlap, no non-degenerate connector exists.
        try:
            intersection = Topology.Intersect(
                edgeA,
                edgeB,
                tolerance=tolerance,
            )

            if intersection is not None:
                return None
        except Exception:
            pass

        def coordinates(edge, u):
            vertex = Edge.VertexByParameter(
                edge,
                u=u,
                tolerance=tolerance,
                silent=True,
            )

            if not Topology.IsInstance(vertex, "Vertex"):
                return None

            return Vertex.Coordinates(
                vertex,
                mantissa=None,
            )

        def squared_distance(u, v):
            pointA = coordinates(edgeA, u)
            pointB = coordinates(edgeB, v)

            if pointA is None or pointB is None:
                return float("inf")

            dx = pointA[0] - pointB[0]
            dy = pointA[1] - pointB[1]
            dz = pointA[2] - pointB[2]

            return dx * dx + dy * dy + dz * dz

        # First perform a coarse global scan. This avoids assuming that the
        # closest points occur near the endpoints or near a single local minimum.
        divisions = 10

        candidates = []

        for i in range(divisions + 1):
            u = i / divisions

            for j in range(divisions + 1):
                v = j / divisions

                value = squared_distance(u, v)

                if math.isfinite(value):
                    candidates.append(
                        [value, u, v]
                    )

        if len(candidates) == 0:
            if not silent:
                print("Edge.Connection - Error: Could not evaluate the input edge geometries. Returning None.")
            return None

        candidates.sort(key=lambda item: item[0])

        best_value, best_u, best_v = candidates[0]

        # Refine several of the best coarse candidates without introducing an
        # optimization dependency. Each seed undergoes a shrinking 3x3 local
        # search in the two normalized curve parameters.
        seed_count = min(8, len(candidates))
        initial_step = 1.0 / divisions
        for _, seed_u, seed_v in candidates[:seed_count]:
            local_u, local_v = seed_u, seed_v
            local_value = squared_distance(local_u, local_v)
            step_size = initial_step
            for _ in range(28):
                improved = False
                trial_best = (local_value, local_u, local_v)
                for du in (-step_size, 0.0, step_size):
                    for dv in (-step_size, 0.0, step_size):
                        u = max(0.0, min(1.0, local_u + du))
                        v = max(0.0, min(1.0, local_v + dv))
                        value = squared_distance(u, v)
                        if value < trial_best[0]:
                            trial_best = (value, u, v)
                            improved = True
                local_value, local_u, local_v = trial_best
                if not improved:
                    step_size *= 0.5
                if step_size <= 1.0e-10:
                    break
            if local_value < best_value:
                best_value, best_u, best_v = local_value, local_u, local_v

        if not math.isfinite(best_value):
            if not silent:
                print("Edge.Connection - Error: Could not determine the closest points between the input edges. Returning None.")
            return None

        best_distance = math.sqrt(max(0.0, best_value))

        if best_distance <= tolerance:
            return None

        vertexA = Edge.VertexByParameter(
            edgeA,
            u=best_u,
            tolerance=tolerance,
            silent=True,
        )

        vertexB = Edge.VertexByParameter(
            edgeB,
            u=best_v,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(vertexA, "Vertex") or not Topology.IsInstance(vertexB, "Vertex"):
            if not silent:
                print("Edge.Connection - Error: Could not determine the closest vertices on the input edges. Returning None.")
            return None

        result = Edge.ByStartVertexEndVertex(
            vertexA,
            vertexB,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.Connection - Warning: Could not create a non-degenerate connecting edge. Returning None.")
            return None

        return result

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
        Returns the 2D line equation of the input geometrically linear edge in the XY plane.

        This method is intended only for geometrically linear edges. Curved edges
        do not have a single line equation and will cause the method to return None.

        For non-vertical lines, the returned dictionary contains the slope and
        Y-intercept. For vertical lines, the slope is positive infinity and the
        X-intercept is returned.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        mantissa : int , optional
            The number of decimal places to round the returned values to.
            Default is 6.
        tolerance : float , optional
            The desired tolerance used to identify a degenerate XY projection or
            vertical line. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        dict
            A dictionary with keys ``slope``, ``x_intercept``, and
            ``y_intercept``, or None if the input is invalid, curved, or has a
            degenerate XY projection.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.Equation2D - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Equation2D - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Equation2D - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.Equation2D - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.Equation2D - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Equation2D - Error: The input edge is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        sv = Edge.StartVertex(edge, silent=True)
        ev = Edge.EndVertex(edge, silent=True)

        if not Topology.IsInstance(sv, "Vertex") or not Topology.IsInstance(ev, "Vertex"):
            if not silent:
                print("Edge.Equation2D - Error: Could not determine the start or end vertex of the input edge. Returning None.")
            return None

        x1, y1, _ = Vertex.Coordinates(sv, mantissa=None)
        x2, y2, _ = Vertex.Coordinates(ev, mantissa=None)

        dx = x2 - x1
        dy = y2 - y1

        length2D = math.sqrt(dx * dx + dy * dy)

        if length2D <= tolerance:
            if not silent:
                print("Edge.Equation2D - Error: The XY projection of the input edge is degenerate. Returning None.")
            return None

        if abs(dx) <= tolerance:
            return {
                "slope": float("inf"),
                "x_intercept": round(float(x1), mantissa),
                "y_intercept": None,
            }

        slope = dy / dx
        y_intercept = y1 - slope * x1

        return {
            "slope": round(float(slope), mantissa),
            "x_intercept": None,
            "y_intercept": round(float(y_intercept), mantissa),
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
        Extends the first geometrically linear edge to meet the second geometrically linear edge.

        The input edges must be linear. Their infinite supporting lines are solved
        analytically, avoiding heuristic finite extensions. If the supporting lines
        intersect outside the finite extent of edgeB, the shortest connection between
        the two finite edges is returned. Curved inputs are rejected.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The edge to extend.
        edgeB : topologic_core.Edge
            The finite target edge.
        mantissa : int , optional
            Retained for API compatibility. Default is 6.
        step : bool , optional
            Retained for API compatibility. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The extended edge, the original edgeA when it already reaches edgeB,
            a shortest connection in non-intersecting fallback cases, or None for
            invalid/curved input.
        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None
        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None
        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.ExtendToEdge - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.ExtendToEdge - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None
        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeA is curved. This method only supports geometrically linear edges. Returning None.")
            return None
        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edgeB is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        if not Edge.IsCoplanar(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Error: The input edges are not coplanar. Returning the original edge.")
            return edgeA
        if Edge.IsCollinear(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are collinear. Returning their shortest connection.")
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)
        if Edge.IsParallel(edgeA, edgeB, mantissa=mantissa, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.ExtendToEdge - Warning: The input edges are parallel. Returning their shortest connection.")
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)

        startA = Edge.StartVertex(edgeA, silent=True)
        endA = Edge.EndVertex(edgeA, silent=True)
        startB = Edge.StartVertex(edgeB, silent=True)
        endB = Edge.EndVertex(edgeB, silent=True)
        if not all(Topology.IsInstance(v, "Vertex") for v in [startA, endA, startB, endB]):
            return None

        a0 = Vertex.Coordinates(startA, mantissa=None)
        a1 = Vertex.Coordinates(endA, mantissa=None)
        b0 = Vertex.Coordinates(startB, mantissa=None)
        b1 = Vertex.Coordinates(endB, mantissa=None)

        dA = [a1[i] - a0[i] for i in range(3)]
        dB = [b1[i] - b0[i] for i in range(3)]
        lenA = math.sqrt(sum(v*v for v in dA))
        lenB = math.sqrt(sum(v*v for v in dB))
        if lenA <= tolerance or lenB <= tolerance:
            return None
        dA = [v/lenA for v in dA]
        dB = [v/lenB for v in dB]

        cross = [
            dA[1]*dB[2] - dA[2]*dB[1],
            dA[2]*dB[0] - dA[0]*dB[2],
            dA[0]*dB[1] - dA[1]*dB[0],
        ]
        denom = sum(v*v for v in cross)
        if denom <= tolerance*tolerance:
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)

        delta = [b0[i] - a0[i] for i in range(3)]
        delta_cross_b = [
            delta[1]*dB[2] - delta[2]*dB[1],
            delta[2]*dB[0] - delta[0]*dB[2],
            delta[0]*dB[1] - delta[1]*dB[0],
        ]
        tA = sum(delta_cross_b[i]*cross[i] for i in range(3)) / denom
        intersection = Vertex.ByCoordinates(*[a0[i] + tA*dA[i] for i in range(3)])
        if not Topology.IsInstance(intersection, "Vertex"):
            return None

        distance_to_b = Vertex.Distance(intersection, edgeB, mantissa=None, tolerance=tolerance, silent=True)
        if distance_to_b is None or distance_to_b > tolerance:
            return Edge.Connection(edgeA, edgeB, tolerance=tolerance, silent=silent)

        if -tolerance <= tA <= lenA + tolerance:
            return edgeA

        if tA < 0.0:
            result = Edge.ByStartVertexEndVertex(intersection, endA, tolerance=tolerance, silent=True)
        else:
            result = Edge.ByStartVertexEndVertex(startA, intersection, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(result, "Edge"):
            return None
        try:
            updated = Topology.SetDictionary(result, Topology.Dictionary(edgeA, silent=True), silent=True)
            if Topology.IsInstance(updated, "Edge"):
                result = updated
        except Exception:
            pass
        return result
    
    @staticmethod
    def ExternalBoundary(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external boundary of the input edge.

        For an open edge, the external boundary is a Cluster containing its start
        and end vertices. A closed edge has an empty boundary, in which case None
        is returned.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Cluster
            A Cluster containing the two boundary vertices of an open edge.
            None if the input edge is closed or invalid.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.ExternalBoundary - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        closed = Edge.IsClosed(
            edge,
            tolerance=tolerance,
            silent=True,
        )

        if closed is None:
            if not silent:
                print("Edge.ExternalBoundary - Error: Could not determine if the input edge is closed. Returning None.")
            return None

        if closed:
            return None

        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)

        if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
            if not silent:
                print("Edge.ExternalBoundary - Error: Could not determine the boundary vertices of the input edge. Returning None.")
            return None

        boundary = Cluster.ByTopologies(
            [start, end],
            silent=True,
        )

        if not Topology.IsInstance(boundary, "Cluster"):
            if not silent:
                print("Edge.ExternalBoundary - Error: Could not create the external boundary. Returning None.")
            return None

        return boundary
    
    @staticmethod
    def Index(edge, edges: list, strict: bool = False, tolerance: float = 0.0001, silent: bool = False) -> int:
        """
        Returns the index of the input edge in the input list of edges.

        If strict is True, the matching edge must be topologically identical to
        the input edge.

        If strict is False, geometric equivalence is tested instead. The candidate
        must have the same closure state and geometric length, and sampled points
        from each actual curve must lie on the other curve within the specified
        tolerance. Edge orientation, curve parameterization, and the seam location
        of a closed edge do not affect a non-strict match.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        edges : list
            The input list of edges.
        strict : bool , optional
            If set to True, the matching edge must be topologically identical.
            Otherwise, geometric equivalence is used. Default is False.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        int
            The index of the matching edge in the original input list, or None if
            no matching edge is found.

        """
        import math
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

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Index - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Index - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not any(Topology.IsInstance(candidate, "Edge") for candidate in edges):
            if not silent:
                print("Edge.Index - Error: The input edges parameter contains no valid edges. Returning None.")
            return None

        # Strict mode is purely topological.
        if strict:
            for index, candidate in enumerate(edges):
                if not Topology.IsInstance(candidate, "Edge"):
                    continue

                try:
                    if Topology.IsSame(edge, candidate):
                        return index
                except Exception:
                    pass

            return None

        lengthA = Edge.Length(
            edge,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        closedA = Edge.IsClosed(
            edge,
            tolerance=tolerance,
            silent=True,
        )

        if lengthA is None or closedA is None:
            if not silent:
                print("Edge.Index - Error: Could not evaluate the input edge geometry. Returning None.")
            return None

        # Point-on-edge sampling is intentionally symmetric. This avoids treating
        # a short curve that happens to lie on part of a longer curve as equal.
        sample_count = 33

        def sampled_vertices(source):
            vertices = []

            for i in range(sample_count):
                u = float(i) / float(sample_count - 1)

                vertex = Edge.VertexByParameter(
                    source,
                    u=u,
                    tolerance=tolerance,
                    silent=True,
                )

                if not Topology.IsInstance(vertex, "Vertex"):
                    return None

                vertices.append(vertex)

            return vertices

        samplesA = sampled_vertices(edge)

        if samplesA is None:
            if not silent:
                print("Edge.Index - Error: Could not evaluate the input edge geometry. Returning None.")
            return None

        def samples_lie_on(samples, target):
            for vertex in samples:
                distance = Vertex.Distance(
                    vertex,
                    target,
                    mantissa=None,
                    tolerance=tolerance,
                    silent=True,
                )

                if distance is None or distance > tolerance:
                    return False

            return True

        for index, candidate in enumerate(edges):
            if not Topology.IsInstance(candidate, "Edge"):
                continue

            # Topological identity is also geometric identity.
            try:
                if Topology.IsSame(edge, candidate):
                    return index
            except Exception:
                pass

            closedB = Edge.IsClosed(
                candidate,
                tolerance=tolerance,
                silent=True,
            )

            if closedB is None or closedA != closedB:
                continue

            lengthB = Edge.Length(
                candidate,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

            if lengthB is None:
                continue

            if abs(float(lengthA) - float(lengthB)) > tolerance:
                continue

            samplesB = sampled_vertices(candidate)

            if samplesB is None:
                continue

            if not samples_lie_on(samplesA, candidate):
                continue

            if not samples_lie_on(samplesB, edge):
                continue

            return index

        return None

    @staticmethod
    def Intersect2D(edgeA, edgeB, silent: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the intersection of the infinite 2D lines defined by two
        geometrically linear edges.

        The input edges are interpreted in the XY plane. The returned intersection
        is the intersection of their infinite supporting lines and therefore does
        not need to lie within the finite extents of either edge.

        This method is intended only for geometrically linear edges. Curved edges
        do not define unique supporting lines and will cause the method to return
        None.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.
        mantissa : int , optional
            The number of decimal places to round the returned coordinates to.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Vertex
            The intersection vertex, or None if the input edges are invalid,
            curved, parallel, collinear, or degenerate in the XY plane.

        """
        import math
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

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Intersect2D - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Intersect2D - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.Intersect2D - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.Intersect2D - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Intersect2D - Error: The input edgeA is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Intersect2D - Error: The input edgeB is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        a0 = Edge.StartVertex(edgeA, silent=True)
        a1 = Edge.EndVertex(edgeA, silent=True)
        b0 = Edge.StartVertex(edgeB, silent=True)
        b1 = Edge.EndVertex(edgeB, silent=True)

        if not all(Topology.IsInstance(v, "Vertex") for v in [a0, a1, b0, b1]):
            if not silent:
                print("Edge.Intersect2D - Error: Could not determine the input edge vertices. Returning None.")
            return None

        ax0, ay0, _ = Vertex.Coordinates(a0, mantissa=None)
        ax1, ay1, _ = Vertex.Coordinates(a1, mantissa=None)
        bx0, by0, _ = Vertex.Coordinates(b0, mantissa=None)
        bx1, by1, _ = Vertex.Coordinates(b1, mantissa=None)

        rx = ax1 - ax0
        ry = ay1 - ay0
        sx = bx1 - bx0
        sy = by1 - by0

        r_length = math.sqrt(rx * rx + ry * ry)
        s_length = math.sqrt(sx * sx + sy * sy)

        if r_length <= tolerance or s_length <= tolerance:
            if not silent:
                print("Edge.Intersect2D - Error: One or both input edges have a degenerate XY projection. Returning None.")
            return None

        denominator = rx * sy - ry * sx

        sine = abs(denominator) / (r_length * s_length)

        if sine <= tolerance:
            qpx = bx0 - ax0
            qpy = by0 - ay0

            distance = abs(qpx * ry - qpy * rx) / r_length

            if not silent:
                if distance <= tolerance:
                    print("Edge.Intersect2D - Error: The input edges are collinear. A unique intersection cannot be determined. Returning None.")
                else:
                    print("Edge.Intersect2D - Error: The input edges are parallel. Returning None.")
            return None

        qpx = bx0 - ax0
        qpy = by0 - ay0

        t = (qpx * sy - qpy * sx) / denominator

        x = ax0 + t * rx
        y = ay0 + t * ry

        return Vertex.ByCoordinates(
            round(float(x), mantissa),
            round(float(y), mantissa),
            0.0,
        )

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
            The desired tolerance used by the fallback closure test.
            Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

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
                result = Core.EdgeUtility.IsClosed(edge, tolerance)
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

        # Prefer actual topological identity.
        try:
            if Topology.IsSame(start, end):
                return True
        except Exception:
            pass

        # Conservative fallback for backends that return separate wrappers for the
        # same geometric boundary vertex.
        return bool(
            Vertex.IsCoincident(
                start,
                end,
                tolerance=tolerance,
                silent=True,
            )
        )

    @staticmethod
    def IsCollinear(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if two geometrically linear edges lie on the same infinite
        line. Returns False otherwise.

        This method is intended only for geometrically linear edges. If either
        input edge is curved, False is returned. The actual geometry of a curved
        edge is never replaced by its endpoint chord for this test.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            Retained for API compatibility. Geometric calculations are performed
            at full precision. Default is 6.
        tolerance : float , optional
            The desired distance tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if both edges are geometrically linear and collinear.
            False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge") or not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsCollinear - Error: One or both input parameters are not valid topologic edges. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.IsCollinear - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.IsCollinear - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.IsCollinear - Warning: The input edgeA is curved. Collinearity is defined only for geometrically linear edges. Returning False.")
            return False

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.IsCollinear - Warning: The input edgeB is curved. Collinearity is defined only for geometrically linear edges. Returning False.")
            return False

        a0 = Vertex.Coordinates(Edge.StartVertex(edgeA, silent=True), mantissa=None)
        a1 = Vertex.Coordinates(Edge.EndVertex(edgeA, silent=True), mantissa=None)
        b0 = Vertex.Coordinates(Edge.StartVertex(edgeB, silent=True), mantissa=None)
        b1 = Vertex.Coordinates(Edge.EndVertex(edgeB, silent=True), mantissa=None)

        ax = a1[0] - a0[0]
        ay = a1[1] - a0[1]
        az = a1[2] - a0[2]

        lengthA = math.sqrt(ax * ax + ay * ay + az * az)

        if lengthA <= tolerance:
            if not silent:
                print("Edge.IsCollinear - Error: The input edgeA is degenerate. Returning None.")
            return None

        def point_line_distance(point):
            px = point[0] - a0[0]
            py = point[1] - a0[1]
            pz = point[2] - a0[2]

            cx = py * az - pz * ay
            cy = pz * ax - px * az
            cz = px * ay - py * ax

            return math.sqrt(cx * cx + cy * cy + cz * cz) / lengthA

        return bool(
            point_line_distance(b0) <= tolerance
            and point_line_distance(b1) <= tolerance
        )
    
    @staticmethod
    def IsCoplanar(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if the two input edges lie in a common plane. Returns False otherwise.

        The input edges may be linear or curved. The method evaluates the actual
        geometry of both edges rather than their endpoint chords.

        For geometrically linear edges, the usual line-line coplanarity definition
        applies. Parallel lines are always coplanar. For curved edges, points and
        tangents are sampled along the actual curves and tested against a common
        candidate plane.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places used for scalar comparisons where
            appropriate. Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if both input edges lie in a common plane. False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.IsCoplanar - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None

        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsCoplanar - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.IsCoplanar - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.IsCoplanar - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.IsCoplanar - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.IsCoplanar - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        def _coords(edge, u):
            vertex = Edge.VertexByParameter(
                edge,
                u=u,
                tolerance=tolerance,
                silent=True,
            )
            if not Topology.IsInstance(vertex, "Vertex"):
                return None
            return Vertex.Coordinates(vertex, mantissa=None)

        def _subtract(a, b):
            return [
                a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2],
            ]

        def _cross(a, b):
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        def _dot(a, b):
            return (
                a[0] * b[0] +
                a[1] * b[1] +
                a[2] * b[2]
            )

        def _magnitude(v):
            return math.sqrt(_dot(v, v))

        # Sample the actual curves. Including several non-dyadic values helps
        # avoid repeatedly sampling geometrically symmetric locations.
        parameters = [
            0.0,
            0.0625,
            0.125,
            0.1875,
            0.25,
            0.3125,
            0.375,
            0.4375,
            0.5,
            0.5625,
            0.625,
            0.6875,
            0.75,
            0.8125,
            0.875,
            0.9375,
            1.0,
        ]

        pointsA = []
        pointsB = []

        for u in parameters:
            p = _coords(edgeA, u)
            if p is not None:
                pointsA.append(p)

            p = _coords(edgeB, u)
            if p is not None:
                pointsB.append(p)

        if len(pointsA) < 2 or len(pointsB) < 2:
            if not silent:
                print("Edge.IsCoplanar - Error: Could not evaluate sufficient points on one or both input edges. Returning None.")
            return None

        points = pointsA + pointsB

        # Remove geometrically duplicate points.
        unique_points = []

        for point in points:
            duplicate = False

            for existing in unique_points:
                dx = point[0] - existing[0]
                dy = point[1] - existing[1]
                dz = point[2] - existing[2]

                if math.sqrt(dx * dx + dy * dy + dz * dz) <= tolerance:
                    duplicate = True
                    break

            if not duplicate:
                unique_points.append(point)

        if len(unique_points) < 2:
            if not silent:
                print("Edge.IsCoplanar - Error: Could not determine sufficient distinct points from the input edges. Returning None.")
            return None

        # Find three non-collinear points to define a candidate plane.
        origin = None
        normal = None

        for i in range(len(unique_points) - 2):
            p0 = unique_points[i]

            for j in range(i + 1, len(unique_points) - 1):
                p1 = unique_points[j]
                v1 = _subtract(p1, p0)

                if _magnitude(v1) <= tolerance:
                    continue

                for k in range(j + 1, len(unique_points)):
                    p2 = unique_points[k]
                    v2 = _subtract(p2, p0)

                    n = _cross(v1, v2)
                    magnitude = _magnitude(n)

                    if magnitude > tolerance:
                        origin = p0
                        normal = [
                            n[0] / magnitude,
                            n[1] / magnitude,
                            n[2] / magnitude,
                        ]
                        break

                if normal is not None:
                    break

            if normal is not None:
                break

        # If all sampled points are collinear, both edges lie on the same line
        # and therefore necessarily share infinitely many possible planes.
        if normal is None:
            return True

        # Verify that every sampled point from both actual edge geometries lies
        # within tolerance of the candidate plane.
        for point in points:
            vector = _subtract(point, origin)
            distance = abs(_dot(vector, normal))

            if distance > tolerance:
                return False

        # For curved edges, additionally check sampled tangents. A tangent to a
        # planar curve must lie in the curve's plane.
        tangent_parameters = [
            0.0625,
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.75,
            0.875,
            0.9375,
        ]

        for edge in [edgeA, edgeB]:
            for u in tangent_parameters:
                try:
                    tangent = Edge.TangentAtParameter(
                        edge,
                        u=u,
                        mantissa=None,
                        tolerance=tolerance,
                        silent=True,
                    )
                except Exception:
                    tangent = None

                if not isinstance(tangent, (list, tuple)) or len(tangent) != 3:
                    continue

                try:
                    tangent = [
                        float(tangent[0]),
                        float(tangent[1]),
                        float(tangent[2]),
                    ]
                except Exception:
                    continue

                magnitude = _magnitude(tangent)

                if magnitude <= tolerance:
                    continue

                tangent = [
                    tangent[0] / magnitude,
                    tangent[1] / magnitude,
                    tangent[2] / magnitude,
                ]

                # A vector lying in the plane must be perpendicular to its normal.
                if abs(_dot(tangent, normal)) > tolerance:
                    return False

        return True

    @staticmethod
    def IsLinear(edge, tolerance: float = 0.0001, silent: bool = False) -> bool:
        """
        Returns True if the input edge is geometrically linear. Returns False otherwise.

        An edge is considered linear if its actual geometry follows a straight line
        within the specified tolerance. This method examines the geometry of the edge,
        not merely its start and end vertices. Therefore, a geometrically straight
        B-spline or other curve representation may also be considered linear.

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
        from topologicpy.Topology import Topology

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

        if tolerance <= 0:
            if not silent:
                print("Edge.IsLinear - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        return bool(Edge._IsLinear(edge, tolerance=tolerance))
    
    @staticmethod
    def IsParallel(edgeA, edgeB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns True if two geometrically linear edges are parallel.
        Returns False otherwise.

        This method is intended only for geometrically linear edges. If either
        input edge is curved, False is returned. The start-to-end chord of a curved
        edge is never used as a substitute for its actual geometry.

        Oppositely oriented linear edges are considered parallel.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            Retained for API compatibility. Geometric calculations are performed
            at full precision. Default is 6.
        tolerance : float , optional
            The desired tolerance applied to the sine of the angle between the
            edge directions. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        bool
            True if both edges are geometrically linear and parallel.
            False otherwise.

        """
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge") or not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.IsParallel - Error: One or both input parameters are not valid topologic edges. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.IsParallel - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.IsParallel - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.IsParallel - Warning: The input edgeA is curved. Parallelism is defined here only for geometrically linear edges. Returning False.")
            return False

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.IsParallel - Warning: The input edgeB is curved. Parallelism is defined here only for geometrically linear edges. Returning False.")
            return False

        a0 = Vertex.Coordinates(Edge.StartVertex(edgeA, silent=True), mantissa=None)
        a1 = Vertex.Coordinates(Edge.EndVertex(edgeA, silent=True), mantissa=None)
        b0 = Vertex.Coordinates(Edge.StartVertex(edgeB, silent=True), mantissa=None)
        b1 = Vertex.Coordinates(Edge.EndVertex(edgeB, silent=True), mantissa=None)

        ax = a1[0] - a0[0]
        ay = a1[1] - a0[1]
        az = a1[2] - a0[2]

        bx = b1[0] - b0[0]
        by = b1[1] - b0[1]
        bz = b1[2] - b0[2]

        lengthA = math.sqrt(ax * ax + ay * ay + az * az)
        lengthB = math.sqrt(bx * bx + by * by + bz * bz)

        if lengthA <= tolerance or lengthB <= tolerance:
            if not silent:
                print("Edge.IsParallel - Error: One or both input edges are degenerate. Returning None.")
            return None

        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx

        cross_length = math.sqrt(cx * cx + cy * cy + cz * cz)

        sine = cross_length / (lengthA * lengthB)

        return bool(sine <= tolerance)

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
    def TangentAtParameter(edge, u: float = 0.5, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """Returns the unit tangent vector to the actual edge curve at normalized parameter ``u``."""
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.TangentAtParameter - Error: The input edge is invalid. Returning None.")
            return None
        try:
            u = max(0.0, min(1.0, float(u)))
        except Exception:
            return None
        tangent = None
        try:
            if Core.HasAttribute("EdgeUtility", "TangentAtParameter"):
                tangent = Core.EdgeUtility.TangentAtParameter(edge, u)
        except Exception:
            tangent = None
        if tangent is None:
            # Frozen TopologicCore exposes exact point evaluation but no tangent.
            # Differentiate the actual curve numerically rather than using its chord.
            du = max(1.0e-6, min(1.0e-3, tolerance * 10.0))
            ua, ub = max(0.0, u-du), min(1.0, u+du)
            if ub <= ua:
                return None
            va = Edge.VertexByParameter(edge, ua, tolerance=tolerance, silent=True)
            vb = Edge.VertexByParameter(edge, ub, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(va, "Vertex") or not Topology.IsInstance(vb, "Vertex"):
                return None
            a = Vertex.Coordinates(va, mantissa=None); b = Vertex.Coordinates(vb, mantissa=None)
            tangent = [b[i]-a[i] for i in range(3)]
        try:
            values = [float(v) for v in tangent[:3]]
            mag = math.sqrt(sum(v*v for v in values))
            if mag <= tolerance:
                return None
            values = [v/mag for v in values]
            return values if mantissa is None else [round(v, mantissa) for v in values]
        except Exception:
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
        if not Edge._IsLinear(edge, tolerance=tolerance):
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

    @staticmethod
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
    def ParameterAtVertex(edge, vertex, mantissa: int = 6, silent: bool = False, tolerance: float = 0.0001) -> float:
        """Returns the normalized curve parameter of a vertex lying on the edge."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(edge, "Edge") or not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Edge.ParameterAtVertex - Error: One or more input parameters are invalid. Returning None.")
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
    def Spread(edgeA, edgeB, mantissa: int = 6, bracket: bool = False, tolerance: float = 0.0001, silent: bool = False) -> float:
        """
        Returns the rational-trigonometry spread between two geometrically linear edges.

        This method is intended only for geometrically linear edges. An arbitrary
        curved edge does not have a single global direction, so curved edges are
        not represented by their endpoint chords and will cause the method to
        return None.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge.
        edgeB : topologic_core.Edge
            The second input edge.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        bracket : bool , optional
            If set to True, the spread is invariant under reversal of either edge.
            Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        float
            The spread between the two input linear edges, or None if either edge
            is curved or the spread cannot be determined.

        """
        import math
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

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.Spread - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.Spread - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.Spread - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.Spread - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        if not Edge.IsLinear(edgeA, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Spread - Error: The input edgeA is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        if not Edge.IsLinear(edgeB, tolerance=tolerance, silent=True):
            if not silent:
                print("Edge.Spread - Error: The input edgeB is curved. This method only supports geometrically linear edges. Returning None.")
            return None

        directionA = Edge.Direction(
            edgeA,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        directionB = Edge.Direction(
            edgeB,
            mantissa=None,
            tolerance=tolerance,
            silent=True,
        )

        if directionA is None or directionB is None:
            if not silent:
                print("Edge.Spread - Error: Could not determine the directions of the input edges. Returning None.")
            return None

        spread = Vector.Spread(
            directionA,
            directionB,
            mantissa=mantissa,
            bracket=bracket,
        )

        if spread is None and not silent:
            print("Edge.Spread - Error: Could not determine the spread between the input edges. Returning None.")

        return spread

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
        Trims the first input edge at its intersection with the second input edge.

        The geometry of edgeA is preserved. The method determines the intersection
        between edgeA and edgeB, identifies valid internal intersection vertices on
        edgeA, and trims edgeA using its native curve parameters.

        If multiple valid intersections exist, the intersection closest to the
        start of edgeA is used when reverse is False. If reverse is True, the
        intersection closest to the end of edgeA is used.

        If reverse is False, the returned edge is oriented from the original start
        vertex of edgeA toward the selected intersection. If reverse is True, the
        returned edge is oriented from the original end vertex of edgeA toward the
        selected intersection.

        The second input edge may be linear or curved. The first input edge may also
        be linear or curved. No curved edge is reconstructed from its endpoints.

        If no valid internal intersection is found, the original edgeA is returned.

        Parameters
        ----------
        edgeA : topologic_core.Edge
            The first input edge. This edge will be trimmed by edgeB.
        edgeB : topologic_core.Edge
            The second input edge. This edge will be used to trim edgeA.
        reverse : bool , optional
            If set to False, the segment adjacent to the start vertex of edgeA is
            preserved and retains the orientation from the start vertex toward the
            intersection. If set to True, the segment adjacent to the end vertex is
            preserved and is oriented from the end vertex toward the intersection.
            Default is False.
        mantissa : int , optional
            The number of decimal places to use when comparing parameter values.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Edge
            The trimmed edge, the original edgeA if no valid internal intersection
            is found, or None if the operation fails.

        """
        import math
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(edgeA, "Edge"):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edgeA parameter is not a valid topologic edge. Returning None.")
            return None

        if not Topology.IsInstance(edgeB, "Edge"):
            if not silent:
                print("Edge.TrimByEdge - Error: The input edgeB parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("Edge.TrimByEdge - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Edge.TrimByEdge - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("Edge.TrimByEdge - Error: The input mantissa parameter is not a valid integer. Returning None.")
            return None

        if mantissa < 0:
            if not silent:
                print("Edge.TrimByEdge - Error: The input mantissa parameter must be zero or greater. Returning None.")
            return None

        # Compute the actual geometric/topological intersection.
        try:
            intersection = Topology.Intersect(
                edgeA,
                edgeB,
                tolerance=tolerance,
            )
        except Exception:
            intersection = None

        if intersection is None:
            return edgeA

        # Extract intersection vertices. This also handles overlapping results
        # whose boundary vertices can provide usable trim parameters.
        if Topology.IsInstance(intersection, "Vertex"):
            vertices = [intersection]
        else:
            try:
                vertices = Topology.Vertices(intersection)
            except Exception:
                vertices = []

        if not isinstance(vertices, list) or len(vertices) == 0:
            return edgeA

        # Convert valid internal intersection vertices to normalized parameters
        # on the actual geometry of edgeA.
        parameters = []

        for vertex in vertices:
            if not Topology.IsInstance(vertex, "Vertex"):
                continue

            u = Edge.ParameterAtVertex(
                edgeA,
                vertex,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )

            if u is None:
                continue

            try:
                u = float(u)
            except Exception:
                continue

            if not math.isfinite(u):
                continue

            # Intersections at existing endpoints do not trim the edge.
            if u <= tolerance or u >= 1.0 - tolerance:
                continue

            rounded_u = round(u, mantissa)

            if not any(
                round(existing, mantissa) == rounded_u
                for existing in parameters
            ):
                parameters.append(u)

        if len(parameters) == 0:
            return edgeA

        parameters.sort()

        if reverse:
            # Select the intersection nearest the original end and orient the
            # result FROM the original end TOWARD the intersection.
            u = parameters[-1]
            uA = 1.0
            uB = u
        else:
            # Select the intersection nearest the original start and orient the
            # result FROM the original start TOWARD the intersection.
            u = parameters[0]
            uA = 0.0
            uB = u

        result = Edge.TrimByParameters(
            edgeA,
            uA=uA,
            uB=uB,
            tolerance=tolerance,
            silent=True,
        )

        if not Topology.IsInstance(result, "Edge"):
            if not silent:
                print("Edge.TrimByEdge - Error: Could not trim edgeA at the detected intersection. Returning None.")
            return None

        return result

    @staticmethod
    def TrimByParameters(edge, uA: float = 0.0, uB: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the portion of the input edge between two normalized curve parameters.

        The input parameters uA and uB are normalized to the range 0 to 1, where
        0 represents the start of the edge and 1 represents the end of the edge.

        The geometry of the input edge is preserved. If the active backend supports
        native curve trimming, the underlying curve is trimmed directly. Linear
        edges may safely be reconstructed from their trimmed endpoints. A curved
        edge is never approximated or replaced by its endpoint chord; if the active
        backend cannot trim the curve exactly, None is returned.

        If uA is greater than uB, the returned edge has the reverse orientation.
        Supplying uA=0 and uB=1 returns the input edge unchanged. Supplying uA=1
        and uB=0 returns the reversed input edge.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        uA : float , optional
            The normalized parameter corresponding to the start of the returned
            edge. It must be in the range 0 to 1. Default is 0.0.
        uB : float , optional
            The normalized parameter corresponding to the end of the returned
            edge. It must be in the range 0 to 1. Default is 1.0.
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
                print("Edge.TrimByParameters - Error: The input edge parameter is not a valid topologic edge. Returning None.")
            return None

        try:
            tolerance = float(tolerance)
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

        # Preserve the original topology when no trimming is required.
        if uA == 0.0 and uB == 1.0:
            return edge

        if uA == 1.0 and uB == 0.0:
            return Edge.Reverse(
                edge,
                tolerance=tolerance,
                silent=silent,
            )

        # Prefer exact native curve trimming.
        try:
            if Core.HasAttribute("EdgeUtility", "Trim"):
                result = Core.EdgeUtility.Trim(edge, uA, uB)

                if Topology.IsInstance(result, "Edge"):
                    length = Edge.Length(
                        result,
                        mantissa=None,
                        tolerance=tolerance,
                        silent=True,
                    )

                    if length is not None and length > tolerance:
                        return result

                    if not silent:
                        print("Edge.TrimByParameters - Error: The trimmed edge is shorter than or equal to the input tolerance. Returning None.")
                    return None
        except Exception:
            pass

        # Endpoint reconstruction is exact only for geometrically linear edges.
        if Edge.IsLinear(edge, tolerance=tolerance, silent=True):
            vertexA = Edge.VertexByParameter(
                edge,
                u=uA,
                tolerance=tolerance,
                silent=True,
            )

            vertexB = Edge.VertexByParameter(
                edge,
                u=uB,
                tolerance=tolerance,
                silent=True,
            )

            if not Topology.IsInstance(vertexA, "Vertex") or not Topology.IsInstance(vertexB, "Vertex"):
                if not silent:
                    print("Edge.TrimByParameters - Error: Could not determine the trimmed edge vertices. Returning None.")
                return None

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

        if not silent:
            print("Edge.TrimByParameters - Error: The active backend cannot trim the curved input edge exactly. Returning None.")

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

        if Edge._IsLinear(edge, tolerance=tolerance):
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

        def arc_length(a, b, samples=96):
            if abs(b-a) <= 1.0e-15:
                return 0.0
            previous = Edge.VertexByParameter(edge, u=a, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(previous, "Vertex"):
                return None
            total = 0.0
            for i in range(1, samples+1):
                u = a + (b-a)*i/samples
                current = Edge.VertexByParameter(edge, u=u, tolerance=tolerance, silent=True)
                if not Topology.IsInstance(current, "Vertex"):
                    return None
                segment = Vertex.Distance(previous, current, mantissa=None, tolerance=tolerance, silent=True)
                if segment is None:
                    return None
                total += float(segment)
                previous = current
            return total

        def solve_between(a, b, requested):
            available = arc_length(a, b, samples=128)
            if available is None or requested > available + tolerance:
                return None
            low, high = 0.0, 1.0
            for _ in range(60):
                fraction = 0.5*(low+high)
                candidate = a + fraction*(b-a)
                length_now = arc_length(a, candidate, samples=64)
                if length_now is None:
                    return None
                if length_now < requested:
                    low = fraction
                else:
                    high = fraction
            return a + 0.5*(low+high)*(b-a)

        limit = 1.0 if sign > 0.0 else 0.0
        available = arc_length(u0, limit, samples=128)
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
        """Creates a vertex at normalized parameter ``u`` on the actual edge curve."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Edge.VertexByParameter - Error: The input edge is invalid. Returning None.")
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
        if not Topology.IsInstance(vertex, "Vertex") and not silent:
            print("Edge.VertexByParameter - Error: The active backend could not evaluate the edge curve. Returning None.")
        return vertex if Topology.IsInstance(vertex, "Vertex") else None

    @staticmethod
    def Vertices(edge, silent: bool = False) -> list:
        """
        Returns the topological vertices of the input edge in start-to-end order.

        An open edge returns its distinct start and end vertices. A closed edge has
        one topological seam vertex and therefore returns a one-item list.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of topological edge vertices.
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

        if Edge.IsClosed(edge, silent=True):
            if vertices:
                return [vertices[0]]
            start = Edge.StartVertex(edge, silent=True)
            return [start] if Topology.IsInstance(start, "Vertex") else []

        if len(vertices) >= 2:
            return vertices[:2]
        start = Edge.StartVertex(edge, silent=True)
        end = Edge.EndVertex(edge, silent=True)
        return [vertex for vertex in [start, end] if Topology.IsInstance(vertex, "Vertex")]
