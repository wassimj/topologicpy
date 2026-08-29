from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from .topology import (
    Topology,
    _is_null_shape,
    _merge_backend_dictionaries,
    BRepAlgoAPI_Common as _BRepAlgoAPI_Common,
)
from .vertex import Vertex
from .occ_utils import make_occ_edge
from .helpers import same_vertex, unique_by_uuid


def _curve_data(edge):
    """Return the OCCT curve and its stored parameter bounds for an Edge."""
    if not isinstance(edge, Edge) or _is_null_shape(getattr(edge, "shape", None)):
        return None
    try:
        from OCC.Core.BRep import BRep_Tool

        curve, first, last = BRep_Tool.Curve(edge.shape)
        if curve is None:
            return None
        return curve, float(first), float(last)
    except Exception:
        return None


def _oriented_parameter_bounds(edge):
    """Return curve parameters corresponding to the topological start and end."""
    data = _curve_data(edge)
    if data is None:
        return None
    curve, first, last = data

    # OCCT stores the geometric parameter interval independently of topological
    # edge orientation. Use the actual TopAbs orientation first. This is essential
    # for closed periodic edges, whose first and last geometric points coincide.
    try:
        from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
        orientation = edge.shape.Orientation()
        if orientation == TopAbs_REVERSED:
            return curve, last, first
        if orientation == TopAbs_FORWARD:
            return curve, first, last
    except Exception:
        pass

    # Conservative fallback for unusual INTERNAL/EXTERNAL orientations.
    start = edge.start
    if not isinstance(start, Vertex):
        return None
    try:
        p_first = curve.Value(first)
        p_last = curve.Value(last)
        d_first2 = (
            (float(start.x) - float(p_first.X())) ** 2
            + (float(start.y) - float(p_first.Y())) ** 2
            + (float(start.z) - float(p_first.Z())) ** 2
        )
        d_last2 = (
            (float(start.x) - float(p_last.X())) ** 2
            + (float(start.y) - float(p_last.Y())) ** 2
            + (float(start.z) - float(p_last.Z())) ** 2
        )
        return (curve, first, last) if d_first2 <= d_last2 else (curve, last, first)
    except Exception:
        return None


def _raw_parameter(edge, normalized_parameter):
    """Map a normalized topological Edge parameter to the OCCT curve parameter."""
    data = _oriented_parameter_bounds(edge)
    if data is None:
        return None
    curve, start_parameter, end_parameter = data
    try:
        u = float(normalized_parameter)
    except Exception:
        return None
    return curve, start_parameter + u * (end_parameter - start_parameter)


def _point_at_raw_parameter(curve, parameter):
    """Return a backend Vertex evaluated on an OCCT curve."""
    try:
        point = curve.Value(float(parameter))
        return Vertex.ByCoordinates(float(point.X()), float(point.Y()), float(point.Z()))
    except Exception:
        return None


def _tangent_at_raw_parameter(edge, curve, parameter):
    """Return the unit tangent in the topological orientation of an Edge."""
    bounds = _oriented_parameter_bounds(edge)
    if bounds is None:
        return None
    _, start_parameter, end_parameter = bounds
    orientation = 1.0 if end_parameter >= start_parameter else -1.0
    try:
        from OCC.Core.gp import gp_Pnt, gp_Vec

        point = gp_Pnt()
        vector = gp_Vec()
        curve.D1(float(parameter), point, vector)
        x = float(vector.X()) * orientation
        y = float(vector.Y()) * orientation
        z = float(vector.Z()) * orientation
        magnitude = math.sqrt(x * x + y * y + z * z)
        if magnitude <= 0.0:
            return None
        return [x / magnitude, y / magnitude, z / magnitude]
    except Exception:
        return None


def _wrap_shape_like(source, shape):
    """Wrap an OCCT edge while preserving backend metadata from the source."""
    if not isinstance(source, Edge) or _is_null_shape(shape):
        return None
    return Edge.ByOcctShape(
        shape,
        dictionary=getattr(source, "dictionary", None),
        contents=getattr(source, "contents", None),
        contexts=getattr(source, "contexts", None),
        apertures=getattr(source, "apertures", None),
    )

def _same_edge_topology(edge_a, edge_b):
    """Return True when two backend Edge wrappers reference the same OCCT TShape."""
    if not isinstance(edge_a, Edge) or not isinstance(edge_b, Edge):
        return False
    if _is_null_shape(getattr(edge_a, "shape", None)) or _is_null_shape(getattr(edge_b, "shape", None)):
        return False
    try:
        return bool(edge_a.shape.IsSame(edge_b.shape))
    except Exception:
        return edge_a is edge_b


def _normal_at_raw_parameter(edge, curve, parameter):
    """Return the unit principal normal at an OCCT curve parameter, or None at zero curvature."""
    if not isinstance(edge, Edge):
        return None
    try:
        from OCC.Core.gp import gp_Pnt, gp_Vec
        point = gp_Pnt()
        d1 = gp_Vec()
        d2 = gp_Vec()
        curve.D2(float(parameter), point, d1, d2)
        tx, ty, tz = float(d1.X()), float(d1.Y()), float(d1.Z())
        tmag = math.sqrt(tx*tx + ty*ty + tz*tz)
        if tmag <= 1.0e-15:
            return None
        tx, ty, tz = tx/tmag, ty/tmag, tz/tmag
        ax, ay, az = float(d2.X()), float(d2.Y()), float(d2.Z())
        projection = ax*tx + ay*ty + az*tz
        nx, ny, nz = ax - projection*tx, ay - projection*ty, az - projection*tz
        nmag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nmag <= 1.0e-15:
            return None
        return [nx/nmag, ny/nmag, nz/nmag]
    except Exception:
        return None


@dataclass(eq=False, init=False)
class Edge(Topology):
    """
    PythonOCC backend Edge wrapper with lazy endpoint materialization.

    The OCCT edge shape is wrapped immediately, while start and end Vertex
    wrappers are created only when first requested. This keeps subtopology
    extraction shallow and avoids eagerly constructing two Vertex wrappers for
    every returned Edge.
    """

    _start: Optional[Vertex] = None
    _end: Optional[Vertex] = None
    _start_loaded: bool = False
    _end_loaded: bool = False

    def __init__(
        self,
        shape=None,
        start: Optional[Vertex] = None,
        end: Optional[Vertex] = None,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
        _uuid=None,
    ):
        """Initialize a backend Edge wrapper."""
        kwargs = {
            "shape": shape,
            "dictionary": dictionary,
            "contents": list(contents) if contents else [],
            "contexts": list(contexts) if contexts else [],
            "apertures": list(apertures) if apertures else [],
        }
        if _uuid is not None:
            kwargs["_uuid"] = _uuid
        super().__init__(**kwargs)
        self._start = start if isinstance(start, Vertex) else None
        self._end = end if isinstance(end, Vertex) else None
        self._start_loaded = isinstance(start, Vertex)
        self._end_loaded = isinstance(end, Vertex)

    def _load_start(self):
        """Materialize and cache the oriented start Vertex."""
        if self._start_loaded:
            return self._start
        self._start_loaded = True
        if _is_null_shape(getattr(self, "shape", None)):
            self._start = None
            return None
        try:
            from OCC.Core.TopExp import topexp

            occ_vertex = topexp.FirstVertex(self.shape, True)
            self._start = None if _is_null_shape(occ_vertex) else Vertex.ByOcctShape(occ_vertex)
        except Exception:
            self._start = None
        return self._start

    def _load_end(self):
        """Materialize and cache the oriented end Vertex."""
        if self._end_loaded:
            return self._end
        self._end_loaded = True
        if _is_null_shape(getattr(self, "shape", None)):
            self._end = None
            return None
        try:
            from OCC.Core.TopExp import topexp

            occ_vertex = topexp.LastVertex(self.shape, True)
            self._end = None if _is_null_shape(occ_vertex) else Vertex.ByOcctShape(occ_vertex)
        except Exception:
            self._end = None
        return self._end

    @property
    def start(self) -> Optional[Vertex]:
        """Return the oriented start Vertex."""
        return self._load_start()

    @start.setter
    def start(self, value):
        """Set the cached oriented start Vertex."""
        self._start = value if isinstance(value, Vertex) else None
        self._start_loaded = True

    @property
    def end(self) -> Optional[Vertex]:
        """Return the oriented end Vertex."""
        return self._load_end()

    @end.setter
    def end(self, value):
        """Set the cached oriented end Vertex."""
        self._end = value if isinstance(value, Vertex) else None
        self._end_loaded = True


    @staticmethod
    def ByStartVertexEndVertex(
        startVertex,
        endVertex,
        tolerance: float = 0.0001
    ):
        """
        Creates a straight Edge between two backend Vertices.

        Parameters
        ----------
        startVertex : Vertex
            The start Vertex.
        endVertex : Vertex
            The end Vertex.
        tolerance : float , optional
            The minimum permitted distance between the two Vertices.
            Default is 0.0001.

        Returns
        -------
        Edge
            The created Edge, or None if construction fails.

        """
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if not math.isfinite(tolerance):
            return None

        if same_vertex(
            startVertex,
            endVertex,
            tolerance=tolerance,
        ):
            return None

        try:
            shape = make_occ_edge(
                startVertex,
                endVertex,
            )
        except Exception:
            return None

        if _is_null_shape(shape):
            return None

        return Edge(
            shape=shape,
            start=startVertex,
            end=endVertex,
        )

    @staticmethod
    def ByVertices(vertices):
        """Create a straight Edge between the first and last input Vertex."""
        if vertices is None or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    @staticmethod
    def ByStartVertexEndVertexTolerance(
        startVertex,
        endVertex,
        tolerance: float = 0.0001
    ):
        """
        Creates a straight Edge between two Vertices using the input tolerance.

        This method delegates to ByStartVertexEndVertex and is retained as a
        backend compatibility entry point.

        Parameters
        ----------
        startVertex : Vertex
            The start Vertex.
        endVertex : Vertex
            The end Vertex.
        tolerance : float , optional
            The minimum permitted distance between the two Vertices.
            Default is 0.0001.

        Returns
        -------
        Edge
            The created Edge, or None if construction fails.

        """
        return Edge.ByStartVertexEndVertex(
            startVertex,
            endVertex,
            tolerance=tolerance,
        )
    @staticmethod
    def ByNurbsParameters(controlPoints, weights=None, knots=None, isRational: bool = False, isPeriodic: bool = False, degree: int = 3):
        """Create an Edge from expanded NURBS/B-spline parameters."""
        vertices = [v for v in (controlPoints or []) if isinstance(v, Vertex)]
        if len(vertices) < 2:
            return None
        try:
            degree = int(degree)
        except Exception:
            return None
        if degree < 1 or degree >= len(vertices):
            return None
        if weights is None:
            weights = [1.0] * len(vertices)
        try:
            weights = [float(v) for v in weights]
        except Exception:
            return None
        if len(weights) != len(vertices) or any(not math.isfinite(v) or v <= 0.0 for v in weights):
            return None
        if not bool(isRational):
            weights = [1.0] * len(vertices)
        if knots is None:
            if bool(isPeriodic):
                knots = [float(i) for i in range(len(vertices) + 1)]
            else:
                interior = len(vertices) - degree - 1
                knots = [0.0] * (degree + 1)
                if interior > 0:
                    knots += [float(i) / float(interior + 1) for i in range(1, interior + 1)]
                knots += [1.0] * (degree + 1)
        return EdgeUtility.ByNurbsCurve(
            vertices,
            knots,
            weights,
            degree,
            bool(isPeriodic),
            bool(isRational),
        )

    @staticmethod
    def ByCurve(points, degree: int = 3, periodic: bool = False, tolerance: float = 0.0001):
        """Create one non-rational B-spline Edge using the input vertices as control points."""
        vertices = [p for p in (points or []) if isinstance(p, Vertex)]
        if len(vertices) < 2:
            return None
        return Edge.ByNurbsParameters(
            vertices,
            weights=[1.0] * len(vertices),
            knots=None,
            isRational=False,
            isPeriodic=bool(periodic),
            degree=degree,
        )

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        """Wrap an existing OCCT Edge without eagerly materializing endpoints."""
        if _is_null_shape(shape):
            return None
        try:
            from OCC.Core.TopoDS import topods

            occ_edge = topods.Edge(shape)
            if _is_null_shape(occ_edge):
                return None
        except Exception:
            return None
        return Edge(
            shape=occ_edge,
            dictionary=dictionary,
            contents=contents,
            contexts=contexts,
            apertures=apertures,
        )

    def StartVertex(self):
        """Return the oriented start Vertex."""
        return self.start

    def EndVertex(self):
        """Return the oriented end Vertex."""
        return self.end

    def Vertices(self, hostTopology=None, vertices=None):
        """Return or append the unique topological Vertex objects of this Edge."""
        start = self.start
        end = self.end
        result = []
        if isinstance(start, Vertex):
            result.append(start)
        if isinstance(end, Vertex):
            closed = False
            try:
                closed = bool(EdgeUtility.IsClosed(self))
            except Exception:
                closed = same_vertex(start, end) if isinstance(start, Vertex) else False
            if not closed:
                result.append(end)
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        """Return or append this Edge."""
        result = [self]
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def AdjacentEdges(self, hostTopology=None, output=None):
        """Return Edges in hostTopology that share an endpoint with this Edge."""
        result = []
        if hostTopology is not None:
            candidates = Topology.Edges(hostTopology) or []
            for other in candidates:
                if not isinstance(other, Edge) or _same_edge_topology(other, self):
                    continue
                if (
                    same_vertex(self.start, other.start)
                    or same_vertex(self.start, other.end)
                    or same_vertex(self.end, other.start)
                    or same_vertex(self.end, other.end)
                ):
                    result.append(other)
            result = unique_by_uuid(result)
        if output is not None:
            output.extend(result)
            return 0
        return result

    @staticmethod
    def Reverse(edge, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns an Edge with the reverse topological orientation of the input Edge.

        The underlying OCCT curve geometry is preserved exactly. No reconstruction
        from the Edge endpoints is performed.

        Parameters
        ----------
        edge : Edge
            The input Edge.
        tolerance : float , optional
            The desired tolerance. This parameter is accepted for API compatibility.
            Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        Edge
            The reversed Edge, or None if the Edge cannot be reversed while
            preserving its geometry.

        """
        if not isinstance(edge, Edge):
            return None

        if _is_null_shape(getattr(edge, "shape", None)):
            return None

        try:
            shape = edge.shape.Reversed()

            if _is_null_shape(shape):
                return None

            result = _wrap_shape_like(edge, shape)

            if isinstance(result, Edge):
                return result

        except Exception:
            pass

        return None

    def Direction(self, mantissa: int = 6):
        """Return the unit chord direction from the oriented start to end Vertex."""
        if not isinstance(self.start, Vertex) or not isinstance(self.end, Vertex):
            return None
        dx = float(self.end.x) - float(self.start.x)
        dy = float(self.end.y) - float(self.start.y)
        dz = float(self.end.z) - float(self.start.z)
        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
        if magnitude <= 0.0:
            return None
        result = [dx / magnitude, dy / magnitude, dz / magnitude]
        if mantissa is None:
            return result
        return [round(value, int(mantissa)) for value in result]

    def VertexByParameter(self, u: float = 0.0):
        """Return a Vertex at normalized parameter u along the OCCT curve."""
        return EdgeUtility.PointAtParameter(self, u)

    def ParameterAtVertex(self, vertex, mantissa: int = 6, tolerance: float = 0.0001):
        """Return the normalized parameter of a Vertex lying on this Edge."""
        try:
            value = EdgeUtility.ParameterAtPoint(self, vertex, tolerance=tolerance)
        except Exception:
            return None
        return None if value is None else round(float(value), mantissa)

    def Length(self):
        """Return the exact OCCT curve length of this Edge."""
        return EdgeUtility.Length(self)

    def Intersect(self, otherTopology, transferDictionary: bool = False):
        """Return the exact OCCT intersection with another topology."""
        result = Topology._binary_boolean(
            self,
            otherTopology,
            _BRepAlgoAPI_Common,
            transferDictionary,
        )
        if result is not None or not isinstance(otherTopology, Edge):
            return result

        # BRepAlgoAPI_Common may not materialize a zero-dimensional result for
        # transversal Edge intersections. Recover such a point from exact OCCT
        # shape-distance extrema instead of intersecting endpoint chords.
        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

            extrema = BRepExtrema_DistShapeShape(self.shape, otherTopology.shape)
            extrema.Perform()
            if not extrema.IsDone() or extrema.NbSolution() < 1:
                return None
            if float(extrema.Value()) > 1.0e-7:
                return None
            p1 = extrema.PointOnShape1(1)
            p2 = extrema.PointOnShape2(1)
            vertex = Vertex.ByCoordinates(
                0.5 * (float(p1.X()) + float(p2.X())),
                0.5 * (float(p1.Y()) + float(p2.Y())),
                0.5 * (float(p1.Z()) + float(p2.Z())),
            )
            if vertex is None:
                return None
            if transferDictionary:
                vertex.dictionary = _merge_backend_dictionaries(
                    Topology.GetDictionary(self),
                    Topology.GetDictionary(otherTopology),
                )
            return vertex
        except Exception:
            return None


class EdgeUtility:
    """OCCT-native utility operations for backend Edge objects."""


    @staticmethod
    def Arc(
        radius: float = 0.5,
        fromAngle: float = 0.0,
        toAngle: float = 180.0,
        tolerance: float = 0.0001
    ):
        """
        Creates an exact open circular arc centred at the global origin in the XY plane.

        The arc is created from the native OCCT circular-curve geometry. Angles are
        measured in degrees counter-clockwise from the positive X-axis when viewed
        from the positive Z-axis.

        This backend method creates only the canonical geometry. Placement and
        orientation are handled by the public `Edge.Arc` method.

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

        Returns
        -------
        Edge
            The created open circular arc, or None if the arc cannot be created.

        """
        try:
            radius = abs(float(radius))
            fromAngle = float(fromAngle)
            toAngle = float(toAngle)
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if not all(math.isfinite(value) for value in [radius, fromAngle, toAngle, tolerance]):
            return None

        if tolerance <= 0.0 or radius <= tolerance:
            return None

        while toAngle < fromAngle:
            toAngle += 360.0

        sweep = toAngle - fromAngle

        if sweep <= 1.0e-12 or sweep >= 360.0 - 1.0e-12:
            return None

        chord_length = 2.0 * radius * abs(math.sin(math.radians(sweep) * 0.5))
        if chord_length <= tolerance:
            return None

        try:
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
            from OCC.Core.Geom import Geom_Circle
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            axis = gp_Ax2(
                gp_Pnt(0.0, 0.0, 0.0),
                gp_Dir(0.0, 0.0, 1.0),
                gp_Dir(1.0, 0.0, 0.0),
            )

            circle = Geom_Circle(axis, radius)

            start_parameter = math.radians(fromAngle)
            end_parameter = math.radians(toAngle)

            maker = BRepBuilderAPI_MakeEdge(
                circle,
                start_parameter,
                end_parameter,
            )

            if not maker.IsDone():
                return None

            shape = maker.Edge()

            if _is_null_shape(shape):
                return None

            return Edge.ByOcctShape(shape)

        except Exception:
            return None

    @staticmethod
    def ByCircle(
        centerPoint,
        radius,
        xAxisX,
        xAxisY,
        xAxisZ,
        normalX,
        normalY,
        normalZ
    ):
        """
        Creates a single closed circular Edge.

        The circle is created using native OCCT circular geometry. The input
        centerPoint defines the centre of the circle, the X-axis vector defines
        the zero-parameter direction of the circle, and the normal vector defines
        the normal to the plane containing the circle.

        This method mirrors the signature and behaviour of
        `topologic_core.EdgeUtility.ByCircle`.

        Parameters
        ----------
        centerPoint : Vertex
            The centre vertex of the circle.
        radius : float
            The radius of the circle.
        xAxisX : float
            The X component of the circle's local X-axis.
        xAxisY : float
            The Y component of the circle's local X-axis.
        xAxisZ : float
            The Z component of the circle's local X-axis.
        normalX : float
            The X component of the circle plane normal.
        normalY : float
            The Y component of the circle plane normal.
        normalZ : float
            The Z component of the circle plane normal.

        Returns
        -------
        Edge
            The created closed circular Edge, or None if the circle cannot be
            created.

        """
        if not isinstance(centerPoint, Vertex):
            return None

        try:
            radius = abs(float(radius))
            xAxisX = float(xAxisX)
            xAxisY = float(xAxisY)
            xAxisZ = float(xAxisZ)
            normalX = float(normalX)
            normalY = float(normalY)
            normalZ = float(normalZ)
        except Exception:
            return None

        values = [
            radius,
            xAxisX, xAxisY, xAxisZ,
            normalX, normalY, normalZ,
        ]

        if not all(math.isfinite(value) for value in values):
            return None

        if radius <= 0.0:
            return None

        x_magnitude = math.sqrt(
            xAxisX * xAxisX +
            xAxisY * xAxisY +
            xAxisZ * xAxisZ
        )

        normal_magnitude = math.sqrt(
            normalX * normalX +
            normalY * normalY +
            normalZ * normalZ
        )

        if x_magnitude <= 0.0 or normal_magnitude <= 0.0:
            return None

        # The local X-axis must not be parallel to the circle normal.
        cx = xAxisY * normalZ - xAxisZ * normalY
        cy = xAxisZ * normalX - xAxisX * normalZ
        cz = xAxisX * normalY - xAxisY * normalX

        cross_magnitude = math.sqrt(cx * cx + cy * cy + cz * cz)

        if cross_magnitude <= 1.0e-12:
            return None

        try:
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
            from OCC.Core.Geom import Geom_Circle
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            axis = gp_Ax2(
                gp_Pnt(
                    float(centerPoint.x),
                    float(centerPoint.y),
                    float(centerPoint.z),
                ),
                gp_Dir(
                    normalX,
                    normalY,
                    normalZ,
                ),
                gp_Dir(
                    xAxisX,
                    xAxisY,
                    xAxisZ,
                ),
            )

            curve = Geom_Circle(axis, radius)

            maker = BRepBuilderAPI_MakeEdge(curve)

            if not maker.IsDone():
                return None

            shape = maker.Edge()

            if _is_null_shape(shape):
                return None

            return Edge.ByOcctShape(shape)

        except Exception:
            return None
    @staticmethod
    def ByNurbsCurve(controlPoints, knots, weights, degree: int = 3, isPeriodic: bool = False, isRational: bool = False):
        """Create one exact OCCT B-spline/NURBS Edge from an expanded knot vector."""
        vertices = [v for v in (controlPoints or []) if isinstance(v, Vertex)]
        if len(vertices) < 2:
            return None
        try:
            degree = int(degree)
            expanded_knots = [float(v) for v in knots]
            weights = [float(v) for v in weights]
        except Exception:
            return None
        if degree < 1 or degree >= len(vertices):
            return None
        if len(weights) != len(vertices) or any(not math.isfinite(v) or v <= 0.0 for v in weights):
            return None
        if not bool(isRational):
            weights = [1.0] * len(vertices)
        if any(not math.isfinite(v) for v in expanded_knots):
            return None
        if any(expanded_knots[i] > expanded_knots[i+1] for i in range(len(expanded_knots)-1)):
            return None

        unique_knots = []
        multiplicities = []
        for value in expanded_knots:
            if unique_knots and value == unique_knots[-1]:
                multiplicities[-1] += 1
            else:
                unique_knots.append(value)
                multiplicities.append(1)
        if len(unique_knots) < 2:
            return None
        if bool(isPeriodic):
            if multiplicities[0] != multiplicities[-1]:
                return None
            if any(m < 1 or m > degree for m in multiplicities):
                return None
            if sum(multiplicities) - multiplicities[0] != len(vertices):
                return None
        else:
            if sum(multiplicities) != len(vertices) + degree + 1:
                return None
            if any(m < 1 or m > degree for m in multiplicities[1:-1]):
                return None
            if not (1 <= multiplicities[0] <= degree + 1 and 1 <= multiplicities[-1] <= degree + 1):
                return None

        try:
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TColgp import TColgp_Array1OfPnt
            from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCC.Core.Geom import Geom_BSplineCurve
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            poles = TColgp_Array1OfPnt(1, len(vertices))
            weight_array = TColStd_Array1OfReal(1, len(vertices))
            for i, (vertex, weight) in enumerate(zip(vertices, weights), start=1):
                poles.SetValue(i, gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)))
                weight_array.SetValue(i, float(weight))

            knot_array = TColStd_Array1OfReal(1, len(unique_knots))
            mult_array = TColStd_Array1OfInteger(1, len(unique_knots))
            for i, (knot, mult) in enumerate(zip(unique_knots, multiplicities), start=1):
                knot_array.SetValue(i, float(knot))
                mult_array.SetValue(i, int(mult))

            curve = Geom_BSplineCurve(
                poles,
                weight_array,
                knot_array,
                mult_array,
                degree,
                bool(isPeriodic),
                True,
            )
            maker = BRepBuilderAPI_MakeEdge(curve)
            if not maker.IsDone():
                return None
            shape = maker.Edge()
            return None if _is_null_shape(shape) else Edge.ByOcctShape(shape)
        except Exception:
            return None

    @staticmethod
    def Connection(edgeA, edgeB, tolerance: float = 0.0001):
        """
        Returns the shortest straight Edge connecting two input Edges.

        The closest points are computed from the complete OCCT edge geometries,
        not merely from their endpoint vertices.

        Parameters
        ----------
        edgeA : Edge
            The first input Edge.
        edgeB : Edge
            The second input Edge.
        tolerance : float , optional
            The desired tolerance. If the minimum distance is less than or equal
            to this value, None is returned because no non-degenerate connecting
            Edge can be created. Default is 0.0001.

        Returns
        -------
        Edge
            The shortest straight connecting Edge, or None if the input Edges
            intersect, touch, overlap, or the operation fails.

        """
        if not isinstance(edgeA, Edge) or not isinstance(edgeB, Edge):
            return None

        if _is_null_shape(getattr(edgeA, "shape", None)):
            return None

        if _is_null_shape(getattr(edgeB, "shape", None)):
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        try:
            from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

            extrema = BRepExtrema_DistShapeShape(
                edgeA.shape,
                edgeB.shape,
            )

            extrema.Perform()

            if not extrema.IsDone() or extrema.NbSolution() < 1:
                return None

            distance = float(extrema.Value())

            if not math.isfinite(distance) or distance <= tolerance:
                return None

            pointA = extrema.PointOnShape1(1)
            pointB = extrema.PointOnShape2(1)

            vertexA = Vertex.ByCoordinates(
                float(pointA.X()),
                float(pointA.Y()),
                float(pointA.Z()),
            )

            vertexB = Vertex.ByCoordinates(
                float(pointB.X()),
                float(pointB.Y()),
                float(pointB.Z()),
            )

            if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
                return None

            return Edge.ByStartVertexEndVertex(
                vertexA,
                vertexB,
                tolerance=tolerance,
            )

        except Exception:
            return None
    
    @staticmethod
    def IsClosed(edge, tolerance: float = 0.0001):
        """
        Returns True if the input Edge is topologically closed. Returns False otherwise.

        Parameters
        ----------
        edge : Edge
            The input Edge.
        tolerance : float , optional
            The desired tolerance. This parameter is accepted for backend API
            compatibility. Native OCCT topological closure is evaluated exactly.
            Default is 0.0001.

        Returns
        -------
        bool
            True if the input Edge is topologically closed. False otherwise.

        """
        if not isinstance(edge, Edge):
            return False

        if _is_null_shape(getattr(edge, "shape", None)):
            return False

        try:
            from OCC.Core.BRep import BRep_Tool
            return bool(BRep_Tool.IsClosed(edge.shape))
        except Exception:
            return False
    @staticmethod
    def IsLinear(edge, tolerance: float = 0.0001):
        """Return True only when the actual OCCT edge geometry is geometrically linear."""
        if not isinstance(edge, Edge) or _is_null_shape(getattr(edge, "shape", None)):
            return False
        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
            from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_BSplineCurve, GeomAbs_BezierCurve
            adaptor = BRepAdaptor_Curve(edge.shape)
            curve_type = adaptor.GetType()
            if curve_type == GeomAbs_Line:
                return True
            if curve_type == GeomAbs_BSplineCurve:
                curve = adaptor.BSpline()
            elif curve_type == GeomAbs_BezierCurve:
                curve = adaptor.Bezier()
            else:
                return False
            count = int(curve.NbPoles())
            if count < 2:
                return False
            first = curve.Pole(1)
            last = curve.Pole(count)
            ax, ay, az = float(first.X()), float(first.Y()), float(first.Z())
            dx = float(last.X()) - ax
            dy = float(last.Y()) - ay
            dz = float(last.Z()) - az
            chord = math.sqrt(dx*dx + dy*dy + dz*dz)
            if chord <= tolerance:
                return False
            for i in range(2, count):
                p = curve.Pole(i)
                px, py, pz = float(p.X())-ax, float(p.Y())-ay, float(p.Z())-az
                cx = py*dz - pz*dy
                cy = pz*dx - px*dz
                cz = px*dy - py*dx
                if math.sqrt(cx*cx + cy*cy + cz*cz) / chord > tolerance:
                    return False
            # Collinear poles are not sufficient if the curve doubles back along
            # the same line. Require actual curve length to equal the endpoint chord.
            curve_length = EdgeUtility.Length(edge, tolerance=tolerance)
            return bool(curve_length is not None and abs(float(curve_length) - chord) <= tolerance)
        except Exception:
            return False
    
    @staticmethod
    def Length(edge, tolerance: float = 0.0001):
        """Return the exact geometric length of an Edge."""
        if not isinstance(edge, Edge) or _is_null_shape(getattr(edge, "shape", None)):
            return None
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            properties = GProp_GProps()
            brepgprop.LinearProperties(edge.shape, properties)
            value = float(properties.Mass())
            return value if math.isfinite(value) else None
        except Exception:
            return None

    @staticmethod
    def PointAtParameter(edge, parameter):
        """Return a Vertex at normalized parameter [0, 1] on the OCCT curve."""
        if not isinstance(edge, Edge):
            return None
        try:
            u = float(parameter)
        except Exception:
            return None
        if u == 0.0:
            return edge.start
        if u == 1.0:
            return edge.end
        raw = _raw_parameter(edge, u)
        if raw is None:
            return None
        curve, parameter_value = raw
        return _point_at_raw_parameter(curve, parameter_value)

    @staticmethod
    def ParameterAtPoint(edge, vertex, tolerance: float = 0.0001):
        """Return normalized parameter of a Vertex on the trimmed OCCT Edge."""
        if not isinstance(edge, Edge) or not isinstance(vertex, Vertex):
            return None
        bounds = _oriented_parameter_bounds(edge)
        if bounds is None:
            return None
        curve, start_parameter, end_parameter = bounds
        denominator = end_parameter - start_parameter
        if abs(denominator) <= 1.0e-15:
            return 0.0 if same_vertex(edge.start, vertex, tolerance=tolerance) else None
        try:
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

            point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            lower = min(start_parameter, end_parameter)
            upper = max(start_parameter, end_parameter)
            projection = GeomAPI_ProjectPointOnCurve(point, curve, lower, upper)
            if projection.NbPoints() < 1:
                return None
            if float(projection.LowerDistance()) > abs(float(tolerance)):
                return None
            raw_parameter = float(projection.LowerDistanceParameter())
            normalized = (raw_parameter - start_parameter) / denominator
            eps = max(abs(float(tolerance)), 1.0e-12)
            if normalized < -eps or normalized > 1.0 + eps:
                return None
            if abs(normalized) <= eps:
                normalized = 0.0
            elif abs(normalized - 1.0) <= eps:
                normalized = 1.0
            return max(0.0, min(1.0, normalized))
        except Exception:
            return None

    @staticmethod
    def PointAtDistance(edge, distance: float = 0.0, origin=None, tolerance: float = 0.0001):
        """Return a Vertex at signed curvilinear distance from an origin on the Edge.

        Closed edges wrap around their periodic path. Open curved edges are never
        extrapolated beyond their finite domain; open linear edges may be extended.
        """
        if not isinstance(edge, Edge):
            return None
        if not isinstance(origin, Vertex):
            origin = edge.start
        if not isinstance(origin, Vertex):
            return None
        try:
            requested_distance = float(distance)
            tol = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return None
        if abs(requested_distance) <= tol:
            return origin

        normalized_origin = EdgeUtility.ParameterAtPoint(edge, origin, tolerance=tol)
        if normalized_origin is None:
            return None
        bounds = _oriented_parameter_bounds(edge)
        if bounds is None:
            return None
        curve, start_parameter, end_parameter = bounds
        denominator = end_parameter - start_parameter
        if abs(denominator) <= 1.0e-15:
            return None
        raw_origin = start_parameter + normalized_origin * denominator
        parameter_orientation = 1.0 if denominator >= 0.0 else -1.0
        closed = EdgeUtility.IsClosed(edge, tolerance=tol)

        # On a closed edge reduce arbitrarily large travel distances to one period.
        effective_distance = requested_distance
        if closed:
            total_length = EdgeUtility.Length(edge, tolerance=tol)
            if total_length is None or total_length <= tol:
                return None
            effective_distance = math.fmod(requested_distance, total_length)
            if abs(effective_distance) <= tol:
                return origin

        try:
            from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
            from OCC.Core.GCPnts import GCPnts_AbscissaPoint
            adaptor = GeomAdaptor_Curve(curve)
            solver = GCPnts_AbscissaPoint(
                tol,
                adaptor,
                effective_distance * parameter_orientation,
                raw_origin,
            )
            if solver.IsDone():
                parameter_value = float(solver.Parameter())
                normalized = (parameter_value - start_parameter) / denominator
                if closed:
                    normalized = normalized % 1.0
                    parameter_value = start_parameter + normalized * denominator
                    return _point_at_raw_parameter(curve, parameter_value)
                if -tol <= normalized <= 1.0 + tol:
                    return _point_at_raw_parameter(curve, parameter_value)
        except Exception:
            pass

        if closed:
            return None
        # Extrapolation beyond a finite edge is exact only for linear geometry.
        if not EdgeUtility.IsLinear(edge, tolerance=tol):
            return None
        tangent = _tangent_at_raw_parameter(edge, curve, raw_origin)
        if tangent is None:
            return None
        return Vertex.ByCoordinates(
            float(origin.x) + tangent[0] * requested_distance,
            float(origin.y) + tangent[1] * requested_distance,
            float(origin.z) + tangent[2] * requested_distance,
        )


    @staticmethod
    def TangentAtParameter(edge, parameter: float = 0.5):
        """Return the oriented unit tangent vector at normalized parameter u."""
        if not isinstance(edge, Edge):
            return None
        raw = _raw_parameter(edge, parameter)
        if raw is None:
            return None
        curve, parameter_value = raw
        return _tangent_at_raw_parameter(edge, curve, parameter_value)

    @staticmethod
    def Angle(edgeA, edgeB):
        """Return the angle in degrees between two geometrically linear Edges."""
        if not EdgeUtility.IsLinear(edgeA) or not EdgeUtility.IsLinear(edgeB):
            return None
        direction_a = edgeA.Direction(mantissa=None)
        direction_b = edgeB.Direction(mantissa=None)
        if direction_a is None or direction_b is None:
            return None
        dot = sum(a * b for a, b in zip(direction_a, direction_b))
        dot = min(1.0, max(-1.0, dot))
        return math.degrees(math.acos(dot))

    @staticmethod
    def NormalAtParameter(edge, parameter):
        """Return the principal unit normal of the actual OCCT curve, or None at zero curvature."""
        if not isinstance(edge, Edge):
            return None
        raw = _raw_parameter(edge, parameter)
        if raw is None:
            return None
        curve, parameter_value = raw
        return _normal_at_raw_parameter(edge, curve, parameter_value)

    @staticmethod
    def Trim(edge, parameterA: float = 0.0, parameterB: float = 1.0):
        """Return a curve-preserving sub-Edge between two normalized parameters."""
        if not isinstance(edge, Edge):
            return None
        bounds = _oriented_parameter_bounds(edge)
        if bounds is None:
            return None
        curve, start_parameter, end_parameter = bounds
        try:
            u_a = float(parameterA)
            u_b = float(parameterB)
        except Exception:
            return None
        raw_a = start_parameter + u_a * (end_parameter - start_parameter)
        raw_b = start_parameter + u_b * (end_parameter - start_parameter)
        if abs(raw_a - raw_b) <= 1.0e-15:
            return None
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            lower = min(raw_a, raw_b)
            upper = max(raw_a, raw_b)
            maker = BRepBuilderAPI_MakeEdge(curve, lower, upper)
            if not maker.IsDone():
                return None
            shape = maker.Edge()
            if raw_a > raw_b:
                shape = shape.Reversed()
            return _wrap_shape_like(edge, shape)
        except Exception:
            return None

    @staticmethod
    def AdjacentWires(
        edge,
        hostTopology,
        output
    ):
        """
        Populates output with Wires in hostTopology that contain the input Edge.

        OCCT topological identity is used rather than endpoint coincidence so
        geometrically coincident but distinct Edges are not treated as the same
        topology.
        """
        if not isinstance(edge, Edge) or hostTopology is None:
            return 1

        candidates = []
        Topology.Wires(
            hostTopology,
            None,
            candidates,
        )

        result = []

        for wire in candidates:
            wire_edges = []
            Topology.Edges(
                wire,
                None,
                wire_edges,
            )

            if any(
                _same_edge_topology(edge, candidate)
                for candidate in wire_edges
                if isinstance(candidate, Edge)
            ):
                result.append(wire)

        if output is not None:
            output.extend(result)

        return 0

    @staticmethod
    def AdjacentFaces(
        edge,
        hostTopology,
        output
    ):
        """
        Populates output with Faces in hostTopology that contain the input Edge.

        OCCT topological identity is used rather than endpoint coincidence so
        geometrically coincident but distinct Edges are not treated as the same
        topology.
        """
        if not isinstance(edge, Edge) or hostTopology is None:
            return 1

        candidates = []
        Topology.Faces(
            hostTopology,
            None,
            candidates,
        )

        result = []

        for face in candidates:
            face_edges = []
            Topology.Edges(
                face,
                None,
                face_edges,
            )

            if any(
                _same_edge_topology(edge, candidate)
                for candidate in face_edges
                if isinstance(candidate, Edge)
            ):
                result.append(face)

        if output is not None:
            output.extend(result)

        return 0

    @staticmethod
    def AdjacentShells(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Shells(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentCells(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Cells(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentCellComplexes(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.CellComplexes(
            hostTopology,
            output,
        )
