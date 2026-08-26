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
from .helpers import distance3, same_vertex


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
    start = edge.start
    if not isinstance(start, Vertex):
        return None
    try:
        p_first = curve.Value(first)
        p_last = curve.Value(last)
        dx = float(start.x) - float(p_first.X())
        dy = float(start.y) - float(p_first.Y())
        dz = float(start.z) - float(p_first.Z())
        d_first2 = dx * dx + dy * dy + dz * dz
        dx = float(start.x) - float(p_last.X())
        dy = float(start.y) - float(p_last.Y())
        dz = float(start.z) - float(p_last.Z())
        d_last2 = dx * dx + dy * dy + dz * dz
        if d_first2 <= d_last2:
            return curve, first, last
        return curve, last, first
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
    def ByStartVertexEndVertex(startVertex, endVertex):
        """Create a straight Edge between two backend Vertex objects."""
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex):
            return None
        try:
            shape = make_occ_edge(startVertex, endVertex)
        except Exception:
            return None
        if _is_null_shape(shape):
            return None
        return Edge(shape=shape, start=startVertex, end=endVertex)

    @staticmethod
    def ByVertices(vertices):
        """Create a straight Edge between the first and last input Vertex."""
        if vertices is None or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    @staticmethod
    def ByStartVertexEndVertexTolerance(startVertex, endVertex, tolerance: float = 0.0001):
        """Create a straight Edge if its endpoints are farther apart than tolerance."""
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex, tolerance=tolerance):
            return None
        try:
            shape = make_occ_edge(startVertex, endVertex)
        except Exception:
            return None
        if _is_null_shape(shape):
            return None
        return Edge(shape=shape, start=startVertex, end=endVertex)

    @staticmethod
    def ByNurbsParameters(controlPoints, weights, knots, isRational: bool = False, isPeriodic: bool = False, degree: int = 3):
        """Create an OCCT B-spline/NURBS Edge from exact NURBS parameters.

        ``knots`` uses the Topologic convention: the knot vector is expanded,
        so repeated knot values are repeated in the list rather than supplied
        separately as multiplicities.
        """
        vertices = [vertex for vertex in (controlPoints or []) if isinstance(vertex, Vertex)]
        if len(vertices) < 2:
            return None
        try:
            degree = int(degree)
            if degree < 1 or degree >= len(vertices):
                return None
            knot_values = [float(value) for value in knots]
            weight_values = [float(value) for value in weights]
        except Exception:
            return None
        if len(weight_values) != len(vertices) or not knot_values:
            return None
        if any(value <= 0.0 or not math.isfinite(value) for value in weight_values):
            return None
        if any(not math.isfinite(value) for value in knot_values):
            return None
        if any(knot_values[i] > knot_values[i + 1] for i in range(len(knot_values) - 1)):
            return None

        # Convert expanded Topologic knots to OCCT's unique knot + multiplicity form.
        unique_knots = []
        multiplicities = []
        for value in knot_values:
            if unique_knots and abs(value - unique_knots[-1]) <= 1.0e-14:
                multiplicities[-1] += 1
            else:
                unique_knots.append(value)
                multiplicities.append(1)
        if len(unique_knots) < 2:
            return None

        try:
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TColgp import TColgp_Array1OfPnt
            from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCC.Core.Geom import Geom_BSplineCurve
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            poles = TColgp_Array1OfPnt(1, len(vertices))
            for index, vertex in enumerate(vertices, start=1):
                poles.SetValue(index, gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)))

            occ_knots = TColStd_Array1OfReal(1, len(unique_knots))
            occ_mults = TColStd_Array1OfInteger(1, len(multiplicities))
            for index, (knot, mult) in enumerate(zip(unique_knots, multiplicities), start=1):
                occ_knots.SetValue(index, float(knot))
                occ_mults.SetValue(index, int(mult))

            if bool(isRational):
                occ_weights = TColStd_Array1OfReal(1, len(weight_values))
                for index, weight in enumerate(weight_values, start=1):
                    occ_weights.SetValue(index, float(weight))
                curve = Geom_BSplineCurve(
                    poles, occ_weights, occ_knots, occ_mults,
                    degree, bool(isPeriodic), True,
                )
            else:
                curve = Geom_BSplineCurve(
                    poles, occ_knots, occ_mults,
                    degree, bool(isPeriodic),
                )

            maker = BRepBuilderAPI_MakeEdge(curve)
            if not maker.IsDone():
                return None
            return Edge.ByOcctShape(maker.Edge())
        except Exception:
            return None

    @staticmethod
    def ByCurve(points, degree: int = 3, periodic: bool = False, tolerance: float = 0.0001):
        """Create a non-rational B-spline Edge using the input Vertices as control points."""
        vertices = [vertex for vertex in (points or []) if isinstance(vertex, Vertex)]
        if len(vertices) < 2:
            return None
        try:
            degree = max(1, min(int(degree), len(vertices) - 1))
        except Exception:
            return None

        # Expanded clamped-uniform knot vector. For periodic curves an open
        # uniform vector is supplied and OCCT validates whether the requested
        # control polygon admits the periodic construction.
        knot_count = len(vertices) + degree + 1
        if periodic:
            if knot_count <= 1:
                return None
            knots = [float(i) / float(knot_count - 1) for i in range(knot_count)]
        else:
            interior = len(vertices) - degree - 1
            knots = [0.0] * (degree + 1)
            if interior > 0:
                knots.extend(float(i) / float(interior + 1) for i in range(1, interior + 1))
            knots.extend([1.0] * (degree + 1))

        return Edge.ByNurbsParameters(
            vertices,
            [1.0] * len(vertices),
            knots,
            False,
            bool(periodic),
            degree,
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
        """Return or append the two endpoint Vertex objects."""
        result = [v for v in [self.start, self.end] if isinstance(v, Vertex)]
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
        from .helpers import unique_by_uuid

        result = []
        if hostTopology is not None:
            candidates = Topology.Edges(hostTopology) or []
            for other in candidates:
                if not isinstance(other, Edge) or other is self:
                    continue
                same_geometry = (
                    same_vertex(other.start, self.start) and same_vertex(other.end, self.end)
                ) or (
                    same_vertex(other.start, self.end) and same_vertex(other.end, self.start)
                )
                if same_geometry:
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
        """Return the same OCCT curve with its topological orientation reversed."""
        if not isinstance(edge, Edge):
            return None
        try:
            result = _wrap_shape_like(edge, edge.shape.Reversed())
            if isinstance(result, Edge):
                return result
        except Exception:
            pass
        return Edge.ByStartVertexEndVertex(edge.end, edge.start)

    def Direction(self, mantissa: int = 6):
        """Return the unit chord direction from the oriented start to end Vertex."""
        if not isinstance(self.start, Vertex) or not isinstance(self.end, Vertex):
            return None
        dx = float(self.end.x) - float(self.start.x)
        dy = float(self.end.y) - float(self.start.y)
        dz = float(self.end.z) - float(self.start.z)
        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
        if magnitude <= 0.0:
            return [0, 0, 0]
        return [
            round(dx / magnitude, mantissa),
            round(dy / magnitude, mantissa),
            round(dz / magnitude, mantissa),
        ]

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
            if isinstance(edge.start, Vertex) and isinstance(edge.end, Vertex):
                return distance3(edge.start, edge.end)
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
        """Return a Vertex at curvilinear distance from an origin on the Edge."""
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
        raw_origin = start_parameter + normalized_origin * (end_parameter - start_parameter)
        parameter_orientation = 1.0 if end_parameter >= start_parameter else -1.0

        try:
            from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
            from OCC.Core.GCPnts import GCPnts_AbscissaPoint

            adaptor = GeomAdaptor_Curve(curve)
            solver = GCPnts_AbscissaPoint(
                tol,
                adaptor,
                requested_distance * parameter_orientation,
                raw_origin,
            )
            if solver.IsDone():
                parameter_value = float(solver.Parameter())
                result = _point_at_raw_parameter(curve, parameter_value)
                if result is not None:
                    return result
        except Exception:
            pass

        # Conservative fallback: extend from the origin along the local OCCT
        # tangent. This remains exact for lines and avoids an endpoint-chord
        # assumption when an unbounded curve cannot be evaluated by GCPnts.
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
        """Return the angle in degrees between the endpoint chord directions."""
        if not isinstance(edgeA, Edge) or not isinstance(edgeB, Edge):
            return None
        direction_a = edgeA.Direction(mantissa=15)
        direction_b = edgeB.Direction(mantissa=15)
        if direction_a is None or direction_b is None:
            return None
        dot = sum(a * b for a, b in zip(direction_a, direction_b))
        dot = min(1.0, max(-1.0, dot))
        return math.degrees(math.acos(dot))

    @staticmethod
    def NormalAtParameter(edge, parameter: float = 0.5):
        """Return a unit normal at normalized parameter u on the actual OCCT curve."""
        if not isinstance(edge, Edge):
            return None
        raw = _raw_parameter(edge, parameter)
        if raw is None:
            return None
        curve, parameter_value = raw
        tangent = EdgeUtility.TangentAtParameter(edge, parameter)
        if tangent is None:
            return None
        tx, ty, tz = tangent
        eps = 1.0e-12

        # Use the principal normal when curvature is defined.
        try:
            from OCC.Core.gp import gp_Pnt, gp_Vec
            p = gp_Pnt(); d1 = gp_Vec(); d2 = gp_Vec()
            curve.D2(float(parameter_value), p, d1, d2)
            ax, ay, az = float(d2.X()), float(d2.Y()), float(d2.Z())
            dot = ax * tx + ay * ty + az * tz
            nx, ny, nz = ax - dot * tx, ay - dot * ty, az - dot * tz
            magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
            if magnitude > eps:
                return [nx / magnitude, ny / magnitude, nz / magnitude]
        except Exception:
            pass

        # A straight line has no unique Frenet normal. Return a stable unit
        # transverse direction so downstream APIs remain deterministic.
        helper = [0.0, 0.0, 1.0] if abs(tz) < 0.9 else [1.0, 0.0, 0.0]
        dot = tx * helper[0] + ty * helper[1] + tz * helper[2]
        nx = helper[0] - dot * tx
        ny = helper[1] - dot * ty
        nz = helper[2] - dot * tz
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        if magnitude <= eps:
            return None
        return [nx / magnitude, ny / magnitude, nz / magnitude]

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


# Frozen TopologicCore EdgeUtility naming aliases.
EdgeUtility.VertexAtParameter = staticmethod(EdgeUtility.PointAtParameter)
EdgeUtility.ParameterAtVertex = staticmethod(EdgeUtility.ParameterAtPoint)
EdgeUtility.VertexAtDistance = staticmethod(EdgeUtility.PointAtDistance)


# Edge -> Wire: find Wires in hostTopology containing this Edge.
def _adjacent_wires(edge, hostTopology, output):
    """Populate Wires in hostTopology that contain the input Edge."""
    from .topology import Topology
    from .helpers import same_vertex

    if not isinstance(edge, Edge) or hostTopology is None:
        return 1
    result, candidates = [], []
    Topology.Wires(hostTopology, None, candidates)
    for wire in candidates:
        wire_edges = []
        Topology.Edges(wire, None, wire_edges)
        for candidate in wire_edges:
            if (
                same_vertex(edge.start, candidate.start)
                and same_vertex(edge.end, candidate.end)
            ) or (
                same_vertex(edge.start, candidate.end)
                and same_vertex(edge.end, candidate.start)
            ):
                result.append(wire)
                break
    if output is not None:
        output.extend(result)
    return 0


# Edge -> Face: find Faces in hostTopology containing this Edge.
def _adjacent_faces(edge, hostTopology, output):
    """Populate Faces in hostTopology that contain the input Edge."""
    from .topology import Topology
    from .helpers import same_vertex

    if not isinstance(edge, Edge) or hostTopology is None:
        return 1
    result, candidates = [], []
    Topology.Faces(hostTopology, None, candidates)
    for face in candidates:
        face_edges = []
        Topology.Edges(face, None, face_edges)
        for candidate in face_edges:
            if (
                same_vertex(edge.start, candidate.start)
                and same_vertex(edge.end, candidate.end)
            ) or (
                same_vertex(edge.start, candidate.end)
                and same_vertex(edge.end, candidate.start)
            ):
                result.append(face)
                break
    if output is not None:
        output.extend(result)
    return 0


EdgeUtility.AdjacentWires = staticmethod(_adjacent_wires)
EdgeUtility.AdjacentFaces = staticmethod(_adjacent_faces)


def _make_adjacent(method_name):
    """Return a utility function delegating to a topology adjacency method."""

    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)

    return _impl


EdgeUtility.AdjacentShells = _make_adjacent("Shells")
EdgeUtility.AdjacentCells = _make_adjacent("Cells")
EdgeUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

from .helpers import not_implemented as _not_implemented


def _edge_not_implemented(name, return_value=None):
    """Create a backend placeholder for an unsupported Edge operation."""
    def _method(*args, **kwargs):
        return _not_implemented(f"Edge.{name}", return_value)
    return _method


def _edge_utility_not_implemented(name, return_value=None):
    """Create a backend placeholder for an unsupported EdgeUtility operation."""
    def _method(*args, **kwargs):
        return _not_implemented(f"EdgeUtility.{name}", return_value)
    return _method
