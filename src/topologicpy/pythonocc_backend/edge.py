from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .topology import (
    Topology,
    _is_null_shape,
    _merge_backend_dictionaries,
    BRepAlgoAPI_Common as _BRepAlgoAPI_Common,
)
from .vertex import Vertex
from .occ_utils import make_occ_edge
from .helpers import same_vertex



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

    # The geometric parameter interval is independent of topological orientation.
    # Respect the OCCT edge orientation so normalized u=0 and u=1 always mean
    # TopologicPy start and end respectively, including on reversed/closed edges.
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
    import math

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

    The OCCT edge shape is wrapped immediately, but its start and end Vertex
    wrappers are created only when ``start``, ``end``, ``StartVertex()``,
    ``EndVertex()``, or another endpoint-dependent method is accessed.

    This keeps ``Topology.Edges(...)`` shallow and avoids eagerly constructing
    two Vertex wrappers for every returned Edge.
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

    # ---------------------------------------------------------------------
    # Lazy endpoints
    # ---------------------------------------------------------------------

    def _load_start(self):
        if self._start_loaded:
            return self._start

        self._start_loaded = True

        if _is_null_shape(getattr(self, "shape", None)):
            self._start = None
            return None

        try:
            from OCC.Core.TopExp import topexp

            # CumOri=True preserves the edge's own orientation, which is
            # important for edges traversed from oriented wires/faces.
            occ_vertex = topexp.FirstVertex(self.shape, True)

            if _is_null_shape(occ_vertex):
                self._start = None
            else:
                self._start = Vertex.ByOcctShape(occ_vertex)

        except Exception:
            self._start = None

        return self._start

    def _load_end(self):
        if self._end_loaded:
            return self._end

        self._end_loaded = True

        if _is_null_shape(getattr(self, "shape", None)):
            self._end = None
            return None

        try:
            from OCC.Core.TopExp import topexp

            # CumOri=True preserves the edge's own orientation.
            occ_vertex = topexp.LastVertex(self.shape, True)

            if _is_null_shape(occ_vertex):
                self._end = None
            else:
                self._end = Vertex.ByOcctShape(occ_vertex)

        except Exception:
            self._end = None

        return self._end

    @property
    def start(self) -> Optional[Vertex]:
        return self._load_start()

    @start.setter
    def start(self, value):
        self._start = value if isinstance(value, Vertex) else None
        self._start_loaded = True

    @property
    def end(self) -> Optional[Vertex]:
        return self._load_end()

    @end.setter
    def end(self, value):
        self._end = value if isinstance(value, Vertex) else None
        self._end_loaded = True

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @staticmethod
    def ByStartVertexEndVertex(startVertex, endVertex):
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex):
            return None
        return Edge(
            shape=make_occ_edge(startVertex, endVertex),
            start=startVertex,
            end=endVertex,
        )

    @staticmethod
    def ByVertices(vertices):
        if vertices is None or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    @staticmethod
    def ByStartVertexEndVertexTolerance(
        startVertex,
        endVertex,
        tolerance: float = 0.0001,
    ):
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (verified: zero call sites; Edge.ByVertices
        always goes through the tolerance-less ByStartVertexEndVertex). Real
        best-effort implementation for direct Core callers: identical to
        ByStartVertexEndVertex but with a caller-supplied coincidence
        tolerance instead of the hardcoded default in helpers.same_vertex.
        """
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex, tolerance=tolerance):
            return None
        return Edge(
            shape=make_occ_edge(startVertex, endVertex),
            start=startVertex,
            end=endVertex,
        )

    @staticmethod
    def ByCurve(
        points,
        degree: int = 3,
        periodic: bool = False,
        tolerance: float = 0.0001,
    ):
        """
        Not in the guide's checklist and unreferenced by the algorithm layer.
        Best-effort for direct Core callers: B-spline through the given points;
        start/end stay straight endpoints; periodic is accepted but ignored.
        """
        try:
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TColgp import TColgp_Array1OfPnt
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
            from OCC.Core.GeomAbs import GeomAbs_C2
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        except Exception:
            return None

        vertices = [p for p in (points or []) if isinstance(p, Vertex)]
        if len(vertices) < 2:
            return None

        try:
            occ_points = TColgp_Array1OfPnt(1, len(vertices))
            for i, v in enumerate(vertices, start=1):
                occ_points.SetValue(i, gp_Pnt(v.x, v.y, v.z))

            deg_max = max(1, min(int(degree), len(vertices) - 1))
            builder = GeomAPI_PointsToBSpline(
                occ_points,
                1,
                deg_max,
                GeomAbs_C2,
                tolerance,
            )
            curve = builder.Curve()
            shape = BRepBuilderAPI_MakeEdge(curve).Edge()
        except Exception:
            return None

        return Edge(
            shape=shape,
            start=vertices[0],
            end=vertices[-1],
        )

    @staticmethod
    def ByOcctShape(
        shape,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
    ):
        """
        Wraps an existing OCCT edge without eagerly constructing its endpoint
        Vertex wrappers.

        Start and end vertices are resolved lazily on first access and then
        cached on this Edge wrapper.
        """
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

    # ---------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------

    def StartVertex(self):
        return self.start

    def EndVertex(self):
        return self.end

    def Vertices(self, hostTopology=None, vertices=None):
        result = [v for v in [self.start, self.end] if isinstance(v, Vertex)]
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = [self]
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def AdjacentEdges(self, hostTopology=None, output=None):
        """Edges in hostTopology (other than self) that share a vertex with self."""
        from .helpers import unique_by_uuid

        result = []
        if hostTopology is not None:
            candidates = Topology.Edges(hostTopology) or []
            for other in candidates:
                if (
                    other is self
                    or not isinstance(other, Edge)
                    or (
                        same_vertex(other.start, self.start)
                        and same_vertex(other.end, self.end)
                    )
                ):
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
        """Returns a new Edge with start and end swapped."""
        if not isinstance(edge, Edge):
            return None
        return Edge.ByStartVertexEndVertex(edge.end, edge.start)

    def Direction(self, mantissa: int = 6):
        """Returns the direction vector [dx, dy, dz] of the edge."""
        import math

        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag == 0:
            return [0, 0, 0]
        return [
            round(dx / mag, mantissa),
            round(dy / mag, mantissa),
            round(dz / mag, mantissa),
        ]

    def VertexByParameter(self, u: float = 0.0):
        """Return a Vertex at normalized parameter u along the actual OCCT curve."""
        return EdgeUtility.PointAtParameter(self, u)

    def ParameterAtVertex(self, vertex, mantissa: int = 6, tolerance: float = 0.0001):
        """Return the normalized parameter of a Vertex lying on this Edge."""
        try:
            value = EdgeUtility.ParameterAtPoint(self, vertex, tolerance=tolerance)
        except Exception:
            return None
        return None if value is None else round(float(value), mantissa)

    def Length(self):
        """Returns the exact geometric length of the edge."""
        return EdgeUtility.Length(self)

    def Intersect(self, otherTopology, transferDictionary: bool = False):
        """
        Instance method so both calling conventions work. Edge-vs-edge uses an
        analytic closest-point test (BRepAlgoAPI_Common misses transversal
        zero-length crossings); all other cases (incl. collinear overlap) use
        the general boolean.
        """
        if not isinstance(otherTopology, Edge):
            return Topology._binary_boolean(
                self,
                otherTopology,
                _BRepAlgoAPI_Common,
                transferDictionary,
            )

        p1 = (self.start.x, self.start.y, self.start.z)
        p2 = (self.end.x, self.end.y, self.end.z)
        p3 = (
            otherTopology.start.x,
            otherTopology.start.y,
            otherTopology.start.z,
        )
        p4 = (
            otherTopology.end.x,
            otherTopology.end.y,
            otherTopology.end.z,
        )

        hit = _segment_segment_intersection(p1, p2, p3, p4)
        if hit is None:
            # Parallel/collinear/degenerate: fall back to the general boolean
            # path, which correctly handles a genuinely overlapping
            # (coincident) sub-region between the two edges.
            return Topology._binary_boolean(
                self,
                otherTopology,
                _BRepAlgoAPI_Common,
                transferDictionary,
            )

        result = Vertex.ByCoordinates(hit[0], hit[1], hit[2])
        if result is None:
            return None

        if transferDictionary:
            result.dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self),
                Topology.GetDictionary(otherTopology),
            )
        return result


def _segment_segment_intersection(
    p1,
    p2,
    p3,
    p4,
    tolerance: float = 0.0001,
):
    """
    Returns the (x, y, z) point where finite 3-D segments p1-p2 and p3-p4
    meet (within tolerance), or None if they are parallel/collinear (the
    caller should fall back to a boolean-based test in that case) or if the
    segments' closest approach lies outside either segment's own extent or
    farther apart than tolerance.
    """
    import numpy as np

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)

    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3

    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    if a <= tolerance**2 or e <= tolerance**2:
        return None

    f = float(np.dot(d2, r))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d1, r))
    denom = a * e - b * b

    if abs(denom) <= 1e-12:
        # Parallel (or nearly so) segments: let the caller fall back to the
        # general boolean path, which can still detect true overlap.
        return None

    s = (b * f - c * e) / denom
    t = (a * f - b * c) / denom

    eps = tolerance
    if s < -eps or s > 1 + eps or t < -eps or t > 1 + eps:
        return None

    s = min(max(s, 0.0), 1.0)
    t = min(max(t, 0.0), 1.0)
    closest_on_1 = p1 + d1 * s
    closest_on_2 = p3 + d2 * t
    gap = float(np.linalg.norm(closest_on_1 - closest_on_2))

    if gap > tolerance:
        return None

    midpoint = (closest_on_1 + closest_on_2) / 2.0
    return (
        float(midpoint[0]),
        float(midpoint[1]),
        float(midpoint[2]),
    )


class EdgeUtility:
    @staticmethod
    def IsClosed(edge, tolerance: float = 0.0001):
        """Returns True if the input Edge is topologically closed."""
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
        """Returns True if the actual OCCT edge geometry is one straight segment."""
        import math

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
            chord = math.sqrt(dx * dx + dy * dy + dz * dz)
            if chord <= tolerance:
                return False

            # A straight Bezier/B-spline must have collinear control poles.
            for i in range(2, count):
                pole = curve.Pole(i)
                px = float(pole.X()) - ax
                py = float(pole.Y()) - ay
                pz = float(pole.Z()) - az
                cx = py * dz - pz * dy
                cy = pz * dx - px * dz
                cz = px * dy - py * dx
                if math.sqrt(cx * cx + cy * cy + cz * cz) / chord > tolerance:
                    return False

            # Collinear poles alone are insufficient: a curve can double back
            # along the same line. Exact curve length must also equal the chord.
            curve_length = EdgeUtility.Length(edge, tolerance=tolerance)
            return bool(
                curve_length is not None
                and abs(float(curve_length) - chord) <= tolerance
            )
        except Exception:
            return False

    @staticmethod
    def Length(edge, tolerance: float = 0.0001):
        """Returns the exact geometric length of an Edge."""
        import math

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
        """Return a Vertex at normalized parameter [0, 1] on the actual OCCT curve."""
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
        """Return the normalized parameter of a Vertex on the trimmed OCCT Edge."""
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
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (Edge.Angle in the algorithm layer is a
        self-contained vector-math implementation that never reaches Core;
        verified: zero call sites for Core.EdgeUtility.Angle). Best-effort
        real implementation for direct Core callers: the angle in degrees
        between the two edges' direction vectors (start -> end), in [0, 180].
        """
        import math

        if not isinstance(edgeA, Edge) or not isinstance(edgeB, Edge):
            return None

        ax = edgeA.end.x - edgeA.start.x
        ay = edgeA.end.y - edgeA.start.y
        az = edgeA.end.z - edgeA.start.z
        bx = edgeB.end.x - edgeB.start.x
        by = edgeB.end.y - edgeB.start.y
        bz = edgeB.end.z - edgeB.start.z

        mag_a = math.sqrt(ax * ax + ay * ay + az * az)
        mag_b = math.sqrt(bx * bx + by * by + bz * bz)
        if mag_a == 0 or mag_b == 0:
            return None

        dot = (ax * bx + ay * by + az * bz) / (mag_a * mag_b)
        dot = min(1.0, max(-1.0, dot))
        return math.degrees(math.acos(dot))

    @staticmethod
    def NormalAtParameter(edge, parameter):
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (verified: zero call sites). Best-effort
        real implementation for direct Core callers: uses the edge's real
        OCCT curve (straight or, for Edge.ByCurve-built edges, a B-spline) via
        GeomLProp_CLProps to get the tangent at the given [0, 1] parameter,
        then returns any unit vector perpendicular to that tangent (a 1-D
        curve alone does not define a unique normal/binormal frame).
        """
        import math

        if not isinstance(edge, Edge):
            return None

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomLProp import GeomLProp_CLProps

            curve, first, last = BRep_Tool.Curve(edge.shape)
            u = first + (last - first) * float(parameter)
            props = GeomLProp_CLProps(curve, u, 1, 1e-9)
            if not props.IsTangentDefined():
                return None

            from OCC.Core.gp import gp_Dir

            tangent_dir = gp_Dir()
            props.Tangent(tangent_dir)
            tx, ty, tz = (
                tangent_dir.X(),
                tangent_dir.Y(),
                tangent_dir.Z(),
            )
        except Exception:
            tx = edge.end.x - edge.start.x
            ty = edge.end.y - edge.start.y
            tz = edge.end.z - edge.start.z
            mag = math.sqrt(tx * tx + ty * ty + tz * tz)
            if mag == 0:
                return None
            tx, ty, tz = tx / mag, ty / mag, tz / mag

        # Any vector not parallel to the tangent, made perpendicular via
        # Gram-Schmidt, then normalized.
        helper = (0.0, 0.0, 1.0) if abs(tz) < 0.9 else (1.0, 0.0, 0.0)
        dot = tx * helper[0] + ty * helper[1] + tz * helper[2]
        nx = helper[0] - dot * tx
        ny = helper[1] - dot * ty
        nz = helper[2] - dot * tz
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if mag == 0:
            return None
        return [nx / mag, ny / mag, nz / mag]

    @staticmethod
    def Trim(edge, parameterA: float = 0.0, parameterB: float = 1.0):
        """Return a curve-preserving sub-edge between normalized parameters."""
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


# Edge -> Wire: find Wires in hostTopology containing this Edge.
def _adjacent_wires(edge, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex

    if not isinstance(edge, Edge) or hostTopology is None:
        return 1

    result, candidates = [], []
    Topology.Wires(hostTopology, None, candidates)

    for w in candidates:
        we = []
        Topology.Edges(w, None, we)

        for e in we:
            if (
                same_vertex(edge.start, e.start)
                and same_vertex(edge.end, e.end)
            ) or (
                same_vertex(edge.start, e.end)
                and same_vertex(edge.end, e.start)
            ):
                result.append(w)
                break

    if output is not None:
        output.extend(result)
    return 0


# Edge -> Face: find Faces in hostTopology containing this Edge.
def _adjacent_faces(edge, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex

    if not isinstance(edge, Edge) or hostTopology is None:
        return 1

    result, candidates = [], []
    Topology.Faces(hostTopology, None, candidates)

    for f in candidates:
        # Faces are shallow in the optimized backend, so do not read the
        # implementation detail ``f.external``. Ask the Face for its Edges
        # through the topology API instead.
        fe = []
        Topology.Edges(f, None, fe)

        for e in fe:
            if (
                same_vertex(edge.start, e.start)
                and same_vertex(edge.end, e.end)
            ) or (
                same_vertex(edge.start, e.end)
                and same_vertex(edge.end, e.start)
            ):
                result.append(f)
                break

    if output is not None:
        output.extend(result)
    return 0


EdgeUtility.AdjacentWires = staticmethod(_adjacent_wires)
EdgeUtility.AdjacentFaces = staticmethod(_adjacent_faces)


def _make_adjacent(method_name):
    """Return a staticmethod that delegates to topology.method(hostTopology, output)."""

    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)

    return _impl


EdgeUtility.AdjacentShells = _make_adjacent("Shells")
EdgeUtility.AdjacentCells = _make_adjacent("Cells")
EdgeUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

# ---------------------------------------------------------------------------
# Edge.ByCurve, Edge.ByStartVertexEndVertexTolerance, EdgeUtility.Angle,
# EdgeUtility.NormalAtParameter, and EdgeUtility.Trim now have real
# implementations defined on the classes above -- do not re-clobber them
# here (see gotcha about stub assignments silently overriding real
# implementations added earlier in the file).
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _edge_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Edge.{name}", return_value)

    return _method


def _edge_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"EdgeUtility.{name}", return_value)

    return _method
