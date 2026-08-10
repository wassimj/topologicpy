from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .topology import Topology, _merge_backend_dictionaries, BRepAlgoAPI_Common as _BRepAlgoAPI_Common
from .vertex import Vertex
from .occ_utils import make_occ_edge
from .helpers import distance3, same_vertex


@dataclass(eq=False)
class Edge(Topology):
    start: Optional[Vertex] = None
    end: Optional[Vertex] = None

    @staticmethod
    def ByStartVertexEndVertex(startVertex, endVertex):
        if not isinstance(startVertex, Vertex) or not isinstance(endVertex, Vertex):
            return None
        if same_vertex(startVertex, endVertex):
            return None
        return Edge(shape=make_occ_edge(startVertex, endVertex), start=startVertex, end=endVertex)

    @staticmethod
    def ByVertices(vertices):
        if vertices is None or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    @staticmethod
    def ByStartVertexEndVertexTolerance(startVertex, endVertex, tolerance: float = 0.0001):
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
        return Edge(shape=make_occ_edge(startVertex, endVertex), start=startVertex, end=endVertex)

    @staticmethod
    def ByCurve(points, degree: int = 3, periodic: bool = False, tolerance: float = 0.0001):
        """
        Not in the guide's checklist and unreferenced by the algorithm layer. Best-effort for
        direct Core callers: B-spline through the given points; start/end stay straight
        endpoints; periodic is accepted but ignored.
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
            builder = GeomAPI_PointsToBSpline(occ_points, 1, deg_max, GeomAbs_C2, tolerance)
            curve = builder.Curve()
            shape = BRepBuilderAPI_MakeEdge(curve).Edge()
        except Exception:
            return None
        return Edge(shape=shape, start=vertices[0], end=vertices[-1])

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        try:
            from OCC.Core.TopExp import topexp
            # CumOri=True makes FirstVertex/LastVertex honor the edge's own
            # Orientation flag. Without it, a REVERSED edge (e.g. the shared
            # boundary edge of two adjacent, non-manifold cells, which is
            # FORWARD on one side and REVERSED on the other) reports the same
            # start/end regardless of orientation, which used to force
            # Wire.ByOcctShape to fabricate a brand-new edge shape just to get
            # walk-order-correct start/end -- silently breaking edge identity
            # (IsSame/hash) between the two cells and doubling shared-edge
            # counts on every non-manifold seam.
            v1 = topexp.FirstVertex(shape, True)
            v2 = topexp.LastVertex(shape, True)
            start = Vertex.ByOcctShape(v1)
            end = Vertex.ByOcctShape(v2)
        except Exception:
            return None
        if start is None or end is None:
            return None
        e = Edge(shape=shape, start=start, end=end)
        e.dictionary = dictionary
        e.contents = list(contents) if contents else []
        e.contexts = list(contexts) if contexts else []
        e.apertures = list(apertures) if apertures else []
        return e

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
                if other is self or not isinstance(other, Edge) or same_vertex(other.start, self.start) and same_vertex(other.end, self.end):
                    continue
                if (
                    same_vertex(self.start, other.start) or same_vertex(self.start, other.end)
                    or same_vertex(self.end, other.start) or same_vertex(self.end, other.end)
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
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if mag == 0:
            return [0, 0, 0]
        return [round(dx/mag, mantissa), round(dy/mag, mantissa), round(dz/mag, mantissa)]

    def VertexByParameter(self, u: float = 0.0):
        """Creates a vertex at parameter u along the edge (0=start, 1=end)."""
        if u == 0:
            return self.start
        elif u == 1:
            return self.end
        else:
            return Vertex.ByCoordinates(
                self.start.x + (self.end.x - self.start.x) * u,
                self.start.y + (self.end.y - self.start.y) * u,
                self.start.z + (self.end.z - self.start.z) * u,
            )

    def ParameterAtVertex(self, vertex, mantissa: int = 6, tolerance: float = 0.0001):
        """Returns the parameter u at the given vertex location."""
        if not isinstance(vertex, Vertex):
            return None
        length2 = (
            (self.end.x - self.start.x) ** 2
            + (self.end.y - self.start.y) ** 2
            + (self.end.z - self.start.z) ** 2
        )
        if length2 == 0:
            return 0
        t = (
            (vertex.x - self.start.x) * (self.end.x - self.start.x)
            + (vertex.y - self.start.y) * (self.end.y - self.start.y)
            + (vertex.z - self.start.z) * (self.end.z - self.start.z)
        ) / length2
        return round(t, mantissa)

    def Length(self):
        """Returns the length of the edge."""
        return distance3(self.start, self.end)

    def Intersect(self, otherTopology, transferDictionary: bool = False):
        """
        Instance method so both calling conventions work. Edge-vs-edge uses an analytic
        closest-point test (BRepAlgoAPI_Common misses transversal zero-length crossings); all
        other cases (incl. collinear overlap) use the general boolean.
        """
        if not isinstance(otherTopology, Edge):
            return Topology._binary_boolean(self, otherTopology, _BRepAlgoAPI_Common, transferDictionary)

        p1 = (self.start.x, self.start.y, self.start.z)
        p2 = (self.end.x, self.end.y, self.end.z)
        p3 = (otherTopology.start.x, otherTopology.start.y, otherTopology.start.z)
        p4 = (otherTopology.end.x, otherTopology.end.y, otherTopology.end.z)

        hit = _segment_segment_intersection(p1, p2, p3, p4)
        if hit is None:
            # Parallel/collinear/degenerate: fall back to the general boolean
            # path, which correctly handles a genuinely overlapping (coincident)
            # sub-region between the two edges.
            return Topology._binary_boolean(self, otherTopology, _BRepAlgoAPI_Common, transferDictionary)

        result = Vertex.ByCoordinates(hit[0], hit[1], hit[2])
        if result is None:
            return None
        if transferDictionary:
            result.dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )
        return result


def _segment_segment_intersection(p1, p2, p3, p4, tolerance: float = 0.0001):
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
    if a <= tolerance ** 2 or e <= tolerance ** 2:
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
    return (float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))


class EdgeUtility:
    @staticmethod
    def Length(edge):
        if isinstance(edge, Edge) and isinstance(edge.start, Vertex) and isinstance(edge.end, Vertex):
            return distance3(edge.start, edge.end)
        return None

    @staticmethod
    def PointAtParameter(edge, parameter):
        if not isinstance(edge, Edge):
            return None
        parameter = float(parameter)
        return Vertex.ByCoordinates(
            edge.start.x + (edge.end.x - edge.start.x) * parameter,
            edge.start.y + (edge.end.y - edge.start.y) * parameter,
            edge.start.z + (edge.end.z - edge.start.z) * parameter,
        )

    @staticmethod
    def ParameterAtPoint(edge, vertex):
        if not isinstance(edge, Edge) or not isinstance(vertex, Vertex):
            return None
        length2 = (
            (edge.end.x - edge.start.x) ** 2
            + (edge.end.y - edge.start.y) ** 2
            + (edge.end.z - edge.start.z) ** 2
        )
        if length2 == 0:
            return 0
        t = (
            (vertex.x - edge.start.x) * (edge.end.x - edge.start.x)
            + (vertex.y - edge.start.y) * (edge.end.y - edge.start.y)
            + (vertex.z - edge.start.z) * (edge.end.z - edge.start.z)
        ) / length2
        # Clamp to [0,1] and check if the vertex is on the edge segment.
        # If the clamped projection is farther than tolerance from the
        # vertex, the point is not on the edge.
        t_clamped = max(0.0, min(1.0, t))
        px = edge.start.x + t_clamped * (edge.end.x - edge.start.x)
        py = edge.start.y + t_clamped * (edge.end.y - edge.start.y)
        pz = edge.start.z + t_clamped * (edge.end.z - edge.start.z)
        dist2 = (vertex.x - px) ** 2 + (vertex.y - py) ** 2 + (vertex.z - pz) ** 2
        if dist2 > 1e-8:
            raise RuntimeError("Vertex is not on the edge")
        return t

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
        ax, ay, az = edgeA.end.x - edgeA.start.x, edgeA.end.y - edgeA.start.y, edgeA.end.z - edgeA.start.z
        bx, by, bz = edgeB.end.x - edgeB.start.x, edgeB.end.y - edgeB.start.y, edgeB.end.z - edgeB.start.z
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
            tx, ty, tz = tangent_dir.X(), tangent_dir.Y(), tangent_dir.Z()
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
        nx, ny, nz = helper[0] - dot * tx, helper[1] - dot * ty, helper[2] - dot * tz
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if mag == 0:
            return None
        return [nx / mag, ny / mag, nz / mag]

    @staticmethod
    def Trim(edge, parameterA: float = 0.0, parameterB: float = 1.0):
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (verified: zero call sites). Best-effort
        real implementation for direct Core callers: returns a new Edge
        between the points at parameterA and parameterB along the input
        edge (straight chord between those two points, matching
        EdgeUtility.PointAtParameter's own straight-line parametrization).
        """
        if not isinstance(edge, Edge):
            return None
        pA = EdgeUtility.PointAtParameter(edge, parameterA)
        pB = EdgeUtility.PointAtParameter(edge, parameterB)
        if pA is None or pB is None:
            return None
        return Edge.ByStartVertexEndVertex(pA, pB)


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
            if (same_vertex(edge.start, e.start) and same_vertex(edge.end, e.end)) or \
               (same_vertex(edge.start, e.end) and same_vertex(edge.end, e.start)):
                result.append(w); break
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
        fw = getattr(f, 'external', None)
        if fw is None:
            continue
        fe = []
        Topology.Edges(fw, None, fe)
        for e in fe:
            if (same_vertex(edge.start, e.start) and same_vertex(edge.end, e.end)) or \
               (same_vertex(edge.start, e.end) and same_vertex(edge.end, e.start)):
                result.append(f); break
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
