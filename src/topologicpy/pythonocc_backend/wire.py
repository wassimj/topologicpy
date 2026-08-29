from __future__ import annotations

# REVISION: 2026-08-15 Wire Union semantic promotion 002

from dataclasses import dataclass, field
import math
from .topology import Topology
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_wire
from .helpers import same_vertex



def _wire_tolerance(tolerance=0.0001):
    """Return a safe positive geometric tolerance."""
    try:
        return max(abs(float(tolerance)), 1.0e-12)
    except Exception:
        return 0.0001


def _all_occ_wire_edges(wire):
    """Return every OCCT Edge stored by a backend Wire."""
    if not isinstance(wire, Wire):
        return []
    shape = getattr(wire, "shape", None)
    if shape is None:
        return []
    result = []
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopExp import TopExp_Explorer
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = Edge.ByOcctShape(explorer.Current())
            if isinstance(edge, Edge):
                result.append(edge)
            explorer.Next()
    except Exception:
        return []
    return result


def _same_occ_edge(edge_a, edge_b):
    """Return True when two backend Edges reference the same OCCT topology."""
    if not isinstance(edge_a, Edge) or not isinstance(edge_b, Edge):
        return False

    shape_a = getattr(edge_a, "shape", None)
    shape_b = getattr(edge_b, "shape", None)

    if shape_a is None or shape_b is None:
        return False

    try:
        return bool(shape_a.IsSame(shape_b))
    except Exception:
        return edge_a is edge_b


def _walk_occ_wire_edges(wire):
    """
    Return all Wire Edges in OCCT traversal order.

    None is returned when the Wire is branching/non-manifold and a single
    BRepTools_WireExplorer walk cannot consume every stored Edge.
    """
    if not isinstance(wire, Wire):
        return None
    shape = getattr(wire, "shape", None)
    if shape is None:
        return None
    walked = []
    try:
        from OCC.Core.BRepTools import BRepTools_WireExplorer
        from OCC.Core.TopoDS import topods
        explorer = BRepTools_WireExplorer(topods.Wire(shape))
        while explorer.More():
            edge = Edge.ByOcctShape(explorer.Current())
            if isinstance(edge, Edge):
                walked.append(edge)
            explorer.Next()
    except Exception:
        walked = []

    if not walked:
        return None

    all_edges = _all_occ_wire_edges(wire)
    if all_edges and len(walked) != len(all_edges):
        return None
    return walked


def _path_edges(wire, tolerance=0.0001):
    """Return a complete oriented head-to-tail Edge path, or None."""
    if not isinstance(wire, Wire):
        return None
    walked = _walk_occ_wire_edges(wire)
    if walked:
        return walked

    edges = [e for e in (getattr(wire, "edges", []) or []) if isinstance(e, Edge)]
    if not edges:
        return None
    ordered = Wire._order_edges(edges, tolerance=_wire_tolerance(tolerance))
    if ordered is None or len(ordered) != len(edges):
        return None
    return ordered


def _unique_vertices_by_position(vertices, tolerance=0.0001):
    """Return coordinate-unique backend Vertices while preserving order."""
    tol = _wire_tolerance(tolerance)
    result = []
    for vertex in vertices or []:
        if not isinstance(vertex, Vertex):
            continue
        if not any(same_vertex(vertex, existing, tolerance=tol) for existing in result):
            result.append(vertex)
    return result


def _path_vertices(wire, tolerance=0.0001):
    """Return canonical traversal-order Vertices for a simple Wire."""
    edges = _path_edges(wire, tolerance=tolerance)
    if not edges:
        return None
    vertices = [edges[0].start]
    for edge in edges:
        if isinstance(edge.end, Vertex):
            vertices.append(edge.end)
    if len(vertices) > 1 and same_vertex(vertices[0], vertices[-1], tolerance=_wire_tolerance(tolerance)):
        vertices.pop()
    return vertices


def _edge_distance_at_parameter(edge, parameter, tolerance=0.0001):
    """
    Return exact curvilinear distance from an Edge start to normalized parameter u.

    No parameter-proportional approximation is used: if OCCT cannot construct the
    required trimmed sub-edge, None is returned.
    """
    from .edge import EdgeUtility, _oriented_parameter_bounds
    if not isinstance(edge, Edge):
        return None
    try:
        u = float(parameter)
    except Exception:
        return None
    u = max(0.0, min(1.0, u))
    if u <= 0.0:
        return 0.0
    edge_length = EdgeUtility.Length(edge, tolerance=tolerance)
    if edge_length is None:
        return None
    if u >= 1.0:
        return float(edge_length)
    bounds = _oriented_parameter_bounds(edge)
    if bounds is None:
        return None
    curve, start_parameter, end_parameter = bounds
    raw_parameter = start_parameter + u * (end_parameter - start_parameter)
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        lower = min(start_parameter, raw_parameter)
        upper = max(start_parameter, raw_parameter)
        maker = BRepBuilderAPI_MakeEdge(curve, lower, upper)
        if not maker.IsDone():
            return None
        shape = maker.Edge()
        if raw_parameter < start_parameter:
            shape = shape.Reversed()
        properties = GProp_GProps()
        brepgprop.LinearProperties(shape, properties)
        value = float(properties.Mass())
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _distance_from_wire_start(wire, vertex, tolerance=0.0001):
    """Return exact curvilinear distance from a simple Wire start to a Vertex."""
    from .edge import EdgeUtility

    if not isinstance(wire, Wire) or not isinstance(vertex, Vertex):
        return None
    tol = _wire_tolerance(tolerance)
    edges = _path_edges(wire, tolerance=tol)
    if not edges:
        return None

    accumulated = 0.0
    for edge in edges:
        local_u = EdgeUtility.ParameterAtPoint(edge, vertex, tolerance=tol)
        edge_length = EdgeUtility.Length(edge, tolerance=tol)
        if edge_length is None:
            return None
        if local_u is not None:
            local_distance = _edge_distance_at_parameter(
                edge,
                local_u,
                tolerance=tol,
            )
            if local_distance is None:
                return None
            return accumulated + local_distance
        accumulated += float(edge_length)
    return None

def _make_occ_wire_batch(edges):
    """
    Builds an OCCT wire from a connected collection of edges in one batch.

    This path is intended for connected branching/non-manifold edge sets for
    which a single head-to-tail ordering does not exist. It also verifies that
    the returned OCCT wire contains every supplied edge, avoiding acceptance
    of a partial result.
    """
    if not edges:
        return None

    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopTools import TopTools_ListOfShape
        from OCC.Core.TopoDS import topods
    except Exception:
        return None

    edge_shapes = TopTools_ListOfShape()
    expected_count = 0

    for edge in edges:
        if not isinstance(edge, Edge):
            continue

        shape = getattr(edge, "shape", None)
        if shape is None:
            continue

        try:
            if shape.IsNull():
                continue
        except Exception:
            pass

        try:
            edge_shapes.Append(topods.Edge(shape))
            expected_count += 1
        except Exception:
            continue

    if expected_count == 0:
        return None

    try:
        maker = BRepBuilderAPI_MakeWire()
        maker.Add(edge_shapes)

        if not maker.IsDone():
            return None

        wire_shape = maker.Wire()

        if wire_shape is None:
            return None

        try:
            if wire_shape.IsNull():
                return None
        except Exception:
            pass

        # BRepBuilderAPI_MakeWire can report Done even if only the connected
        # subset containing the first edge was accepted. Reject such a partial
        # wire rather than silently losing edges.
        actual_count = 0
        explorer = TopExp_Explorer(wire_shape, TopAbs_EDGE)

        while explorer.More():
            actual_count += 1
            explorer.Next()

        if actual_count != expected_count:
            return None

        return wire_shape

    except Exception:
        return None


@dataclass(eq=False)
class Wire(Topology):
    edges: list = field(default_factory=list)

    @staticmethod
    def ByEdges(edges, tolerance=0.0001):
        """Construct a backend Wire from a connected collection of Edge objects."""
        if edges is None:
            return None

        edges = [e for e in edges if isinstance(e, Edge)]

        if not edges:
            return None

        # Preserve the established path for ordinary manifold/open/closed
        # wires where all edges can be ordered head-to-tail.
        ordered = Wire._order_edges(
            edges,
            tolerance=tolerance
        )

        if ordered is not None:
            shape = make_occ_wire(ordered)

            if shape is None:
                return None

            try:
                if shape.IsNull():
                    return None
            except Exception:
                pass

            return Wire(
                shape=shape,
                edges=ordered
            )

        # A connected non-manifold wire may have branch vertices and therefore
        # cannot necessarily be represented by one head-to-tail edge ordering.
        # Build all edges in one OCCT MakeWire batch instead.
        shape = _make_occ_wire_batch(edges)

        if shape is None:
            return None

        return Wire(
            shape=shape,
            edges=list(edges)
        )

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        """
        Wraps an OCCT wire.

        For ordinary manifold wires, BRepTools_WireExplorer is preferred
        because it yields connectivity/walk order. For branching/non-manifold
        wires, WireExplorer may not visit every stored edge, so a TopExp
        traversal is also collected. If TopExp finds more edges, that complete
        edge collection is retained instead.

        Reusing the original OCCT edge shapes preserves shared-edge identity
        across higher-dimensional topologies.
        """
        walk_edges = []
        all_edges = []

        # Preferred ordering for ordinary manifold/open/closed wires.
        try:
            from OCC.Core.BRepTools import BRepTools_WireExplorer
            from OCC.Core.TopoDS import topods

            occ_wire = topods.Wire(shape)
            wire_explorer = BRepTools_WireExplorer(occ_wire)

            while wire_explorer.More():
                occ_edge = wire_explorer.Current()
                edge = Edge.ByOcctShape(occ_edge)

                if edge is not None:
                    walk_edges.append(edge)

                wire_explorer.Next()

        except Exception:
            walk_edges = []

        # Complete storage traversal. This is essential for branching wires,
        # where WireExplorer may only follow one traversable chain.
        try:
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer

            explorer = TopExp_Explorer(
                shape,
                TopAbs_EDGE
            )

            while explorer.More():
                edge = Edge.ByOcctShape(
                    explorer.Current()
                )

                if edge is not None:
                    all_edges.append(edge)

                explorer.Next()

        except Exception:
            all_edges = []

        if all_edges and len(all_edges) > len(walk_edges):
            edges = all_edges
        elif walk_edges:
            edges = walk_edges
        else:
            edges = all_edges

        if not edges:
            return None

        wire = Wire(
            shape=shape,
            edges=edges
        )

        wire.dictionary = dictionary
        wire.contents = list(contents) if contents else []
        wire.contexts = list(contexts) if contexts else []
        wire.apertures = list(apertures) if apertures else []

        return wire

    def Union(self, otherTopology, transferDictionary: bool = False):
        """
        Returns the union of this wire and the input topology.

        For Wire/Wire unions, the generic OCCT Boolean may correctly compute a
        connected branching one-dimensional network but wrap it as a Cluster.
        Topologic permits such a non-manifold connected network to remain a
        Wire. Therefore, after the generic Boolean completes, this method
        promotes the result to a Wire only when every result edge can be
        consumed by Wire.ByEdges without loss.

        Parameters
        ----------
        otherTopology : Topology
            The second input topology.
        transferDictionary : bool , optional
            If True, dictionaries are transferred by the generic Boolean
            operation. Default is False.

        Returns
        -------
        Topology
            The Boolean union result.
        """
        result = Topology.Union(
            self,
            otherTopology,
            transferDictionary
        )

        if result is None:
            return None

        if not isinstance(otherTopology, Wire):
            return result

        if isinstance(result, Wire):
            return result

        try:
            result_edges = result.Edges() or []
        except Exception:
            result_edges = []

        if not result_edges:
            return result

        wire = Wire.ByEdges(
            result_edges
        )

        if wire is None:
            return result

        try:
            wire_edges = wire.Edges() or []
        except Exception:
            return result

        # A connected Wire must preserve every Boolean result Edge. If the
        # reconstruction is partial, retain the original aggregate result.
        if len(wire_edges) != len(result_edges):
            return result

        # Preserve metadata from the generic Boolean result.
        try:
            wire = Topology.SetDictionary(
                wire,
                Topology.GetDictionary(result)
            )
        except Exception:
            try:
                wire.dictionary = getattr(
                    result,
                    "dictionary",
                    {}
                )
            except Exception:
                pass

        for attr in (
            "contents",
            "contexts",
            "apertures",
        ):
            try:
                setattr(
                    wire,
                    attr,
                    list(getattr(result, attr, []) or [])
                )
            except Exception:
                pass

        return wire

    @staticmethod
    def ByVertices(vertices, close=False, tolerance=0.0001):
        """Construct a backend Wire by connecting the input Vertices in order."""
        if vertices is None:
            return None
        vertices = [v for v in vertices if isinstance(v, Vertex)]
        if len(vertices) < 2:
            return None
        edges = []
        for a, b in zip(vertices[:-1], vertices[1:]):
            if not same_vertex(a, b, tolerance):
                e = Edge.ByStartVertexEndVertex(
                    a,
                    b,
                    tolerance=tolerance,
                )
                if e is not None:
                    edges.append(e)
        if close and len(vertices) > 2 and not same_vertex(vertices[-1], vertices[0], tolerance):
            e = Edge.ByStartVertexEndVertex(
                vertices[-1],
                vertices[0],
                tolerance=tolerance,
            )
            if e is not None:
                edges.append(e)
        if not edges:
            return None
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def _order_edges(edges, tolerance=0.0001):
        """
        Return a complete oriented head-to-tail Edge ordering.

        Edge reversal uses the native curve-preserving :meth:`Edge.Reverse` operation.
        If an edge cannot be reversed without preserving its geometry, None is returned.
        """
        if not edges:
            return []
        tol = _wire_tolerance(tolerance)
        unused = [edge for edge in edges if isinstance(edge, Edge)]
        if not unused:
            return None
        ordered = [unused.pop(0)]
        while unused:
            last = ordered[-1].end
            found_index = None
            found_edge = None
            for i, edge in enumerate(unused):
                if same_vertex(edge.start, last, tol):
                    found_index = i
                    found_edge = edge
                    break
                if same_vertex(edge.end, last, tol):
                    found_index = i
                    found_edge = Edge.Reverse(edge, tolerance=tol, silent=True)
                    break
            if found_index is None or not isinstance(found_edge, Edge):
                return None
            ordered.append(found_edge)
            unused.pop(found_index)
        return ordered

    def Edges(self, hostTopology=None, edges=None):
        """Return the Edge objects stored by this Wire."""
        result = list(getattr(self, "edges", []) or [])
        if edges is not None:
            edges.extend(result)
            return 0
        return result


    def Vertices(self, hostTopology=None, vertices=None):
        """Return the Wire Vertices without duplicate coincident endpoint wrappers."""
        result = _path_vertices(self)
        if result is None:
            endpoints = []
            edges = _all_occ_wire_edges(self)
            if not edges:
                edges = [e for e in (getattr(self, "edges", []) or []) if isinstance(e, Edge)]
            for edge in edges:
                if isinstance(edge.start, Vertex):
                    endpoints.append(edge.start)
                if isinstance(edge.end, Vertex):
                    endpoints.append(edge.end)
            result = _unique_vertices_by_position(endpoints)

        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        """Return this Wire as a one-item Wire collection."""
        result = [self]
        if wires is not None:
            wires.extend(result)
            return 0
        return result



    def IsClosed(self, tolerance=0.0001):
        """Return True when the Wire forms one closed head-to-tail loop."""
        tol = _wire_tolerance(tolerance)
        edges = _path_edges(self, tolerance=tol)
        if not edges:
            return False
        return bool(same_vertex(edges[0].start, edges[-1].end, tolerance=tol))

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001):
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (Wire.ByEdgesCluster there goes through
        Topology.Edges + Wire.ByEdges directly, never through
        Core.Wire.ByEdgesCluster; verified: zero call sites). Real
        best-effort implementation for direct Core callers, matching that
        same recipe: pull the edges out of the input cluster and hand them to
        Wire.ByEdges.
        """
        edges = []
        try:
            cluster.Edges(None, edges)
        except Exception:
            return None
        edges = [e for e in edges if isinstance(e, Edge)]
        if not edges:
            return None
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def ByWires(wires, tolerance: float = 0.0001):
        """
        Not in the guide's checklist; unreferenced by the algorithm layer.
        Best-effort for direct Core callers: pool every edge and re-stitch into
        one Wire (or one per connected component) via the edge-ordering logic.
        """
        pooled_edges = []
        for w in (wires or []):
            if isinstance(w, Wire):
                pooled_edges.extend(getattr(w, "edges", []) or [])
        pooled_edges = [e for e in pooled_edges if isinstance(e, Edge)]
        if not pooled_edges:
            return None

        # Group edges into connected components (shared endpoints), then
        # order each component into its own wire.
        remaining = list(pooled_edges)
        components = []
        while remaining:
            comp = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for e in list(remaining):
                    if any(
                        same_vertex(e.start, c.start, tolerance)
                        or same_vertex(e.start, c.end, tolerance)
                        or same_vertex(e.end, c.start, tolerance)
                        or same_vertex(e.end, c.end, tolerance)
                        for c in comp
                    ):
                        comp.append(e)
                        remaining.remove(e)
                        changed = True
            components.append(comp)

        result_wires = [w for w in (Wire.ByEdges(comp, tolerance=tolerance) for comp in components) if w is not None]
        if not result_wires:
            return None
        if len(result_wires) == 1:
            return result_wires[0]
        return result_wires


    def Reverse(self, transferDictionaries: bool = False, tolerance: float = 0.0001):
        """
        Return this Wire with traversal direction reversed while preserving OCCT curves.

        Each Edge is orientation-reversed in OCCT rather than reconstructed from
        endpoints, so arcs, splines, and other native curve geometry are retained.
        """
        edges = _path_edges(self, tolerance=tolerance)
        if edges is None:
            # A branching non-manifold Wire has no unique global traversal direction.
            return None

        reversed_edges = []
        for edge in reversed(edges):
            reversed_edge = Edge.Reverse(edge, tolerance=tolerance, silent=True)
            if not isinstance(reversed_edge, Edge):
                return None
            if transferDictionaries:
                try:
                    reversed_edge.dictionary = getattr(edge, "dictionary", None)
                except Exception:
                    pass
            reversed_edges.append(reversed_edge)

        shape = make_occ_wire(reversed_edges)
        if shape is None:
            return None
        result = Wire(shape=shape, edges=reversed_edges)

        if transferDictionaries:
            try:
                result.dictionary = getattr(self, "dictionary", None)
            except Exception:
                pass
            for attr in ("contents", "contexts", "apertures"):
                try:
                    setattr(result, attr, list(getattr(self, attr, []) or []))
                except Exception:
                    pass
        return result


class WireUtility:

    @staticmethod
    def IsClosed(wire, tolerance: float = 0.0001):
        """Return True when the input backend Wire is closed."""
        if isinstance(wire, Wire):
            return wire.IsClosed(tolerance=tolerance)
        return False


    @staticmethod
    def Length(wire, tolerance: float = 0.0001):
        """Return the exact OCCT linear length of the complete Wire."""
        if not isinstance(wire, Wire):
            return None

        shape = getattr(wire, "shape", None)
        if shape is not None:
            try:
                from OCC.Core.GProp import GProp_GProps
                from OCC.Core.BRepGProp import brepgprop

                properties = GProp_GProps()
                brepgprop.LinearProperties(shape, properties)
                value = float(properties.Mass())
                if math.isfinite(value):
                    return value
            except Exception:
                pass

        from .edge import EdgeUtility
        edges = _all_occ_wire_edges(wire)
        if not edges:
            edges = [e for e in (getattr(wire, "edges", []) or []) if isinstance(e, Edge)]
        if not edges:
            return None

        total = 0.0
        for edge in edges:
            length = EdgeUtility.Length(edge, tolerance=tolerance)
            if length is None:
                return None
            total += float(length)
        return total

    @staticmethod
    def IsManifold(wire, tolerance: float = 0.0001):
        """Return True when no Wire vertex has topological degree greater than two."""
        if not isinstance(wire, Wire):
            return False
        tol = _wire_tolerance(tolerance)
        edges = _all_occ_wire_edges(wire)
        if not edges:
            edges = [e for e in (getattr(wire, "edges", []) or []) if isinstance(e, Edge)]
        if not edges:
            return False

        representatives = []
        degree = []
        for edge in edges:
            for vertex in (edge.start, edge.end):
                if not isinstance(vertex, Vertex):
                    continue
                index = None
                for i, representative in enumerate(representatives):
                    if same_vertex(vertex, representative, tolerance=tol):
                        index = i
                        break
                if index is None:
                    representatives.append(vertex)
                    degree.append(1)
                else:
                    degree[index] += 1
                    if degree[index] > 2:
                        return False
        return True


    @staticmethod
    def StartEndVertices(wire, tolerance: float = 0.0001):
        """Return the oriented start and end Vertices of a simple open Wire."""
        if not isinstance(wire, Wire):
            return None
        tol = _wire_tolerance(tolerance)
        if not WireUtility.IsManifold(wire, tolerance=tol):
            return None
        if WireUtility.IsClosed(wire, tolerance=tol):
            return None
        edges = _path_edges(wire, tolerance=tol)
        if not edges:
            return None
        start = edges[0].start
        end = edges[-1].end
        if not isinstance(start, Vertex) or not isinstance(end, Vertex):
            return None
        return [start, end]


    @staticmethod
    def PointAtParameter(wire, parameter: float = 0.0, tolerance: float = 0.0001):
        """
        Return a Vertex at a normalized global arc-length parameter on a simple Wire.

        Parameter 0 corresponds to the beginning of the OCCT traversal and 1 to
        its end. For a closed Wire those locations are coincident.
        """
        from .edge import EdgeUtility

        if not isinstance(wire, Wire):
            return None
        try:
            u = float(parameter)
        except Exception:
            return None
        if u < 0.0 or u > 1.0:
            return None

        tol = _wire_tolerance(tolerance)
        edges = _path_edges(wire, tolerance=tol)
        if not edges:
            return None

        lengths = []
        total = 0.0
        for edge in edges:
            length = EdgeUtility.Length(edge, tolerance=tol)
            if length is None:
                return None
            length = float(length)
            lengths.append(length)
            total += length

        if total <= tol:
            return None
        if u <= 0.0:
            return edges[0].start
        if u >= 1.0:
            return edges[-1].end

        target = u * total
        accumulated = 0.0
        for edge, length in zip(edges, lengths):
            if target <= accumulated + length + tol:
                local_distance = max(0.0, min(length, target - accumulated))
                if local_distance <= tol:
                    return edge.start
                if abs(local_distance - length) <= tol:
                    return edge.end
                return EdgeUtility.PointAtDistance(
                    edge,
                    local_distance,
                    origin=edge.start,
                    tolerance=tol,
                )
            accumulated += length
        return edges[-1].end


    @staticmethod
    def ParameterAtPoint(wire, vertex, tolerance: float = 0.0001):
        """Return normalized global arc-length parameter of a Vertex on a simple Wire."""
        if not isinstance(wire, Wire) or not isinstance(vertex, Vertex):
            return None
        tol = _wire_tolerance(tolerance)
        total = WireUtility.Length(wire, tolerance=tol)
        if total is None or float(total) <= tol:
            return None
        distance = _distance_from_wire_start(wire, vertex, tolerance=tol)
        if distance is None:
            return None
        value = float(distance) / float(total)
        eps = tol / max(float(total), 1.0)
        if abs(value) <= eps:
            value = 0.0
        elif abs(value - 1.0) <= eps:
            value = 1.0
        return max(0.0, min(1.0, value))


    @staticmethod
    def DistanceAtPoint(wire, vertex, origin=None, tolerance: float = 0.0001):
        """Return curvilinear distance along a simple Wire between two Vertices."""
        if not isinstance(wire, Wire) or not isinstance(vertex, Vertex):
            return None
        tol = _wire_tolerance(tolerance)
        edges = _path_edges(wire, tolerance=tol)
        if not edges:
            return None

        if not isinstance(origin, Vertex):
            origin = edges[0].start
        if not isinstance(origin, Vertex):
            return None

        d_vertex = _distance_from_wire_start(wire, vertex, tolerance=tol)
        d_origin = _distance_from_wire_start(wire, origin, tolerance=tol)
        if d_vertex is None or d_origin is None:
            return None
        return abs(float(d_vertex) - float(d_origin))


    @staticmethod
    def PointAtDistance(wire, distance: float = 0.0, origin=None, tolerance: float = 0.0001):
        """
        Return a Vertex at curvilinear distance from an origin along a simple Wire.

        From the start or an interior origin, positive distance follows the Wire
        traversal direction. From the end of an open Wire, positive distance moves
        backward into the Wire, matching the historical TopologicPy convention.
        """
        from .edge import EdgeUtility

        if not isinstance(wire, Wire):
            return None
        try:
            requested = float(distance)
        except Exception:
            return None

        tol = _wire_tolerance(tolerance)
        edges = _path_edges(wire, tolerance=tol)
        if not edges:
            return None
        if not WireUtility.IsManifold(wire, tolerance=tol):
            return None

        total = WireUtility.Length(wire, tolerance=tol)
        if total is None or float(total) <= tol:
            return None
        total = float(total)

        start = edges[0].start
        end = edges[-1].end
        if not isinstance(origin, Vertex):
            origin = start
        if not isinstance(origin, Vertex):
            return None

        origin_distance = _distance_from_wire_start(wire, origin, tolerance=tol)
        if origin_distance is None:
            return None

        if same_vertex(origin, end, tolerance=tol) and not same_vertex(start, end, tolerance=tol):
            target = total - requested
        else:
            target = float(origin_distance) + requested

        closed = same_vertex(start, end, tolerance=tol)
        if closed:
            if total <= tol:
                return None
            target = target % total
        else:
            if target < -tol or target > total + tol:
                return None
            target = max(0.0, min(total, target))

        if target <= tol:
            return start
        if abs(target - total) <= tol:
            return end

        accumulated = 0.0
        for edge in edges:
            edge_length = EdgeUtility.Length(edge, tolerance=tol)
            if edge_length is None:
                return None
            edge_length = float(edge_length)
            if target <= accumulated + edge_length + tol:
                local_distance = max(0.0, min(edge_length, target - accumulated))
                return EdgeUtility.PointAtDistance(
                    edge,
                    local_distance,
                    origin=edge.start,
                    tolerance=tol,
                )
            accumulated += edge_length
        return end

    @staticmethod
    def Project(wire, face, direction, tolerance: float = 0.0001):
        """
        Project a complete OCCT Wire onto a receiving Face along a direction.

        The operation uses BRepProj_Projection so native curves are retained.
        """
        if not isinstance(wire, Wire):
            return None
        if face is None or getattr(face, "shape", None) is None:
            return None
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            return None
        try:
            dx, dy, dz = [float(value) for value in direction]
            magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
            if magnitude <= _wire_tolerance(tolerance):
                return None
            from OCC.Core.gp import gp_Dir
            from OCC.Core.BRepProj import BRepProj_Projection
            from OCC.Core.TopoDS import topods

            projection = BRepProj_Projection(
                wire.shape,
                face.shape,
                gp_Dir(dx / magnitude, dy / magnitude, dz / magnitude),
            )
            if not projection.IsDone():
                return None

            candidates = []
            try:
                while projection.More():
                    shape = projection.Current()
                    candidate = Wire.ByOcctShape(topods.Wire(shape))
                    if isinstance(candidate, Wire):
                        candidates.append(candidate)
                    projection.Next()
            except Exception:
                # Some PythonOCC builds expose the first result without iterator
                # methods once the projection is done.
                try:
                    candidate = Wire.ByOcctShape(topods.Wire(projection.Current()))
                    if isinstance(candidate, Wire):
                        candidates.append(candidate)
                except Exception:
                    pass

            if not candidates:
                return None

            # TopologicPy Wire.Project returns one Wire. Prefer the longest
            # projected connected result if OCCT generates more than one branch.
            best = None
            best_length = -1.0
            for candidate in candidates:
                length = WireUtility.Length(candidate, tolerance=tolerance)
                if length is not None and float(length) > best_length:
                    best = candidate
                    best_length = float(length)
            return best if best is not None else candidates[0]
        except Exception:
            return None

    @staticmethod
    def Cycles(wire, tolerance: float = 0.0001):
        """
        Return all simple elementary cycles found in a backend Wire edge graph.

        Returned cycles reuse the actual OCCT edge shapes and reverse orientation only
        through :meth:`Edge.Reverse`; endpoint-chord reconstruction is never used.
        """
        if not isinstance(wire, Wire):
            return []
        edges = _all_occ_wire_edges(wire) or [e for e in (getattr(wire, "edges", []) or []) if isinstance(e, Edge)]
        if not edges:
            return []
        tol = _wire_tolerance(tolerance)
        representatives = []
        endpoints = []
        adjacency = {}
        def node_index(vertex):
            for i, representative in enumerate(representatives):
                if same_vertex(vertex, representative, tolerance=tol):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1
        for index, edge in enumerate(edges):
            a = node_index(edge.start)
            b = node_index(edge.end)
            endpoints.append((a, b))
            adjacency.setdefault(a, []).append(index)
            adjacency.setdefault(b, []).append(index)
        found = {}
        def record(path):
            key = tuple(sorted(step[0] for step in path))
            found.setdefault(key, list(path))
        def walk(start, current, path_nodes, path_edges, used_edges):
            for edge_index in adjacency.get(current, []):
                if edge_index in used_edges:
                    continue
                a, b = endpoints[edge_index]
                nxt = b if a == current else a
                step = (edge_index, current, nxt)
                if nxt == start:
                    if path_edges or a == b:
                        record(path_edges + [step])
                    continue
                if nxt in path_nodes:
                    continue
                walk(start, nxt, path_nodes + [nxt], path_edges + [step], used_edges | {edge_index})
        for start in range(len(representatives)):
            walk(start, start, [start], [], set())
        result = []
        for path in found.values():
            oriented = []
            valid = True
            for edge_index, from_node, to_node in path:
                source = edges[edge_index]
                a, b = endpoints[edge_index]
                if a == from_node and b == to_node:
                    edge = source
                elif b == from_node and a == to_node:
                    edge = Edge.Reverse(source, tolerance=tol, silent=True)
                else:
                    valid = False
                    break
                if not isinstance(edge, Edge):
                    valid = False
                    break
                oriented.append(edge)
            if valid:
                cycle = Wire.ByEdges(oriented, tolerance=tol)
                if isinstance(cycle, Wire) and cycle.IsClosed(tolerance=tol):
                    result.append(cycle)
        return result

    @staticmethod
    def Split(wire, tolerance: float = 0.0001):
        """
        Split a branching backend Wire at vertices of degree greater than two.

        Maximal runs reuse actual OCCT edge shapes. Any required orientation reversal
        uses :meth:`Edge.Reverse`; no endpoint-chord fallback is permitted.
        """
        if not isinstance(wire, Wire):
            return None
        edges = _all_occ_wire_edges(wire) or [e for e in (getattr(wire, "edges", []) or []) if isinstance(e, Edge)]
        if not edges:
            return None
        tol = _wire_tolerance(tolerance)
        representatives = []
        endpoints = []
        adjacency = {}
        def node_index(vertex):
            for i, representative in enumerate(representatives):
                if same_vertex(vertex, representative, tolerance=tol):
                    return i
            representatives.append(vertex)
            return len(representatives) - 1
        for index, edge in enumerate(edges):
            a = node_index(edge.start)
            b = node_index(edge.end)
            endpoints.append((a, b))
            adjacency.setdefault(a, []).append(index)
            adjacency.setdefault(b, []).append(index)
        if all(len(indices) <= 2 for indices in adjacency.values()):
            return [wire]
        used = set()
        runs = []
        for seed in range(len(edges)):
            if seed in used:
                continue
            a, b = endpoints[seed]
            current = a if len(adjacency[a]) != 2 else (b if len(adjacency[b]) != 2 else a)
            edge_index = seed
            run = []
            while edge_index is not None and edge_index not in used:
                source = edges[edge_index]
                a, b = endpoints[edge_index]
                if a == current:
                    oriented = source
                    nxt = b
                elif b == current:
                    oriented = Edge.Reverse(source, tolerance=tol, silent=True)
                    nxt = a
                else:
                    break
                if not isinstance(oriented, Edge):
                    return None
                run.append(oriented)
                used.add(edge_index)
                if len(adjacency.get(nxt, [])) != 2:
                    break
                candidates = [idx for idx in adjacency[nxt] if idx not in used]
                if not candidates:
                    break
                current = nxt
                edge_index = candidates[0]
            if run:
                runs.append(run)
        result = []
        for run in runs:
            if len(run) == 1:
                result.append(run[0])
            else:
                item = Wire.ByEdges(run, tolerance=tol)
                if item is not None:
                    result.append(item)
        return result if result else [wire]

    @staticmethod
    def AdjacentVertices(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Vertices(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentEdges(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Edges(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentWires(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Wires(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentFaces(
        topology,
        hostTopology,
        output
    ):
        if topology is None:
            return 1

        return topology.Faces(
            hostTopology,
            output,
        )

    @staticmethod
    def AdjacentShells(
        wire,
        hostTopology,
        output
    ):
        """
        Populates output with Shells whose boundaries contain the input Wire.

        The comparison uses OCCT Edge identity, preserving the distinction
        between coincident geometry and shared topology.
        """
        if not isinstance(wire, Wire) or hostTopology is None:
            return 1

        source_edges = (
            _all_occ_wire_edges(wire)
            or [
                edge
                for edge in getattr(wire, "edges", []) or []
                if isinstance(edge, Edge)
            ]
        )

        if not source_edges:
            return 1

        candidates = []
        Topology.Shells(
            hostTopology,
            None,
            candidates,
        )

        result = []

        for shell in candidates:
            matched = False

            for face in getattr(shell, "faces", []) or []:
                boundary = getattr(face, "external", None)

                if not isinstance(boundary, Wire):
                    continue

                candidate_edges = (
                    _all_occ_wire_edges(boundary)
                    or [
                        edge
                        for edge in getattr(boundary, "edges", []) or []
                        if isinstance(edge, Edge)
                    ]
                )

                if (
                    len(candidate_edges) == len(source_edges)
                    and all(
                        any(
                            _same_occ_edge(candidate, source)
                            for source in source_edges
                        )
                        for candidate in candidate_edges
                    )
                ):
                    matched = True
                    break

            if matched:
                result.append(shell)

        if output is not None:
            output.extend(result)

        return 0

    @staticmethod
    def AdjacentCells(
        wire,
        hostTopology,
        output
    ):
        """
        Populates output with Cells whose Shell boundaries contain the input Wire.

        The comparison uses OCCT Edge identity, preserving the distinction
        between coincident geometry and shared topology.
        """
        if not isinstance(wire, Wire) or hostTopology is None:
            return 1

        source_edges = (
            _all_occ_wire_edges(wire)
            or [
                edge
                for edge in getattr(wire, "edges", []) or []
                if isinstance(edge, Edge)
            ]
        )

        if not source_edges:
            return 1

        candidates = []
        Topology.Cells(
            hostTopology,
            None,
            candidates,
        )

        result = []

        for cell in candidates:
            matched = False

            for shell in getattr(cell, "shells", []) or []:
                for face in getattr(shell, "faces", []) or []:
                    boundary = getattr(face, "external", None)

                    if not isinstance(boundary, Wire):
                        continue

                    candidate_edges = (
                        _all_occ_wire_edges(boundary)
                        or [
                            edge
                            for edge in getattr(boundary, "edges", []) or []
                            if isinstance(edge, Edge)
                        ]
                    )

                    if (
                        len(candidate_edges) == len(source_edges)
                        and all(
                            any(
                                _same_occ_edge(candidate, source)
                                for source in source_edges
                            )
                            for candidate in candidate_edges
                        )
                    ):
                        matched = True
                        break

                if matched:
                    break

            if matched:
                result.append(cell)

        if output is not None:
            output.extend(result)

        return 0

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
