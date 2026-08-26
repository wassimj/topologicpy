from __future__ import annotations

# REVISION: 2026-08-15 Wire Union semantic promotion 002

from dataclasses import dataclass, field
import math
from .topology import Topology
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_wire
from .helpers import same_vertex, unique_by_uuid



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
    """Return exact curvilinear distance from an Edge start to normalized u."""
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
        properties = GProp_GProps()
        brepgprop.LinearProperties(maker.Edge(), properties)
        value = float(properties.Mass())
        return value if math.isfinite(value) else None
    except Exception:
        # This fallback is parameter-proportional and therefore only exact for
        # constant-speed parameterisations. It is retained solely as a last
        # resort if OCCT cannot build the trimmed sub-edge.
        return float(edge_length) * u


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
                e = Edge.ByStartVertexEndVertex(a, b)
                if e is not None:
                    edges.append(e)
        if close and len(vertices) > 2 and not same_vertex(vertices[-1], vertices[0], tolerance):
            e = Edge.ByStartVertexEndVertex(vertices[-1], vertices[0])
            if e is not None:
                edges.append(e)
        if not edges:
            return None
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def _order_edges(edges, tolerance=0.0001):
        """Return a complete head-to-tail Edge ordering, or None if branching/disconnected."""
        if not edges:
            return []
        unused = list(edges)
        ordered = [unused.pop(0)]
        while unused:
            last = ordered[-1].end
            found_index = None
            found_edge = None
            for i, edge in enumerate(unused):
                if same_vertex(edge.start, last, tolerance):
                    found_index = i
                    found_edge = edge
                    break
                if same_vertex(edge.end, last, tolerance):
                    found_index = i
                    # Reuse the original edge's shape (just orientation-
                    # flipped via .Reversed(), which OCCT keeps IsSame/hash-
                    # equal to the original) instead of fabricating a brand
                    # new edge shape -- Edge.ByStartVertexEndVertex would
                    # sever this edge's identity from any other reference to
                    # it elsewhere in the same result, silently duplicating
                    # its vertices when the two are later merged/deduped.
                    found_edge = None
                    try:
                        found_edge = Edge.ByOcctShape(edge.shape.Reversed())
                    except Exception:
                        found_edge = None
                    if found_edge is None:
                        found_edge = Edge.ByStartVertexEndVertex(edge.end, edge.start)
                    if found_edge is not None:
                        found_edge.dictionary = edge.dictionary
                    break
            if found_index is None:
                return None
            ordered.append(found_edge)
            unused.pop(found_index)
            if len(ordered) > len(edges) + 1:
                return None
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

    def IsClosed(wire, tolerance: float = 0.0001):
        """Return True when the input backend Wire is closed."""
        if isinstance(wire, Wire):
            return wire.IsClosed(tolerance=tolerance)
        return False


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
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (verified: zero call sites). Real
        best-effort implementation for direct Core callers: finds the
        elementary cycles in the wire's edge/vertex graph (relevant for
        non-manifold wires with branches; a simple open or closed wire -- the
        only kind this backend's Wire.ByEdges/.ByVertices ever build -- has
        at most one cycle, itself, when closed).
        """
        if not isinstance(wire, Wire):
            return []
        edges = getattr(wire, "edges", []) or []
        if not edges:
            return []

        tol = _wire_tolerance(tolerance)

        def vkey(v):
            return (round(v.x / tol), round(v.y / tol), round(v.z / tol))

        # Build an undirected adjacency list keyed by rounded vertex position.
        adjacency = {}
        for e in edges:
            ka, kb = vkey(e.start), vkey(e.end)
            adjacency.setdefault(ka, []).append((kb, e))
            adjacency.setdefault(kb, []).append((ka, e))

        visited_edges = set()
        cycles = []
        for e in edges:
            eid = id(e)
            if eid in visited_edges:
                continue
            # Walk forward from e.start through e until we either return to
            # the start (a cycle) or run out of unvisited connections (not a
            # cycle from this edge).
            path_edges = [e]
            visited_edges.add(eid)
            start_key = vkey(e.start)
            current_key = vkey(e.end)
            found_cycle = same_vertex(e.start, e.end, tolerance)
            while not found_cycle:
                next_edge = None
                for (other_key, cand) in adjacency.get(current_key, []):
                    if id(cand) in visited_edges:
                        continue
                    next_edge = cand
                    next_key = other_key
                    break
                if next_edge is None:
                    break
                path_edges.append(next_edge)
                visited_edges.add(id(next_edge))
                current_key = next_key
                if current_key == start_key:
                    found_cycle = True
            if found_cycle and len(path_edges) > 1:
                cycle_wire = Wire.ByEdges(path_edges, tolerance=tolerance)
                if cycle_wire is not None:
                    cycles.append(cycle_wire)
        return cycles

    @staticmethod
    def Split(wire, tolerance: float = 0.0001):
        """
        Not in the guide's checklist; unreferenced by the algorithm layer.
        Best-effort for direct Core callers: split a (possibly branching) wire at
        every vertex of degree != 2 into its maximal simple edge runs.
        """
        if not isinstance(wire, Wire):
            return None
        edges = getattr(wire, "edges", []) or []
        if not edges:
            return None

        tol = _wire_tolerance(tolerance)

        def vkey(v):
            return (round(v.x / tol), round(v.y / tol), round(v.z / tol))

        degree = {}
        for e in edges:
            for k in (vkey(e.start), vkey(e.end)):
                degree[k] = degree.get(k, 0) + 1

        branch_points = {k for k, d in degree.items() if d != 2}

        remaining = list(edges)
        runs = []
        while remaining:
            run = [remaining.pop(0)]
            # Extend forward and backward while the joining vertex is not a
            # branch point (degree != 2), matching how a "simple run" between
            # branch points is conventionally defined.
            extended = True
            while extended:
                extended = False
                # forward extension
                last_key = vkey(run[-1].end)
                if last_key not in branch_points:
                    for e in list(remaining):
                        if same_vertex(e.start, run[-1].end, tol):
                            run.append(e)
                            remaining.remove(e)
                            extended = True
                            break
                        if same_vertex(e.end, run[-1].end, tol):
                            flipped = None
                            try:
                                flipped = Edge.ByOcctShape(e.shape.Reversed())
                            except Exception:
                                flipped = None
                            if flipped is None:
                                flipped = Edge.ByStartVertexEndVertex(e.end, e.start)
                            if flipped is not None:
                                run.append(flipped)
                                remaining.remove(e)
                                extended = True
                                break
                if extended:
                    continue
                # backward extension
                first_key = vkey(run[0].start)
                if first_key not in branch_points:
                    for e in list(remaining):
                        if same_vertex(e.end, run[0].start, tol):
                            run.insert(0, e)
                            remaining.remove(e)
                            extended = True
                            break
                        if same_vertex(e.start, run[0].start, tol):
                            flipped = None
                            try:
                                flipped = Edge.ByOcctShape(e.shape.Reversed())
                            except Exception:
                                flipped = None
                            if flipped is None:
                                flipped = Edge.ByStartVertexEndVertex(e.end, e.start)
                            if flipped is not None:
                                run.insert(0, flipped)
                                remaining.remove(e)
                                extended = True
                                break
            runs.append(run)

        result = [w for w in (Wire.ByEdges(run, tolerance=tolerance) for run in runs) if w is not None]
        return result if result else None

# Wire -> Shell: find Shells in hostTopology containing this Wire.
def _adjacent_shells(wire, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex
    if not isinstance(wire, Wire) or hostTopology is None:
        return 1
    result, we_src, candidates = [], (getattr(wire, "edges", []) or []), []
    Topology.Shells(hostTopology, None, candidates)
    for s in candidates:
        for sf_face in (getattr(s, "faces", []) or []):
            wf = [sf_face.external] if getattr(sf_face, "external", None) else []
            for wf_wire in wf:
                we = getattr(wf_wire, "edges", []) or []
                if len(we) == len(we_src) and all(
                    any(same_vertex(a.start, b.start) and same_vertex(a.end, b.end) for b in we_src)
                    or any(same_vertex(a.start, b.end) and same_vertex(a.end, b.start) for b in we_src)
                    for a in we
                ):
                    result.append(s); break
            if result and result[-1] is s: break
    if output is not None: output.extend(result)
    return 0

def _adjacent_cells(wire, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex
    if not isinstance(wire, Wire) or hostTopology is None:
        return 1
    result, we_src, candidates = [], (getattr(wire, "edges", []) or []), []
    Topology.Cells(hostTopology, None, candidates)
    for c in candidates:
        for cs in (getattr(c, "shells", []) or []):
            for cs_face in (getattr(cs, "faces", []) or []):
                wf = [cs_face.external] if getattr(cs_face, "external", None) else []
                for wf_wire in wf:
                    we = getattr(wf_wire, "edges", []) or []
                    if len(we) == len(we_src) and all(
                        any(same_vertex(a.start, b.start) and same_vertex(a.end, b.end) for b in we_src)
                        or any(same_vertex(a.start, b.end) and same_vertex(a.end, b.start) for b in we_src)
                        for a in we
                    ):
                        result.append(c); break
                if result and result[-1] is c: break
            if result and result[-1] is c: break
    if output is not None: output.extend(result)
    return 0


def _make_adjacent(method_name):
    """Return a staticmethod that delegates to topology.method(hostTopology, output)."""
    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)
    return _impl

WireUtility.AdjacentVertices = _make_adjacent("Vertices")
WireUtility.AdjacentEdges = _make_adjacent("Edges")
WireUtility.AdjacentWires = _make_adjacent("Wires")
WireUtility.AdjacentFaces = _make_adjacent("Faces")
WireUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

# ---------------------------------------------------------------------------
# Wire.ByEdgesCluster, Wire.ByWires, Wire.Reverse, WireUtility.Length,
# WireUtility.Cycles, and WireUtility.Split now have real implementations
# defined on the classes above -- do not re-clobber them here (see gotcha
# about stub assignments silently overriding real implementations added
# earlier in the file).
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _wire_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Wire.{name}", return_value)
    return _method


def _wire_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"WireUtility.{name}", return_value)
    return _method
