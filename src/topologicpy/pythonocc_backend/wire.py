from __future__ import annotations

from dataclasses import dataclass, field
from .topology import Topology
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_wire
from .helpers import same_vertex, unique_by_uuid


@dataclass(eq=False)
class Wire(Topology):
    edges: list = field(default_factory=list)

    @staticmethod
    def ByEdges(edges, tolerance=0.0001):
        if edges is None:
            return None
        edges = [e for e in edges if isinstance(e, Edge)]
        if not edges:
            return None
        ordered = Wire._order_edges(edges, tolerance=tolerance)
        if ordered is None:
            ordered = edges
        return Wire(shape=make_occ_wire(ordered), edges=ordered)

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        """
        Use BRepTools_WireExplorer (connectivity/walk order), not TopExp_Explorer (storage
        order), so the rebuilt .edges are head-to-tail -- the algorithm layer's IsClosed/Close
        only compare edges[0].start to edges[-1].end. Reuse each occ_edge as-is: a reversed
        edge stays IsSame to its forward twin, preserving shared-edge identity across
        non-manifold cells.
        """
        edges = []
        try:
            from OCC.Core.BRepTools import BRepTools_WireExplorer
            from OCC.Core.TopoDS import topods
            occ_wire = topods.Wire(shape)
            wire_explorer = BRepTools_WireExplorer(occ_wire)
            while wire_explorer.More():
                occ_edge = wire_explorer.Current()
                e = Edge.ByOcctShape(occ_edge)
                if e is not None:
                    edges.append(e)
                wire_explorer.Next()
        except Exception:
            edges = []

        if not edges:
            try:
                from OCC.Core.TopAbs import TopAbs_EDGE
                from OCC.Core.TopExp import TopExp_Explorer
                explorer = TopExp_Explorer(shape, TopAbs_EDGE)
                while explorer.More():
                    e = Edge.ByOcctShape(explorer.Current())
                    if e is not None:
                        edges.append(e)
                    explorer.Next()
            except Exception:
                return None
        if not edges:
            return None
        w = Wire(shape=shape, edges=edges)
        w.dictionary = dictionary
        w.contents = list(contents) if contents else []
        w.contexts = list(contexts) if contexts else []
        w.apertures = list(apertures) if apertures else []
        return w

    @staticmethod
    def ByVertices(vertices, close=False, tolerance=0.0001):
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
        result = list(getattr(self, "edges", []) or [])
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for edge in getattr(self, "edges", []) or []:
            if isinstance(edge, Edge):
                result.extend([edge.start, edge.end])
        result = unique_by_uuid([v for v in result if isinstance(v, Vertex)])
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        result = [self]
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def IsClosed(self, tolerance=0.0001):
        edges = getattr(self, "edges", []) or []
        if not edges:
            return False
        return same_vertex(edges[0].start, edges[-1].end, tolerance)

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
        Not in the guide's checklist; unreferenced by the algorithm layer.
        Instance method for both calling conventions: reverse edge order and swap
        each edge's start/end.
        """
        edges = getattr(self, "edges", []) or []
        if not edges:
            return None
        reversed_edges = []
        for e in reversed(edges):
            if not isinstance(e, Edge):
                return None
            new_edge = Edge.ByStartVertexEndVertex(e.end, e.start)
            if new_edge is None:
                return None
            if transferDictionaries:
                new_edge.dictionary = e.dictionary
            reversed_edges.append(new_edge)
        result = Wire(shape=make_occ_wire(reversed_edges), edges=reversed_edges)
        if transferDictionaries:
            result.dictionary = self.dictionary
        return result


class WireUtility:
    @staticmethod
    def IsClosed(wire):
        if isinstance(wire, Wire):
            return wire.IsClosed()
        return False

    @staticmethod
    def Length(wire):
        """
        Not part of the guide's minimum checklist and not called by the
        topologicpy algorithm layer (Wire.Length there sums Edge.Length over
        Topology.Edges and never reaches Core; verified: zero call sites).
        Real best-effort implementation for direct Core callers: sum of the
        wire's own edge lengths.
        """
        from .edge import EdgeUtility
        if not isinstance(wire, Wire):
            return None
        edges = getattr(wire, "edges", []) or []
        if not edges:
            return None
        total = 0.0
        for e in edges:
            length = EdgeUtility.Length(e)
            if length is None:
                return None
            total += length
        return total

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

        def vkey(v):
            return (round(v.x / tolerance), round(v.y / tolerance), round(v.z / tolerance))

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

        def vkey(v):
            return (round(v.x / tolerance), round(v.y / tolerance), round(v.z / tolerance))

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
                        if same_vertex(e.start, run[-1].end, tolerance):
                            run.append(e)
                            remaining.remove(e)
                            extended = True
                            break
                        if same_vertex(e.end, run[-1].end, tolerance):
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
                        if same_vertex(e.end, run[0].start, tolerance):
                            run.insert(0, e)
                            remaining.remove(e)
                            extended = True
                            break
                        if same_vertex(e.start, run[0].start, tolerance):
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
