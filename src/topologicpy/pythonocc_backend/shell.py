from __future__ import annotations

import types
from dataclasses import dataclass
from .topology import (
    Topology,
    _is_null_shape,
    _downward_wrappers,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
)
from .face import Face, FaceUtility
from .wire import Wire
from .edge import Edge
from .vertex import Vertex
from .occ_utils import make_occ_shell
from .helpers import unique_by_uuid, edge_key, vertex_key


@dataclass(eq=False)
class Shell(Topology):
    def __init__(self, shape=None, dictionary=None, contents=None, contexts=None, apertures=None, faces=None):
        super().__init__(shape=shape, dictionary=dictionary, contents=contents, contexts=contexts, apertures=apertures)
        self.faces = list(faces) if faces else []


    @staticmethod
    def ByFaces(faces, tolerance: float = 0.0001, silent: bool = False):
        if faces is None:
            if not silent:
                print("Shell.ByFaces - Error: The input faces parameter is None. Returning None.")
            return None
        if not isinstance(faces, list):
            faces = [faces]
        valid_faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(valid_faces) == 0:
            if not silent:
                print("Shell.ByFaces - Error: The input faces list does not contain any valid faces. Returning None.")
            return None
        occ_shell = make_occ_shell(valid_faces)
        if occ_shell is None:
            if not silent:
                print("Shell.ByFaces - Error: Could not create an OpenCascade shell. Returning None.")
            return None
        # Re-derive the Face wrappers from the sewn occ_shell rather than
        # keeping the original, independently-built valid_faces list.
        # make_occ_shell welds coincident vertices/edges across face
        # boundaries via BRepBuilderAPI_Sewing, but that welding is only
        # useful if the Shell's own Faces()/Edges() (which iterate
        # self.faces, not self.shape) actually see the welded topology. Two
        # faces coming from separately-computed boolean fragments (e.g. each
        # face-pair of Topology.Intersect's per-face decomposition) are
        # geometrically coincident along their shared boundary but were
        # never the same OCCT edge/vertex until sewn -- keeping the
        # pre-sewing faces here silently discarded that welding and doubled
        # edge counts along every such seam.
        from .topology import _iter_occ_subshapes, TopAbs_FACE
        sewn_faces = [Topology.ByOcctShape(f) for f in _iter_occ_subshapes(occ_shell, TopAbs_FACE)]
        sewn_faces = [f for f in sewn_faces if isinstance(f, Face)]
        if len(sewn_faces) == len(valid_faces):
            valid_faces = sewn_faces
        shell = Shell(shape=occ_shell, faces=valid_faces)
        Shell._patch_edge_face_membership(shell, valid_faces, tolerance=tolerance)
        return shell

    @staticmethod
    def ByWires(
        wires,
        tolerance: float = 0.0001
    ):
        """
        Creates a curve-preserving Shell by lofting through the input Wires.

        The resulting Shell is constructed directly by OCCT from the supplied
        section Wires. Curved Edges are retained as true curves and the generated
        side Faces are genuine ruled surfaces rather than planar tessellations.

        All section Wires must contain the same number of Edges so that their
        Edges correspond explicitly through the loft.

        Parameters
        ----------
        wires : list
            The ordered list of section Wires.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.

        Returns
        -------
        Shell
            The created curve-preserving Shell, or None if construction fails.

        """
        import math

        if not isinstance(wires, (list, tuple)):
            return None

        wires = [
            wire
            for wire in wires
            if isinstance(wire, Wire)
        ]

        if len(wires) < 2:
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            return None

        try:
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods
        except Exception:
            return None

        occ_wires = []
        edge_count = None

        for wire in wires:

            shape = getattr(
                wire,
                "shape",
                None,
            )

            if shape is None:
                return None

            try:
                if shape.IsNull():
                    return None

                occ_wire = topods.Wire(
                    shape
                )

                if occ_wire.IsNull():
                    return None

            except Exception:
                return None

            # Count Edges so correspondence between sections is explicit.
            try:
                explorer = TopExp_Explorer(
                    occ_wire,
                    TopAbs_EDGE,
                )

                count = 0

                while explorer.More():
                    count += 1
                    explorer.Next()

            except Exception:
                return None

            if count < 1:
                return None

            if edge_count is None:
                edge_count = count
            elif count != edge_count:
                return None

            occ_wires.append(
                occ_wire
            )

        try:
            loft = BRepOffsetAPI_ThruSections(
                False,       # isSolid
                True,        # ruled
                tolerance,
            )

            # The Wires already have corresponding Edge counts. Do not let OCCT
            # split or otherwise modify the sections while trying to establish
            # compatibility.
            loft.CheckCompatibility(
                False
            )

            for occ_wire in occ_wires:
                loft.AddWire(
                    occ_wire
                )

            shape = loft.Shape()

        except Exception:
            return None

        if shape is None:
            return None

        try:
            if shape.IsNull():
                return None
        except Exception:
            return None

        result = Topology.ByOcctShape(
            shape
        )

        if not isinstance(result, Shell):
            return None

        return result

    @staticmethod
    def _edge_face_incidence(faces, tolerance: float = 0.0001):
        """
        Map a geometric edge key (order-independent endpoint coords, tolerance-
        rounded) to the owning (face, edge) pairs. Independently built faces don't
        share OCCT edge shapes even when coincident, so incidence is geometric, not
        shape-identity.
        """
        incidence = {}
        for face in faces:
            if not isinstance(face, Face):
                continue
            for edge in face.Edges():
                if not isinstance(edge, Edge):
                    continue
                key = edge_key(edge, tolerance)
                incidence.setdefault(key, []).append((face, edge))
        return incidence

    @staticmethod
    def _patch_edge_face_membership(shell, faces, tolerance: float = 0.0001):
        """
        Per-edge Faces(host,out) patch keyed by shell _uuid: an edge can be shared by
        several Shell.ByFaces calls, each with a different owning-face count. A recognised
        shell answers from its recorded map; an unrecognised host delegates to the general
        Topology.Faces dispatch; no host falls back to the most recently recorded entry.
        """
        incidence = Shell._edge_face_incidence(faces, tolerance=tolerance)
        seen = set()
        for face in faces:
            if not isinstance(face, Face):
                continue
            for edge in face.Edges():
                if not isinstance(edge, Edge) or id(edge) in seen:
                    continue
                seen.add(id(edge))
                key = edge_key(edge, tolerance)
                owning_faces = unique_by_uuid([f for f, _ in incidence.get(key, [])])

                by_host = getattr(edge, "_shell_faces_by_host", None)
                if by_host is None:
                    by_host = {}
                    edge._shell_faces_by_host = by_host
                by_host[shell._uuid] = owning_faces

                if not getattr(edge, "_shell_faces_patched", False):
                    edge._shell_faces_patched = True

                    def _edge_faces(self, hostTopology=None, output=None):
                        host_map = getattr(self, "_shell_faces_by_host", None) or {}
                        if hostTopology is not None:
                            host_key = getattr(hostTopology, "_uuid", None)
                            if host_key is not None and host_key in host_map:
                                result = list(host_map[host_key])
                            else:
                                # hostTopology is not one of the individual
                                # Shells this edge was recorded against (e.g.
                                # it's the owning CellComplex/Cell instead) --
                                # fall back to the general-purpose vertex-set
                                # SuperTopologies query rather than guessing
                                # or returning an empty list.
                                result = Topology.SuperTopologies(self, hostTopology, "Face") or []
                        elif host_map:
                            # No host given at all: fall back to the most
                            # recently recorded context for this edge.
                            result = list(next(reversed(list(host_map.values()))))
                        else:
                            result = []
                        if output is not None:
                            output.extend(result)
                            return 0
                        return result

                    edge.Faces = types.MethodType(_edge_faces, edge)

    def Faces(self, hostTopology=None, faces=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_FACE
            )
        else:
            result = list(getattr(self, "faces", []) or [])

        if faces is not None:
            faces.extend(result)
            return 0

        return result


    def Edges(self, hostTopology=None, edges=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_EDGE
            )
        else:
            result = []

            for face in getattr(self, "faces", []) or []:
                if isinstance(face, Face):
                    result.extend(face.Edges())

        # Retain the existing backend ordering behaviour.
        result = Shell._boundary_first_ordering(
            result,
            host=self
        )

        if edges is not None:
            edges.extend(result)
            return 0

        return result


    def Vertices(self, hostTopology=None, vertices=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_VERTEX
            )
        else:
            result = []

            for edge in self.Edges():
                result.extend(
                    [edge.start, edge.end]
                )

        if vertices is not None:
            vertices.extend(result)
            return 0

        return result

    @staticmethod
    def _boundary_first_ordering(edges, host=None, tolerance: float = 0.0001):
        """
        Orders free/boundary edges first as walk-ordered (or disjoint walk-ordered) chains,
        then internal edges. The rebuilt Wire's .edges must be in true walk order because
        the algorithm layer's IsClosed/Close only checks edges[0].start vs edges[-1].end;
        boundary chains are ordered on their own (via Wire._order_edges) as they may need
        per-edge reversal.
        """
        edges = [e for e in edges if isinstance(e, Edge)]
        boundary_by_key = {}
        for e in edges:
            key = edge_key(e, tolerance)
            boundary_by_key.setdefault(key, []).append(e)

        boundary_edges = []
        other_edges = []
        seen_boundary_keys = set()
        for e in edges:
            faces_method = getattr(e, "Faces", None)
            owning = faces_method(host) if callable(faces_method) else None
            if isinstance(owning, list) and len(owning) == 1:
                key = edge_key(e, tolerance)
                if key not in seen_boundary_keys:
                    seen_boundary_keys.add(key)
                    boundary_edges.append(e)
            else:
                other_edges.append(e)

        if not boundary_edges:
            return edges

        remaining = list(boundary_edges)
        ordered_boundary = []
        while remaining:
            chain = Wire._order_edges(remaining, tolerance=tolerance)
            if chain is not None:
                ordered_boundary.extend(chain)
                break
            # Not a single simple chain: peel off connected components one at
            # a time (each is itself orderable) until none remain.
            component = [remaining[0]]
            frontier = True
            rest = remaining[1:]
            while frontier:
                frontier = False
                head_key = vertex_key(component[0].start, tolerance) if isinstance(component[0].start, Vertex) else None
                tail_key = vertex_key(component[-1].end, tolerance) if isinstance(component[-1].end, Vertex) else None
                for i, cand in enumerate(rest):
                    c_keys = (vertex_key(cand.start, tolerance), vertex_key(cand.end, tolerance))
                    if head_key in c_keys or tail_key in c_keys:
                        component.append(cand)
                        rest.pop(i)
                        frontier = True
                        break
            sub_order = Wire._order_edges(component, tolerance=tolerance)
            ordered_boundary.extend(sub_order if sub_order is not None else component)
            remaining = rest

        return ordered_boundary + other_edges

    def Shells(self, hostTopology=None, shells=None):
        result = [self]
        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def IsClosed(self, tolerance: float = 0.0001):
        """
        A shell is closed when it has no free (boundary) edges, i.e. every
        edge is shared by exactly two of the shell's faces.
        """
        faces = self.Faces() or []
        if not faces:
            return False
        return len(Shell._boundary_edges(faces, tolerance=tolerance, min_count=1, max_count=1)) == 0

    @staticmethod
    def _boundary_edges(faces, tolerance: float = 0.0001, min_count=None, max_count=None):
        """
        Returns the Edge objects (one representative per geometric location)
        whose face-incidence count falls within [min_count, max_count]
        (either bound may be None to mean "unbounded").
        """
        incidence = Shell._edge_face_incidence(faces, tolerance=tolerance)
        result = []
        for pairs in incidence.values():
            count = len(pairs)
            if min_count is not None and count < min_count:
                continue
            if max_count is not None and count > max_count:
                continue
            result.append(pairs[0][1])
        return result

    @staticmethod
    def _merge_boundary_edges(edges, tolerance: float = 0.0001):
        """
        Stitch boundary edges into a Wire (or Cluster of Wires for disjoint chains). Prefer
        Wire.ByEdges (keeps the walk-ordered .edges the naive IsClosed relies on) over
        _merge_edges_into_wires; fall back for non-simple chains.
        """
        edges = [e for e in edges if isinstance(e, Edge)]
        if not edges:
            return None
        ordered = Wire._order_edges(edges, tolerance=tolerance)
        if ordered is not None:
            wire = Wire.ByEdges(ordered, tolerance=tolerance)
            if wire is not None:
                return wire
        return Topology._merge_edges_into_wires(edges, tolerance=tolerance)

    @staticmethod
    def ExternalBoundary(shell, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the external (free/boundary) Wire of an open Shell: the wire
        stitched from all edges that belong to exactly one face of the shell.

        If the boundary edges stitch into more than one disjoint wire, the
        longest one is returned (matching the tie-break used by the
        algorithm-layer Shell.ExternalBoundary in src/topologicpy/Shell.py).
        """
        if not isinstance(shell, Shell):
            if not silent:
                print("Shell.ExternalBoundary - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None
        faces = shell.Faces() or []
        boundary_edges = Shell._boundary_edges(faces, tolerance=tolerance, min_count=1, max_count=1)
        if not boundary_edges:
            if not silent:
                print("Shell.ExternalBoundary - Error: External boundary could not be found. Returning None.")
            return None
        merged = Shell._merge_boundary_edges(boundary_edges, tolerance=tolerance)
        if merged is None:
            if not silent:
                print("Shell.ExternalBoundary - Error: External boundary could not be found. Returning None.")
            return None
        if Topology.IsInstance(merged, "Wire"):
            return merged
        # Disjoint boundary chains merged into a Cluster of Wires: keep the longest.
        wires = [w for w in getattr(merged, "topologies", []) or [] if Topology.IsInstance(w, "Wire")]
        if not wires:
            if not silent:
                print("Shell.ExternalBoundary - Error: External boundary could not be found. Returning None.")
            return None

        def _wire_length(wire):
            total = 0.0
            for edge in getattr(wire, "edges", []) or []:
                if not isinstance(edge, Edge):
                    continue

                length = Edge.Length(
                    edge,
                    tolerance=tolerance,
                )

                if length is not None:
                    total += float(length)

            return total

        wires.sort(key=_wire_length)
        return wires[-1]

    def Slice(self, otherTopology, transferDictionary: bool = False):
        """
        Slice this Shell's faces by a cutting tool, keeping self's material, and reassemble
        the surviving sub-faces into a single Shell (unlike the generic _partition_by which
        wraps the raw result as a Cluster). Cell.Prism depends on returning a Shell.
        """
        from .topology import (
            _collect_boolean_operand_shapes,
            _postprocess_boolean_result,
            _merge_backend_dictionaries,
            _iter_occ_subshapes,
        )
        try:
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
            from OCC.Core.TopAbs import TopAbs_FACE
        except Exception:
            return None
        if BOPAlgo_CellsBuilder is None:
            return None

        shapes_a = _collect_boolean_operand_shapes(self)
        shapes_b = _collect_boolean_operand_shapes(otherTopology)
        if not shapes_a or not shapes_b:
            return None

        try:
            builder = BOPAlgo_CellsBuilder()
            for shape in shapes_a:
                builder.AddArgument(shape)
            for shape in shapes_b:
                builder.AddArgument(shape)
            builder.Perform()
            if hasattr(builder, "HasErrors") and builder.HasErrors():
                return None

            empty_avoid = TopTools_ListOfShape()
            for shape in shapes_a:
                to_take = TopTools_ListOfShape()
                to_take.Append(shape)
                builder.AddToResult(to_take, empty_avoid)

            builder.MakeContainers()
            result_shape = builder.Shape()
        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)

        result_dictionary = {}
        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )

        result_faces = []
        for occ_face in _iter_occ_subshapes(result_shape, TopAbs_FACE):
            f = Face.ByOcctShape(occ_face)
            if f is not None:
                result_faces.append(f)

        if result_faces:
            new_shell = Shell.ByFaces(result_faces, silent=True)
            if new_shell is not None:
                new_shell.dictionary = result_dictionary
                return new_shell

        # Fall back to the generic wrap (e.g. a genuinely disjoint result).
        return Topology.ByOcctShape(result_shape, dictionary=result_dictionary)

    def Divide(self, otherTopology, transferDictionary: bool = False):
        return self.Slice(otherTopology, transferDictionary=transferDictionary)

    # Impose and Imprint intentionally do NOT alias Slice here (unlike
    # Divide): Impose has its own distinct semantics (keep self's exclusive
    # material AND otherTopology's whole, unsplit material -- see
    # Topology.Impose), and Imprint is already handled correctly by the base
    # Topology._split_by_tool. Aliasing both to Slice used to shadow those
    # base-class implementations for every Shell operand.


class ShellUtility:
    @staticmethod
    def Area(shell):
        """Returns the surface area of the input Shell."""
        if not isinstance(shell, Shell):
            return None

        return sum(
            FaceUtility.Area(face) or 0.0
            for face in shell.Faces()
        )

    @staticmethod
    def ExternalBoundary(
        shell,
        tolerance: float = 0.0001
    ):
        """Returns the external boundary of the input Shell."""
        return Shell.ExternalBoundary(
            shell,
            tolerance=tolerance,
            silent=True,
        )

    @staticmethod
    def InternalBoundaries(
        shell,
        tolerance: float = 0.0001
    ):
        """
        Returns the internal boundary Wires of the input Shell.

        Internal boundary Edges are those incident to two or more Faces. The
        Edges are merged into one or more Wires while preserving their native
        curve geometry.

        Parameters
        ----------
        shell : Shell
            The input Shell.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        list
            The internal boundary Wires.

        """
        if not isinstance(shell, Shell):
            return []

        faces = shell.Faces() or []

        internal_edges = Shell._boundary_edges(
            faces,
            tolerance=tolerance,
            min_count=2,
            max_count=None,
        )

        if not internal_edges:
            return []

        merged = Shell._merge_boundary_edges(
            internal_edges,
            tolerance=tolerance,
        )

        if merged is None:
            return []

        if Topology.IsInstance(merged, "Wire"):
            return [merged]

        return [
            wire
            for wire in getattr(merged, "topologies", []) or []
            if Topology.IsInstance(wire, "Wire")
        ]

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
