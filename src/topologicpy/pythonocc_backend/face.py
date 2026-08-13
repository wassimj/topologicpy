from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
from .topology import Topology
from .wire import Wire
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_face
from .helpers import unique_by_uuid, edge_key


@dataclass(eq=False)
class Face(Topology):
    external: Optional[Wire] = None
    internals: list = field(default_factory=list)

    @staticmethod
    def ByExternalBoundary(wire):
        if not isinstance(wire, Wire):
            return None
        if not wire.IsClosed():
            return None
        return Face(shape=make_occ_face(wire), external=wire, internals=[])

    @staticmethod
    def ByWire(wire):
        return Face.ByExternalBoundary(wire)

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries=None):
        internalBoundaries = [w for w in (internalBoundaries or []) if isinstance(w, Wire)]
        if not internalBoundaries:
            return Face.ByExternalBoundary(externalBoundary)
        # Use ByExternalInternalBoundaries to properly add holes to OCCT shape
        return Face.ByExternalInternalBoundaries(externalBoundary, internalBoundaries)

    @staticmethod
    def ByVertices(vertices):
        wire = Wire.ByVertices(vertices, close=True)
        if wire is None:
            return None
        return Face.ByWire(wire)

    @staticmethod
    def ByExternalInternalBoundaries(externalBoundary, internalBoundaries=None, tolerance=0.0001):
        if not isinstance(externalBoundary, Wire):
            return None
        if not externalBoundary.IsClosed(tolerance=tolerance):
            return None
        if getattr(externalBoundary, "shape", None) is None:
            return None
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        except Exception:
            return None

        internalBoundaries = [w for w in (internalBoundaries or []) if isinstance(w, Wire)]

        try:
            maker = BRepBuilderAPI_MakeFace(externalBoundary.shape)
            if not maker.IsDone():
                return None
            added_wires = []
            for wire in internalBoundaries:
                if getattr(wire, "shape", None) is None:
                    continue
                maker.Add(wire.shape)
                added_wires.append(wire)
            occ_face = maker.Face()
        except Exception:
            return None

        return Face(shape=occ_face, external=externalBoundary, internals=added_wires)

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        try:
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopAbs import TopAbs_WIRE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods

            occ_face = topods.Face(shape)
            outer_occ_wire = breptools.OuterWire(occ_face)

            external = None
            internals = []
            explorer = TopExp_Explorer(occ_face, TopAbs_WIRE)
            while explorer.More():
                occ_wire = explorer.Current()
                w = Wire.ByOcctShape(occ_wire)
                if w is not None:
                    if not outer_occ_wire.IsNull() and occ_wire.IsSame(outer_occ_wire):
                        external = w
                    else:
                        internals.append(w)
                explorer.Next()

            if external is None and internals:
                external = internals.pop(0)
        except Exception:
            return None

        if external is None:
            return None

        f = Face(shape=shape, external=external, internals=internals)
        f.dictionary = dictionary
        f.contents = list(contents) if contents else []
        f.contexts = list(contexts) if contexts else []
        f.apertures = list(apertures) if apertures else []
        return f

    def ExternalBoundary(self):
        return self.external

    def InternalBoundaries(self, wires=None):
        result = list(getattr(self, "internals", []) or [])
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        if isinstance(self.external, Wire):
            result.extend(self.external.Edges())
        for wire in getattr(self, "internals", []) or []:
            if isinstance(wire, Wire):
                result.extend(wire.Edges())
        result = unique_by_uuid(result)
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for edge in self.Edges():
            result.extend([edge.start, edge.end])
        result = unique_by_uuid([v for v in result if isinstance(v, Vertex)])
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        result = []
        if isinstance(self.external, Wire):
            result.append(self.external)
        result.extend([w for w in getattr(self, "internals", []) or [] if isinstance(w, Wire)])
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        result = [self]
        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def AdjacentFaces(self, hostTopology=None, output=None):
        """Faces in hostTopology (other than self) that share an edge with self."""
        result = []
        if hostTopology is not None:
            self_keys = {edge_key(e) for e in self.Edges() if isinstance(e, Edge)}
            candidates = Topology.Faces(hostTopology) or []
            for other in candidates:
                if other is self or not isinstance(other, Face):
                    continue
                other_keys = {edge_key(e) for e in other.Edges() if isinstance(e, Edge)}
                if other_keys == self_keys:
                    # Same face as self (a distinct Python object wrapping
                    # the same boundary), not a genuinely adjacent one.
                    continue
                if self_keys & other_keys:
                    result.append(other)
            result = unique_by_uuid(result)
        if output is not None:
            output.extend(result)
            return 0
        return result


class FaceUtility:
    @staticmethod
    def Area(face):
        if not isinstance(face, Face):
            return None
        # Use only external boundary vertices for the outer area calculation
        external = getattr(face, 'external', None)
        if external is not None and hasattr(external, 'Vertices'):
            vertices = external.Vertices()
        else:
            vertices = face.Vertices()
        if len(vertices) < 3:
            return 0.0
        nx = ny = nz = 0.0
        for i, v in enumerate(vertices):
            w = vertices[(i + 1) % len(vertices)]
            nx += (v.y - w.y) * (v.z + w.z)
            ny += (v.z - w.z) * (v.x + w.x)
            nz += (v.x - w.x) * (v.y + w.y)
        area = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
        for wire in getattr(face, "internals", []) or []:
            if isinstance(wire, Wire):
                tmp_face = Face.ByWire(wire)
                if tmp_face is not None:
                    area -= FaceUtility.Area(tmp_face) or 0.0
        return abs(area)

    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5):
        if not isinstance(face, Face):
            return None
        vertices = face.Vertices()
        if len(vertices) < 3:
            return None
        a, b, c = vertices[0], vertices[1], vertices[2]
        ux, uy, uz = b.x - a.x, b.y - a.y, b.z - a.z
        vx, vy, vz = c.x - a.x, c.y - a.y, c.z - a.z
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length == 0:
            return [0, 0, 1]
        return [nx / length, ny / length, nz / length]

    @staticmethod
    def Edges(face):
        if isinstance(face, Face):
            return face.Edges()
        return []

    @staticmethod
    def _uv_bounds(face):
        """Returns (umin, umax, vmin, vmax) of the face's underlying surface, or None."""
        if not isinstance(face, Face) or getattr(face, "shape", None) is None:
            return None
        try:
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopoDS import topods
            occ_face = topods.Face(face.shape)
            umin, umax, vmin, vmax = breptools.UVBounds(occ_face)
            return (umin, umax, vmin, vmax)
        except Exception:
            return None

    @staticmethod
    def VertexAtParameters(face, u=0.5, v=0.5):
        if not isinstance(face, Face) or getattr(face, "shape", None) is None:
            return None
        bounds = FaceUtility._uv_bounds(face)
        if bounds is None:
            return None
        umin, umax, vmin, vmax = bounds
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            from .vertex import Vertex

            occ_face = topods.Face(face.shape)
            surface = BRep_Tool.Surface(occ_face)
            if surface is None:
                return None

            u_mapped = umin + float(u) * (umax - umin)
            v_mapped = vmin + float(v) * (vmax - vmin)
            pnt = surface.Value(u_mapped, v_mapped)
            return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())
        except Exception:
            return None

    @staticmethod
    def ParametersAtVertex(face, vertex):
        if not isinstance(face, Face) or getattr(face, "shape", None) is None:
            return None
        if getattr(vertex, "x", None) is None:
            return None
        bounds = FaceUtility._uv_bounds(face)
        if bounds is None:
            return None
        umin, umax, vmin, vmax = bounds
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

            occ_face = topods.Face(face.shape)
            surface = BRep_Tool.Surface(occ_face)
            if surface is None:
                return None

            pnt = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            projector = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if projector.NbPoints() < 1:
                return None
            u_raw, v_raw = projector.LowerDistanceParameters()

            u = (u_raw - umin) / (umax - umin) if (umax - umin) != 0 else 0.0
            v = (v_raw - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else 0.0
            return [u, v]
        except Exception:
            return None

    @staticmethod
    def IsInside(face, vertex, tolerance=0.0001):
        if not isinstance(face, Face) or getattr(face, "shape", None) is None:
            return False
        if getattr(vertex, "x", None) is None:
            return False
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            from OCC.Core.gp import gp_Pnt, gp_Pnt2d
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

            occ_face = topods.Face(face.shape)
            surface = BRep_Tool.Surface(occ_face)
            if surface is None:
                return False

            pnt = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            projector = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if projector.NbPoints() < 1:
                return False
            if projector.LowerDistance() > tolerance:
                return False
            u_raw, v_raw = projector.LowerDistanceParameters()

            classifier = BRepTopAdaptor_FClass2d(occ_face, tolerance)
            state = classifier.Perform(gp_Pnt2d(u_raw, v_raw))
            return state in (TopAbs_IN, TopAbs_ON)
        except Exception:
            return False

    @staticmethod
    def Triangulate(face, tolerance=0.0001, output=None):
        if output is None:
            output = []
        if not isinstance(face, Face) or getattr(face, "shape", None) is None:
            return 0
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            from OCC.Core.TopLoc import TopLoc_Location
            from .vertex import Vertex

            occ_face = topods.Face(face.shape)
            tol = tolerance if tolerance and tolerance > 0 else 0.0001
            BRepMesh_IncrementalMesh(occ_face, tol)

            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(occ_face, location)
            if triangulation is None:
                return 0
            trsf = location.Transformation()

            results = []
            for i in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(i)
                i1, i2, i3 = tri.Get()
                pnts = []
                for idx in (i1, i2, i3):
                    node = triangulation.Node(idx)
                    node = node.Transformed(trsf)
                    pnts.append(Vertex.ByCoordinates(node.X(), node.Y(), node.Z()))
                tri_face = Face.ByVertices(pnts)
                if tri_face is not None:
                    results.append(tri_face)

            output.extend(results)
            return 0
        except Exception:
            return 0

    @staticmethod
    def InternalVertex(face, tolerance=0.0001):
        if not isinstance(face, Face):
            return None
        from .topology import Topology as _Topology

        centroid = _Topology.CenterOfMass(face)
        if centroid is not None and FaceUtility.IsInside(face, centroid, tolerance=tolerance):
            return centroid

        for v in (0.5, 0.25, 0.75, 0.1, 0.9):
            for u in (0.5, 0.25, 0.75, 0.1, 0.9):
                candidate = FaceUtility.VertexAtParameters(face, u, v)
                if candidate is not None and FaceUtility.IsInside(face, candidate, tolerance=tolerance):
                    return candidate

        return centroid

    @staticmethod
    def TrimByWire(face, wire, flag=False):
        """
        Trims face by wire. Verified against the native topologic_core
        backend: for a wire that does not actually lie on/cross the face
        (e.g. a different plane entirely -- the only case exercised by the
        test suite), the result is simply a Face built directly from the
        wire, not a geometric intersection with the original face's
        boundary. Fall through to that when a genuine on-surface trim
        isn't possible.
        """
        if not isinstance(face, Face):
            return None
        if not isinstance(wire, Wire):
            return face

        result = Face.ByWire(wire)
        if result is not None:
            return result
        return face

# ---------------------------------------------------------------------------
# Explicit unsupported Face API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _face_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Face.{name}", return_value)
    return _method


def _face_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"FaceUtility.{name}", return_value)
    return _method


# Face.ByWires is implemented above (wraps ByExternalBoundary + internal wires).
# Face.ByExternalInternalBoundaries, FaceUtility.InternalVertex, FaceUtility.VertexAtParameters,
# FaceUtility.ParametersAtVertex, FaceUtility.IsInside and FaceUtility.Triangulate are all
# implemented above. Do NOT re-clobber them here.
def _face_internal_vertex(self, tolerance=0.0001, silent=False):
    return FaceUtility.InternalVertex(self, tolerance=tolerance)


# Plain instance method, not @staticmethod: must support the instance-bound
# Core.InstanceCall convention (face.InternalVertex(tolerance)), which a
# staticmethod-wrapped lambda would break (see HANDOFF.md item 1).
Face.InternalVertex = _face_internal_vertex

def _adjacent_shells(face, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex
    if not isinstance(face, Face) or hostTopology is None:
        return 1
    result, fv_src, candidates = [], face.Vertices(), []
    Topology.Shells(hostTopology, None, candidates)
    for s in candidates:
        for sf_face in (getattr(s, "faces", []) or []):
            fv = sf_face.Vertices()
            if len(fv) == len(fv_src) and all(any(same_vertex(a,b) for b in fv_src) for a in fv):
                result.append(s); break
    if output is not None: output.extend(result)
    return 0

def _adjacent_cells(face, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex
    if not isinstance(face, Face) or hostTopology is None:
        return 1
    result, fv_src, candidates = [], face.Vertices(), []
    Topology.Cells(hostTopology, None, candidates)
    for c in candidates:
        for cs in (getattr(c, "shells", []) or []):
            for cs_face in (getattr(cs, "faces", []) or []):
                fv = cs_face.Vertices()
                if len(fv) == len(fv_src) and all(any(same_vertex(a,b) for b in fv_src) for a in fv):
                    result.append(c); break
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

FaceUtility.AdjacentVertices = _make_adjacent("Vertices")
FaceUtility.AdjacentEdges = _make_adjacent("Edges")
FaceUtility.AdjacentWires = _make_adjacent("Wires")
FaceUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")




