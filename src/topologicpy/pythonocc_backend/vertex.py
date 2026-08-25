from __future__ import annotations

from dataclasses import dataclass
import math

from .topology import Topology, _shape_from_topology, _is_null_shape
from .occ_utils import make_occ_vertex
from .helpers import distance3, same_vertex, unique_by_uuid


@dataclass(eq=False)
class Vertex(Topology):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def ByCoordinates(x=0.0, y=0.0, z=0.0):
        x = float(x)
        y = float(y)
        z = float(z)
        return Vertex(shape=make_occ_vertex(x, y, z), x=x, y=y, z=z)

    @staticmethod
    def ByPoint(point):
        try:
            return Vertex.ByCoordinates(point.X(), point.Y(), point.Z())
        except Exception:
            return None

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        try:
            from OCC.Core.BRep import BRep_Tool
            pnt = BRep_Tool.Pnt(shape)
            x, y, z = pnt.X(), pnt.Y(), pnt.Z()
        except Exception:
            return None
        v = Vertex(shape=shape, x=float(x), y=float(y), z=float(z))
        v.dictionary = dictionary
        v.contents = list(contents) if contents else []
        v.contexts = list(contexts) if contexts else []
        v.apertures = list(apertures) if apertures else []
        return v

    def X(self):
        return float(self.x)

    def Y(self):
        return float(self.y)

    def Z(self):
        return float(self.z)

    def Coordinates(self):
        return [self.x, self.y, self.z]

    def Vertices(self, hostTopology=None, vertices=None):
        result = [self]
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def AdjacentVertices(self, hostTopology=None, output=None):
        """Vertices in hostTopology connected to self by a shared edge."""
        result = []
        if hostTopology is not None:
            edges = Topology.Edges(hostTopology) or []
            for e in edges:
                if not hasattr(e, "start") or not hasattr(e, "end"):
                    continue
                if same_vertex(e.start, self):
                    result.append(e.end)
                elif same_vertex(e.end, self):
                    result.append(e.start)
            result = unique_by_uuid(result)
        if output is not None:
            output.extend(result)
            return 0
        return result

    @staticmethod
    def ByCoordinatesString(coordinatesString, separator=","):
        """
        Parses a coordinate string and returns a Vertex.

        This method mirrors the legacy ``topologic_core.Vertex.ByCoordinatesString``
        API for direct Core callers.
        """
        if not isinstance(coordinatesString, str):
            return None
        try:
            parts = [p.strip() for p in coordinatesString.split(separator)]
            parts = [p for p in parts if p != ""]
            if len(parts) < 2 or len(parts) > 3:
                return None
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) == 3 else 0.0
        except (ValueError, IndexError):
            return None
        return Vertex.ByCoordinates(x, y, z)

    @staticmethod
    def Project(vertex, topology, direction=None, tolerance: float = 0.0001):
        """
        Projects a Vertex onto the underlying geometry of the input topology.

        For a Face and ``direction is None``, OCCT projects the point normally
        onto the Face's supporting surface. For a planar Face and an explicit
        direction, the intersection with the infinite supporting plane is used,
        preserving the historical TopologicPy semantics. For non-planar Faces an
        explicit direction uses OCCT's curve/Face intersector. For non-Face
        topology and no explicit direction, the nearest point on the native OCCT
        shape is returned.
        """
        if not isinstance(vertex, Vertex):
            return None

        target_shape = _shape_from_topology(topology)
        if _is_null_shape(target_shape):
            return None

        tol = abs(float(tolerance)) if isinstance(tolerance, (int, float)) else 0.0001
        if tol <= 0.0:
            tol = 1.0e-12

        # Treat an empty direction as unspecified for compatibility with the
        # historical TopologicPy projection API.
        if isinstance(direction, (list, tuple)) and len(direction) == 0:
            direction = None

        try:
            from OCC.Core.gp import gp_Pnt
            point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
        except Exception:
            return None

        # Face-specific projection. This is intentionally based on the native
        # supporting surface rather than a fitted plane.
        if Topology.IsInstance(topology, "Face"):
            try:
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
                from OCC.Core.GeomAbs import GeomAbs_Plane
                from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
                from OCC.Core.TopoDS import topods

                occ_face = topods.Face(target_shape)
                adaptor = BRepAdaptor_Surface(occ_face, True)

                if direction is None:
                    surface = BRep_Tool.Surface(occ_face)
                    if surface is None:
                        return None
                    projector = GeomAPI_ProjectPointOnSurf(point, surface)
                    if projector.NbPoints() < 1:
                        return None
                    pnt = projector.NearestPoint()
                    return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())

                try:
                    dx = float(direction[0])
                    dy = float(direction[1])
                    dz = float(direction[2])
                except Exception:
                    return None

                norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                if norm <= tol:
                    return None

                # For planar faces, intersect the infinite supporting plane.
                # This preserves the old TopologicPy behaviour even when the
                # projected point falls outside the trimmed Face.
                if adaptor.GetType() == GeomAbs_Plane:
                    plane = adaptor.Plane()
                    location = plane.Location()
                    normal = plane.Axis().Direction()
                    nx, ny, nz = normal.X(), normal.Y(), normal.Z()
                    denom = dx * nx + dy * ny + dz * nz
                    if abs(denom) <= tol:
                        return None
                    signed = (
                        (point.X() - location.X()) * nx
                        + (point.Y() - location.Y()) * ny
                        + (point.Z() - location.Z()) * nz
                    )
                    parameter = -signed / denom
                    return Vertex.ByCoordinates(
                        point.X() + parameter * dx,
                        point.Y() + parameter * dy,
                        point.Z() + parameter * dz,
                    )
            except Exception:
                # Fall through to the native Face-intersection path below.
                pass

            # Directional projection onto a non-planar trimmed Face.
            if direction is not None:
                try:
                    from OCC.Core.gp import gp_Dir, gp_Lin
                    from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector

                    dx = float(direction[0])
                    dy = float(direction[1])
                    dz = float(direction[2])
                    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if norm <= tol:
                        return None

                    line = gp_Lin(point, gp_Dir(dx, dy, dz))
                    intersector = IntCurvesFace_ShapeIntersector()
                    intersector.Load(target_shape, tol)
                    intersector.Perform(line, -1.0e100, 1.0e100)
                    count = intersector.NbPnt()
                    if count < 1:
                        return None

                    best_index = 1
                    best_parameter = abs(float(intersector.WParameter(1)))
                    for i in range(2, count + 1):
                        value = abs(float(intersector.WParameter(i)))
                        if value < best_parameter:
                            best_parameter = value
                            best_index = i
                    pnt = intersector.Pnt(best_index)
                    return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())
                except Exception:
                    return None

        # Nearest point on a generic native shape.
        if direction is None:
            try:
                from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
                dist_calc = BRepExtrema_DistShapeShape(vertex.shape, target_shape)
                dist_calc.Perform()
                if not dist_calc.IsDone() or dist_calc.NbSolution() < 1:
                    return None
                pnt = dist_calc.PointOnShape2(1)
                return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())
            except Exception:
                return None

        return None

    def Fuse(self, otherTopology):
        """
        Mirrors the legacy Core Vertex fuse operation for direct backend callers.
        """
        if not isinstance(otherTopology, Vertex):
            return None
        if same_vertex(self, otherTopology):
            return self
        from .cluster import Cluster
        return Cluster.ByTopologies([self, otherTopology])


class VertexUtility:
    @staticmethod
    def Distance(vertexA, vertexB):
        """Returns the exact distance between two Vertices."""
        if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
            return None
        try:
            distance = Topology.Distance(vertexA, vertexB)
            if distance is not None:
                return float(distance)
        except Exception:
            pass
        return distance3(vertexA, vertexB)

    @staticmethod
    def DistanceToTopology(vertex, topology, tolerance: float = 0.0001):
        """Returns the native OCCT minimum distance from a Vertex to a topology."""
        if not isinstance(vertex, Vertex) or not isinstance(topology, Topology):
            return None
        try:
            value = Topology.Distance(vertex, topology, tolerance=tolerance)
            return None if value is None else float(value)
        except Exception:
            return None

    @staticmethod
    def IsCoincident(vertexA, vertexB, tolerance: float = 0.0001):
        """Returns True when two Vertices are coincident within tolerance."""
        if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
            return False
        try:
            d = VertexUtility.Distance(vertexA, vertexB)
            return d is not None and d <= abs(float(tolerance))
        except Exception:
            return False

    @staticmethod
    def IsInternal(vertex, topology, tolerance: float = 0.0001):
        """
        Classifies a Vertex against a primitive or aggregate backend topology.

        Face and Cell classification use OCCT's native classifiers. Boundary-only
        topology types use exact OCCT shape distance.
        """
        if not isinstance(vertex, Vertex) or not isinstance(topology, Topology):
            return False

        tol = abs(float(tolerance)) if isinstance(tolerance, (int, float)) else 0.0001

        if Topology.IsInstance(topology, "Vertex"):
            return VertexUtility.IsCoincident(vertex, topology, tolerance=tol)

        if Topology.IsInstance(topology, "Face"):
            try:
                from .face import FaceUtility
                return bool(FaceUtility.IsInside(topology, vertex, tol))
            except Exception:
                return False

        if Topology.IsInstance(topology, "Cell"):
            try:
                from .cell import CellUtility
                return CellUtility.Contains(topology, vertex, tol) == 0
            except Exception:
                return False

        if Topology.IsInstance(topology, "CellComplex"):
            try:
                cells = Topology.Cells(topology) or []
                return any(VertexUtility.IsInternal(vertex, cell, tol) for cell in cells)
            except Exception:
                return False

        if Topology.IsInstance(topology, "Cluster"):
            try:
                cells = Topology.Cells(topology) or []
                if any(VertexUtility.IsInternal(vertex, cell, tol) for cell in cells):
                    return True
                faces = Topology.Faces(topology) or []
                if any(VertexUtility.IsInternal(vertex, face, tol) for face in faces):
                    return True
                edges = Topology.Edges(topology) or []
                if any(VertexUtility.IsInternal(vertex, edge, tol) for edge in edges):
                    return True
                vertices = Topology.Vertices(topology) or []
                return any(VertexUtility.IsCoincident(vertex, v, tol) for v in vertices)
            except Exception:
                return False

        if (
            Topology.IsInstance(topology, "Edge")
            or Topology.IsInstance(topology, "Wire")
            or Topology.IsInstance(topology, "Shell")
        ):
            distance = VertexUtility.DistanceToTopology(vertex, topology, tol)
            return distance is not None and distance <= tol

        return False

    @staticmethod
    def SignedDistanceToFace(vertex, face, tolerance: float = 0.0001):
        """
        Returns the signed distance from a Vertex to a planar Face's infinite
        supporting plane. Returns None for a non-planar Face.
        """
        if not isinstance(vertex, Vertex) or not Topology.IsInstance(face, "Face"):
            return None
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane
            from OCC.Core.TopoDS import topods

            shape = _shape_from_topology(face)
            if _is_null_shape(shape):
                return None
            adaptor = BRepAdaptor_Surface(topods.Face(shape), True)
            if adaptor.GetType() != GeomAbs_Plane:
                return None
            plane = adaptor.Plane()
            location = plane.Location()
            normal = plane.Axis().Direction()
            return float(
                (float(vertex.x) - location.X()) * normal.X()
                + (float(vertex.y) - location.Y()) * normal.Y()
                + (float(vertex.z) - location.Z()) * normal.Z()
            )
        except Exception:
            return None

    @staticmethod
    def PerpendicularDistance(vertex, face, tolerance: float = 0.0001):
        """Returns the absolute distance to a planar Face's infinite supporting plane."""
        value = VertexUtility.SignedDistanceToFace(vertex, face, tolerance=tolerance)
        return None if value is None else abs(float(value))

    @staticmethod
    def NearestVertex(vertex, topology, useKDTree=True):
        """
        Returns the discrete Vertex in ``topology`` nearest to ``vertex``.

        ``useKDTree`` is accepted for compatibility with the legacy Core API.
        """
        if not isinstance(vertex, Vertex) or not isinstance(topology, Topology):
            return None
        candidates = []
        topology.Vertices(None, candidates)
        candidates = [v for v in candidates if isinstance(v, Vertex)]
        if not candidates:
            return None
        return min(candidates, key=lambda v: distance3(vertex, v))

    @staticmethod
    def ParameterAtVertex(edge, vertex):
        """Returns the edge parameter corresponding to the input Vertex."""
        from .edge import EdgeUtility
        return EdgeUtility.ParameterAtPoint(edge, vertex)

    @staticmethod
    def AdjacentEdges(vertex, topology, edges):
        from .edge import Edge
        from .graph import Graph
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Graph):
            for edge in topology.edges:
                if same_vertex(edge.start, vertex) or same_vertex(edge.end, vertex):
                    result.append(edge)
        elif isinstance(topology, Topology):
            temp = []
            topology.Edges(None, temp)
            for edge in temp:
                if isinstance(edge, Edge) and (same_vertex(edge.start, vertex) or same_vertex(edge.end, vertex)):
                    result.append(edge)
        edges.extend(unique_by_uuid(result))
        return 0

    @staticmethod
    def AdjacentWires(vertex, topology, wires):
        from .wire import Wire
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Topology):
            temp = []
            Topology.Wires(topology, None, temp)
            for w in temp:
                if not isinstance(w, Wire):
                    continue
                sv, ev = Wire.StartVertex(w), Wire.EndVertex(w)
                if sv is not None and same_vertex(sv, vertex):
                    result.append(w)
                elif ev is not None and same_vertex(ev, vertex):
                    result.append(w)
        wires.extend(unique_by_uuid(result))
        return 0

    @staticmethod
    def AdjacentFaces(vertex, topology, faces):
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Topology):
            temp = []
            Topology.Faces(topology, None, temp)
            for f in temp:
                if not Topology.IsInstance(f, "Face"):
                    continue
                fverts = []
                Topology.Vertices(f, None, fverts)
                if any(same_vertex(v, vertex) for v in fverts):
                    result.append(f)
        faces.extend(unique_by_uuid(result))
        return 0

    @staticmethod
    def AdjacentShells(vertex, topology, shells):
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Topology):
            temp = []
            Topology.Shells(topology, None, temp)
            for s in temp:
                if not Topology.IsInstance(s, "Shell"):
                    continue
                sverts = []
                Topology.Vertices(s, None, sverts)
                if any(same_vertex(sv, vertex) for sv in sverts):
                    result.append(s)
        shells.extend(unique_by_uuid(result))
        return 0

    @staticmethod
    def AdjacentCells(vertex, topology, cells):
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Topology):
            temp = []
            Topology.Cells(topology, None, temp)
            for c in temp:
                if not Topology.IsInstance(c, "Cell"):
                    continue
                cverts = []
                Topology.Vertices(c, None, cverts)
                if any(same_vertex(cv, vertex) for cv in cverts):
                    result.append(c)
        cells.extend(unique_by_uuid(result))
        return 0

    @staticmethod
    def AdjacentCellComplexes(vertex, topology, cellComplexes):
        if not isinstance(vertex, Vertex):
            return 1
        result = []
        if isinstance(topology, Topology):
            temp = []
            Topology.CellComplexes(topology, None, temp)
            for cc in temp:
                if not Topology.IsInstance(cc, "CellComplex"):
                    continue
                ccverts = []
                Topology.Vertices(cc, None, ccverts)
                if any(same_vertex(cvv, vertex) for cvv in ccverts):
                    result.append(cc)
        cellComplexes.extend(unique_by_uuid(result))
        return 0


# ---------------------------------------------------------------------------
# Explicit unsupported Vertex API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _vertex_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Vertex.{name}", return_value)
    return _method


def _vertex_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"VertexUtility.{name}", return_value)
    return _method


Vertex.Origin = staticmethod(lambda: Vertex.ByCoordinates(0.0, 0.0, 0.0))
# Vertex.ByCoordinatesString, Vertex.Project, Vertex.Fuse, VertexUtility.NearestVertex,
# and VertexUtility.ParameterAtVertex have concrete implementations above.
