from __future__ import annotations

from dataclasses import dataclass
import math

from .topology import Topology, _shape_from_topology, _is_null_shape
from .occ_utils import make_occ_vertex
from .helpers import distance3, same_vertex, unique_by_uuid


def _vertex_tolerance(tolerance=0.0001):
    """Returns a safe positive geometric tolerance."""
    try:
        value = abs(float(tolerance))
    except Exception:
        return 0.0001

    if not math.isfinite(value):
        return 0.0001

    return max(value, 1.0e-12)


def _adjacent_topologies_by_vertex(
    vertex,
    topology,
    collection_name,
    topology_type,
):
    """
    Returns topologies of the requested type that contain the input Vertex.

    Membership is determined geometrically using the backend vertex tolerance
    semantics rather than Python wrapper identity.
    """
    if not isinstance(vertex, Vertex) or not isinstance(topology, Topology):
        return []

    getter = getattr(
        Topology,
        collection_name,
        None,
    )

    if not callable(getter):
        return []

    candidates = []

    try:
        getter(
            topology,
            None,
            candidates,
        )
    except Exception:
        return []

    result = []

    for candidate in candidates:
        if not Topology.IsInstance(
            candidate,
            topology_type,
        ):
            continue

        vertices = []

        try:
            Topology.Vertices(
                candidate,
                None,
                vertices,
            )
        except Exception:
            continue

        if any(
            isinstance(candidate_vertex, Vertex)
            and same_vertex(candidate_vertex, vertex)
            for candidate_vertex in vertices
        ):
            result.append(candidate)

    return unique_by_uuid(result)


@dataclass(eq=False)
class Vertex(Topology):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def ByCoordinates(
        x=0.0,
        y=0.0,
        z=0.0
    ):
        """Creates a backend Vertex from Cartesian coordinates."""
        try:
            x = float(x)
            y = float(y)
            z = float(z)
        except Exception:
            return None

        if not all(
            math.isfinite(value)
            for value in (x, y, z)
        ):
            return None

        try:
            shape = make_occ_vertex(
                x,
                y,
                z,
            )
        except Exception:
            return None

        if _is_null_shape(shape):
            return None

        return Vertex(
            shape=shape,
            x=x,
            y=y,
            z=z,
        )

    @staticmethod
    def Origin():
        """Returns the global origin Vertex."""
        return Vertex.ByCoordinates(
            0.0,
            0.0,
            0.0,
        )

    @staticmethod
    def ByPoint(point):
        try:
            return Vertex.ByCoordinates(point.X(), point.Y(), point.Z())
        except Exception:
            return None

    @staticmethod
    def ByOcctShape(
        shape,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None
    ):
        """Wraps an existing OCCT Vertex."""
        if _is_null_shape(shape):
            return None

        try:
            from OCC.Core.BRep import BRep_Tool

            point = BRep_Tool.Pnt(
                shape
            )

            x = float(point.X())
            y = float(point.Y())
            z = float(point.Z())

        except Exception:
            return None

        return Vertex(
            shape=shape,
            x=x,
            y=y,
            z=z,
            dictionary=dictionary,
            contents=list(contents) if contents else [],
            contexts=list(contexts) if contexts else [],
            apertures=list(apertures) if apertures else [],
        )

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

        tol = _vertex_tolerance(tolerance)

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

        tol = _vertex_tolerance(tolerance)

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
    def AdjacentWires(
        vertex,
        topology,
        wires
    ):
        """Populates wires with Wires that contain the input Vertex."""
        if not isinstance(vertex, Vertex):
            return 1

        result = _adjacent_topologies_by_vertex(
            vertex,
            topology,
            "Wires",
            "Wire",
        )

        wires.extend(result)
        return 0

    @staticmethod
    def AdjacentFaces(
        vertex,
        topology,
        faces
    ):
        """Populates faces with Faces that contain the input Vertex."""
        if not isinstance(vertex, Vertex):
            return 1

        result = _adjacent_topologies_by_vertex(
            vertex,
            topology,
            "Faces",
            "Face",
        )

        faces.extend(result)
        return 0

    @staticmethod
    def AdjacentShells(
        vertex,
        topology,
        shells
    ):
        """Populates shells with Shells that contain the input Vertex."""
        if not isinstance(vertex, Vertex):
            return 1

        result = _adjacent_topologies_by_vertex(
            vertex,
            topology,
            "Shells",
            "Shell",
        )

        shells.extend(result)
        return 0

    @staticmethod
    def AdjacentCells(
        vertex,
        topology,
        cells
    ):
        """Populates cells with Cells that contain the input Vertex."""
        if not isinstance(vertex, Vertex):
            return 1

        result = _adjacent_topologies_by_vertex(
            vertex,
            topology,
            "Cells",
            "Cell",
        )

        cells.extend(result)
        return 0

    @staticmethod
    def AdjacentCellComplexes(
        vertex,
        topology,
        cellComplexes
    ):
        """Populates cellComplexes with CellComplexes that contain the input Vertex."""
        if not isinstance(vertex, Vertex):
            return 1

        result = _adjacent_topologies_by_vertex(
            vertex,
            topology,
            "CellComplexes",
            "CellComplex",
        )

        cellComplexes.extend(result)
        return 0
