from __future__ import annotations

from dataclasses import dataclass
import math

from .topology import Topology, _shape_from_topology, _is_null_shape
from .vertex import Vertex
from .edge import Edge
from .wire import Wire


def _face_tolerance(value=0.0001) -> float:
    """Return a finite positive geometric tolerance."""
    try:
        value = abs(float(value))
    except Exception:
        value = 0.0001
    if not math.isfinite(value) or value <= 0.0:
        return 1.0e-12
    return value


def _as_occ_face(face):
    """Return an OCCT TopoDS_Face for a backend Face, or None."""
    if not isinstance(face, Face):
        return None
    shape = _shape_from_topology(face)
    if _is_null_shape(shape):
        return None
    try:
        from OCC.Core.TopoDS import topods
        return topods.Face(shape)
    except Exception:
        return None


def _as_occ_wire(wire):
    """Return an OCCT TopoDS_Wire for a backend Wire, or None."""
    if not isinstance(wire, Wire):
        return None
    shape = _shape_from_topology(wire)
    if _is_null_shape(shape):
        return None
    try:
        from OCC.Core.TopoDS import topods
        return topods.Wire(shape)
    except Exception:
        return None


def _same_shape(shape_a, shape_b) -> bool:
    """Return True when two OCCT shapes reference the same topological entity."""
    if shape_a is None or shape_b is None:
        return False
    try:
        return bool(shape_a.IsSame(shape_b))
    except Exception:
        return False


def _explore_shapes(shape, shape_type):
    """Return OCCT subshapes of the requested type, preserving explorer order."""
    if _is_null_shape(shape):
        return []
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        explorer = TopExp_Explorer(shape, shape_type)
    except Exception:
        return []

    result = []
    while explorer.More():
        current = explorer.Current()
        if not any(_same_shape(current, existing) for existing in result):
            result.append(current)
        explorer.Next()
    return result


def _wire_area(occ_wire) -> float | None:
    """Return the unsigned planar area enclosed by an OCCT wire when possible."""
    if occ_wire is None:
        return None
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop

        maker = BRepBuilderAPI_MakeFace(occ_wire, True)
        if not maker.IsDone():
            return None
        props = GProp_GProps()
        brepgprop.SurfaceProperties(maker.Face(), props)
        value = abs(float(props.Mass()))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _outer_wire_shape(face):
    """Return the native external-boundary wire of a backend Face."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return None

    # OCCT's dedicated outer-wire query is both exact and very fast.  The
    # PythonOCC spelling changed across releases, so support both forms.
    try:
        from OCC.Core.BRepTools import BRepTools
        for name in ("OuterWire_s", "OuterWire"):
            fn = getattr(BRepTools, name, None)
            if callable(fn):
                wire = fn(occ_face)
                if wire is not None and not wire.IsNull():
                    return wire
    except Exception:
        pass

    try:
        from OCC.Core.BRepTools import breptools
        fn = getattr(breptools, "OuterWire", None)
        if callable(fn):
            wire = fn(occ_face)
            if wire is not None and not wire.IsNull():
                return wire
    except Exception:
        pass

    # Defensive fallback: for a valid planar face the outer loop has the
    # greatest enclosed area.  This is used only if the dedicated API is not
    # exposed by the installed PythonOCC build.
    try:
        from OCC.Core.TopAbs import TopAbs_WIRE
        from OCC.Core.TopoDS import topods
        candidates = [topods.Wire(s) for s in _explore_shapes(occ_face, TopAbs_WIRE)]
    except Exception:
        return None

    best = None
    best_area = -1.0
    for wire in candidates:
        area = _wire_area(wire)
        if area is not None and area > best_area:
            best = wire
            best_area = area
    return best


def _internal_wire_shapes(face):
    """Return native hole wires of a backend Face."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return []
    outer = _outer_wire_shape(face)
    try:
        from OCC.Core.TopAbs import TopAbs_WIRE
        from OCC.Core.TopoDS import topods
        wires = [topods.Wire(s) for s in _explore_shapes(occ_face, TopAbs_WIRE)]
    except Exception:
        return []
    if outer is None:
        return wires[1:] if len(wires) > 1 else []
    return [wire for wire in wires if not _same_shape(wire, outer)]


def _surface_and_bounds(face):
    """Return (surface, u0, u1, v0, v1) for a backend Face."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return None
    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

        surface = BRep_Tool.Surface(occ_face)
        if surface is None:
            return None
        adaptor = BRepAdaptor_Surface(occ_face, True)
        u0 = float(adaptor.FirstUParameter())
        u1 = float(adaptor.LastUParameter())
        v0 = float(adaptor.FirstVParameter())
        v1 = float(adaptor.LastVParameter())
        values = (u0, u1, v0, v1)
        if not all(math.isfinite(value) for value in values):
            return None
        return surface, u0, u1, v0, v1
    except Exception:
        return None


def _normalized_to_raw(face, u, v):
    """Map TopologicPy normalized UV parameters to native surface parameters."""
    data = _surface_and_bounds(face)
    if data is None:
        return None
    surface, u0, u1, v0, v1 = data
    try:
        un = float(u)
        vn = float(v)
    except Exception:
        return None
    raw_u = u0 + un * (u1 - u0)
    raw_v = v0 + vn * (v1 - v0)
    return surface, raw_u, raw_v, u0, u1, v0, v1


def _raw_to_normalized(face, raw_u, raw_v):
    """Map native surface parameters to TopologicPy normalized UV parameters."""
    data = _surface_and_bounds(face)
    if data is None:
        return None
    _, u0, u1, v0, v1 = data
    du = u1 - u0
    dv = v1 - v0
    if abs(du) <= 1.0e-30 or abs(dv) <= 1.0e-30:
        return None
    return [(float(raw_u) - u0) / du, (float(raw_v) - v0) / dv]


def _wrap_metadata(source, result):
    """Copy wrapper-level metadata from source to result when possible."""
    if result is None or source is None:
        return result
    for name in ("dictionary", "contents", "contexts", "apertures"):
        try:
            value = getattr(source, name)
            if name in ("contents", "contexts", "apertures"):
                value = list(value) if value else []
            setattr(result, name, value)
        except Exception:
            pass
    return result


@dataclass(eq=False)
class Face(Topology):
    """PythonOCC backend wrapper for an OCCT face."""

    @staticmethod
    def ByExternalBoundary(externalBoundary, tolerance: float = 0.0001):
        """Create a Face from one closed external-boundary Wire."""
        occ_wire = _as_occ_wire(externalBoundary)
        if occ_wire is None:
            return None
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            maker = BRepBuilderAPI_MakeFace(occ_wire, True)
            if not maker.IsDone():
                return None
            occ_face = maker.Face()
            if occ_face is None or occ_face.IsNull():
                return None
            return Face(shape=occ_face)
        except Exception:
            return None

    @staticmethod
    def ByExternalInternalBoundaries(
        externalBoundary,
        internalBoundaries=None,
        tolerance: float = 0.0001,
    ):
        """Create a Face from an external Wire and zero or more hole Wires."""
        occ_outer = _as_occ_wire(externalBoundary)
        if occ_outer is None:
            return None

        holes = [wire for wire in (internalBoundaries or []) if isinstance(wire, Wire)]
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.TopoDS import topods

            maker = BRepBuilderAPI_MakeFace(occ_outer, True)
            if not maker.IsDone():
                return None

            outer_orientation = None
            try:
                outer_orientation = occ_outer.Orientation()
            except Exception:
                pass

            for hole in holes:
                occ_hole = _as_occ_wire(hole)
                if occ_hole is None:
                    continue

                # Inner loops must oppose the external loop.  Use Reversed()
                # rather than mutating the source wire so callers retain their
                # original topology and metadata.
                try:
                    if outer_orientation is None or occ_hole.Orientation() == outer_orientation:
                        occ_hole = topods.Wire(occ_hole.Reversed())
                except Exception:
                    try:
                        occ_hole = topods.Wire(occ_hole.Reversed())
                    except Exception:
                        pass

                maker.Add(occ_hole)

            if not maker.IsDone():
                return None
            occ_face = maker.Face()
            if occ_face is None or occ_face.IsNull():
                return None
            return Face(shape=occ_face)
        except Exception:
            return None

    @staticmethod
    def ByOcctShape(shape, dictionary=None, contents=None, contexts=None, apertures=None):
        """Wrap an existing OCCT face without rebuilding its geometry."""
        if _is_null_shape(shape):
            return None
        try:
            from OCC.Core.TopoDS import topods
            occ_face = topods.Face(shape)
            if occ_face.IsNull():
                return None
        except Exception:
            return None
        face = Face(shape=occ_face)
        face.dictionary = dictionary
        face.contents = list(contents) if contents else []
        face.contexts = list(contexts) if contexts else []
        face.apertures = list(apertures) if apertures else []
        return face

    def ExternalBoundary(self):
        """Return the external boundary Wire of this Face."""
        outer = _outer_wire_shape(self)
        if outer is None:
            return None
        return Wire.ByOcctShape(outer)

    def InternalBoundaries(self, output=None):
        """Return or populate the internal boundary Wires of this Face."""
        result = [Wire.ByOcctShape(shape) for shape in _internal_wire_shapes(self)]
        result = [wire for wire in result if isinstance(wire, Wire)]
        if output is not None:
            output.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, output=None):
        """Return or populate all boundary Wires of this Face."""
        result = []
        outer = self.ExternalBoundary()
        if isinstance(outer, Wire):
            result.append(outer)
        result.extend(self.InternalBoundaries() or [])
        if output is not None:
            output.extend(result)
            return 0
        return result

    def Wire(self):
        """Alias for ExternalBoundary."""
        return self.ExternalBoundary()

    def Edges(self, hostTopology=None, output=None):
        """Return or populate all unique Edges of this Face."""
        occ_face = _as_occ_face(self)
        result = []
        if occ_face is not None:
            try:
                from OCC.Core.TopAbs import TopAbs_EDGE
                from OCC.Core.TopoDS import topods
                for shape in _explore_shapes(occ_face, TopAbs_EDGE):
                    edge = Edge.ByOcctShape(topods.Edge(shape))
                    if edge is not None:
                        result.append(edge)
            except Exception:
                result = []
        if output is not None:
            output.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, output=None):
        """Return or populate all unique Vertices of this Face."""
        occ_face = _as_occ_face(self)
        result = []
        if occ_face is not None:
            try:
                from OCC.Core.TopAbs import TopAbs_VERTEX
                from OCC.Core.TopoDS import topods
                for shape in _explore_shapes(occ_face, TopAbs_VERTEX):
                    vertex = Vertex.ByOcctShape(topods.Vertex(shape))
                    if vertex is not None:
                        result.append(vertex)
            except Exception:
                result = []
        if output is not None:
            output.extend(result)
            return 0
        return result


class FaceUtility:
    """OCCT-native utility namespace matching TopologicCore's FaceUtility API."""

    @staticmethod
    def Area(face):
        """Return the exact OCCT surface area of a Face."""
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            props = GProp_GProps()
            brepgprop.SurfaceProperties(occ_face, props)
            value = float(props.Mass())
            return value if math.isfinite(value) else None
        except Exception:
            return None

    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5, tolerance: float = 0.0001):
        """Return the oriented unit surface normal at normalized UV parameters."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None
        surface, raw_u, raw_v, _, _, _, _ = mapped
        tol = _face_tolerance(tolerance)
        try:
            from OCC.Core.GeomLProp import GeomLProp_SLProps
            from OCC.Core.TopAbs import TopAbs_REVERSED

            props = GeomLProp_SLProps(surface, raw_u, raw_v, 1, tol)
            if not props.IsNormalDefined():
                return None
            normal = props.Normal()
            result = [float(normal.X()), float(normal.Y()), float(normal.Z())]
            occ_face = _as_occ_face(face)
            if occ_face is not None and occ_face.Orientation() == TopAbs_REVERSED:
                result = [-result[0], -result[1], -result[2]]
            length = math.sqrt(sum(value * value for value in result))
            if length <= tol:
                return None
            return [value / length for value in result]
        except Exception:
            return None

    @staticmethod
    def VertexAtParameters(face, u=0.5, v=0.5):
        """Return a Vertex at normalized UV parameters on the Face surface."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None
        surface, raw_u, raw_v, _, _, _, _ = mapped
        try:
            pnt = surface.Value(raw_u, raw_v)
            return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())
        except Exception:
            return None

    @staticmethod
    def ParametersAtVertex(face, vertex, tolerance: float = 0.0001):
        """Return normalized UV parameters of a Vertex on the Face surface."""
        if not isinstance(vertex, Vertex):
            return None
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.gp import gp_Pnt

            surface = BRep_Tool.Surface(occ_face)
            if surface is None:
                return None
            point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            projector = GeomAPI_ProjectPointOnSurf(point, surface)
            if projector.NbPoints() < 1:
                return None
            if float(projector.LowerDistance()) > _face_tolerance(tolerance):
                return None
            raw_u, raw_v = projector.LowerDistanceParameters()
            return _raw_to_normalized(face, raw_u, raw_v)
        except Exception:
            return None

    @staticmethod
    def IsInside(face, vertex, tolerance: float = 0.0001):
        """Return True when a Vertex lies in or on the trimmed Face."""
        if not isinstance(vertex, Vertex):
            return False
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return False
        tol = _face_tolerance(tolerance)
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt, gp_Pnt2d

            surface = BRep_Tool.Surface(occ_face)
            point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            projector = GeomAPI_ProjectPointOnSurf(point, surface)
            if projector.NbPoints() < 1 or float(projector.LowerDistance()) > tol:
                return False
            raw_u, raw_v = projector.LowerDistanceParameters()
            classifier = BRepTopAdaptor_FClass2d(occ_face, tol)
            state = classifier.Perform(gp_Pnt2d(float(raw_u), float(raw_v)))
            return state in (TopAbs_IN, TopAbs_ON)
        except Exception:
            return False

    @staticmethod
    def InternalVertex(face, tolerance: float = 0.0001):
        """Return a deterministic Vertex strictly inside the trimmed Face."""
        occ_face = _as_occ_face(face)
        data = _surface_and_bounds(face)
        if occ_face is None or data is None:
            return None
        surface, u0, u1, v0, v1 = data
        tol = _face_tolerance(tolerance)

        try:
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.TopAbs import TopAbs_IN
            from OCC.Core.gp import gp_Pnt2d
            classifier = BRepTopAdaptor_FClass2d(occ_face, tol)
        except Exception:
            return None

        # Centre first, then an increasingly fine deterministic UV lattice.
        candidates = [(0.5, 0.5)]
        for denominator in (4, 8, 16, 32):
            for i in range(1, denominator):
                for j in range(1, denominator):
                    candidates.append((i / denominator, j / denominator))

        seen = set()
        for un, vn in candidates:
            key = (round(un, 12), round(vn, 12))
            if key in seen:
                continue
            seen.add(key)
            raw_u = u0 + un * (u1 - u0)
            raw_v = v0 + vn * (v1 - v0)
            try:
                state = classifier.Perform(gp_Pnt2d(raw_u, raw_v))
                if state != TopAbs_IN:
                    continue
                pnt = surface.Value(raw_u, raw_v)
                return Vertex.ByCoordinates(pnt.X(), pnt.Y(), pnt.Z())
            except Exception:
                continue
        return None

    @staticmethod
    def IsCoplanar(faceA, faceB, tolerance: float = 0.0001):
        """Return True when two Faces lie on the same native OCCT plane."""
        occ_a = _as_occ_face(faceA)
        occ_b = _as_occ_face(faceB)
        if occ_a is None or occ_b is None:
            return None
        tol = _face_tolerance(tolerance)
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane

            adaptor_a = BRepAdaptor_Surface(occ_a, True)
            adaptor_b = BRepAdaptor_Surface(occ_b, True)
            if adaptor_a.GetType() != GeomAbs_Plane or adaptor_b.GetType() != GeomAbs_Plane:
                return False
            plane_a = adaptor_a.Plane()
            plane_b = adaptor_b.Plane()
            normal_a = plane_a.Axis().Direction()
            normal_b = plane_b.Axis().Direction()
            ax, ay, az = normal_a.X(), normal_a.Y(), normal_a.Z()
            bx, by, bz = normal_b.X(), normal_b.Y(), normal_b.Z()
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            if math.sqrt(cx * cx + cy * cy + cz * cz) > tol:
                return False
            location_a = plane_a.Location()
            location_b = plane_b.Location()
            dx = location_b.X() - location_a.X()
            dy = location_b.Y() - location_a.Y()
            dz = location_b.Z() - location_a.Z()
            distance = abs(dx * ax + dy * ay + dz * az)
            return distance <= tol
        except Exception:
            return None

    @staticmethod
    def Reverse(face):
        """Return the same native Face with its orientation reversed."""
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None
        try:
            from OCC.Core.TopoDS import topods
            reversed_face = topods.Face(occ_face.Reversed())
            return _wrap_metadata(face, Face.ByOcctShape(reversed_face))
        except Exception:
            return None

    @staticmethod
    def Triangulate(face, deflection=0.1, output=None):
        """Triangulate a Face natively with OCCT and return/populate triangle Faces."""
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None if output is None else 0
        try:
            deflection = abs(float(deflection))
        except Exception:
            deflection = 0.1
        if deflection <= 0.0:
            deflection = 1.0e-4

        triangles = []
        try:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
            from OCC.Core.TopAbs import TopAbs_REVERSED
            from OCC.Core.TopLoc import TopLoc_Location

            mesher = BRepMesh_IncrementalMesh(occ_face, deflection, False, 0.5, True)
            try:
                mesher.Perform()
            except Exception:
                pass

            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(occ_face, location)
            if triangulation is None or triangulation.NbTriangles() < 1:
                return [] if output is None else 0
            transform = location.Transformation()
            reversed_orientation = occ_face.Orientation() == TopAbs_REVERSED

            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                indices = triangle.Get()
                if len(indices) != 3:
                    continue
                n1, n2, n3 = indices
                if reversed_orientation:
                    n2, n3 = n3, n2
                points = []
                for index in (n1, n2, n3):
                    pnt = triangulation.Node(index)
                    try:
                        pnt = pnt.Transformed(transform)
                    except Exception:
                        try:
                            pnt.Transform(transform)
                        except Exception:
                            pass
                    points.append(pnt)

                polygon = BRepBuilderAPI_MakePolygon()
                for pnt in points:
                    polygon.Add(pnt)
                polygon.Close()
                if not polygon.IsDone():
                    continue
                maker = BRepBuilderAPI_MakeFace(polygon.Wire(), True)
                if not maker.IsDone():
                    continue
                triangle_face = Face.ByOcctShape(maker.Face())
                if triangle_face is not None:
                    triangles.append(triangle_face)
        except Exception:
            triangles = []

        if output is not None:
            output.extend(triangles)
            return 0
        return triangles

    @staticmethod
    def TrimByWire(face, wire, reverse: bool = False, tolerance: float = 0.0001):
        """Trim a Face by a closed Wire using native OCCT face booleans."""
        occ_face = _as_occ_face(face)
        occ_wire = _as_occ_wire(wire)
        if occ_face is None or occ_wire is None:
            return None
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopoDS import topods

            tool_maker = BRepBuilderAPI_MakeFace(occ_wire, True)
            if not tool_maker.IsDone():
                return None
            tool_face = tool_maker.Face()
            operation = BRepAlgoAPI_Cut(occ_face, tool_face) if reverse else BRepAlgoAPI_Common(occ_face, tool_face)
            operation.Build()
            if not operation.IsDone():
                return None
            shape = operation.Shape()
            faces = _explore_shapes(shape, TopAbs_FACE)
            if len(faces) != 1:
                return None
            return Face.ByOcctShape(topods.Face(faces[0]))
        except Exception:
            return None
