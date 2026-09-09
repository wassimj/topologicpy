from __future__ import annotations

import math
from dataclasses import dataclass
from .topology import (
    Topology,
    _is_null_shape,
    _downward_wrappers,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    )
from .shell import Shell
from .face import Face, FaceUtility
from .wire import Wire
from .edge import Edge
from .vertex import Vertex
from .occ_utils import make_occ_cell
from .helpers import edge_key, vertex_key, dedupe_vertices_by_distance


def _dedupe_vertices(vertices, tolerance: float = 0.0001):
    """
    Dedupe Vertex wrappers by distance, not OCCT shape-hash: faces built by separate
    Face.ByVertices calls never share OCCT vertex sub-shapes even where coincident, so
    hash-dedup leaves duplicates at shared corners.
    """
    return dedupe_vertices_by_distance((v for v in vertices if isinstance(v, Vertex)), tolerance)


@dataclass(eq=False)
class Cell(Topology):
    def __init__(self, shape=None, dictionary=None, contents=None, contexts=None, apertures=None, shells=None):
        super().__init__(shape=shape, dictionary=dictionary, contents=contents, contexts=contexts, apertures=apertures)
        self.shells = list(shells) if shells else []

    @staticmethod
    def _faces_form_closed_shell(faces, tolerance=0.0001):
        edge_counts = {}
        for face in faces:
            external = getattr(face, "external", None)
            if not isinstance(external, Wire):
                continue
            for edge in getattr(external, "edges", []) or []:
                if not isinstance(edge, Edge):
                    continue
                key = edge_key(edge, tolerance)
                edge_counts[key] = edge_counts.get(key, 0) + 1
        return bool(edge_counts) and all(count == 2 for count in edge_counts.values())

    @staticmethod
    def ByShell(shell, tolerance: float = 0.0001, silent: bool = False):
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Cell.ByShell - Error: The input shell parameter is not a valid topologic shell. Returning None.")
            return None
        occ_cell = make_occ_cell(shell)
        if occ_cell is None:
            if not silent:
                print("Cell.ByShell - Error: Could not create an OpenCascade solid. Returning None.")
            return None
        return Cell(shape=occ_cell, shells=[shell])

    @staticmethod
    def ByFaces(faces, planarize: bool = False, tolerance: float = 0.0001, silent: bool = False):
        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Cell.ByFaces - Error: Could not create a shell from the input faces. Returning None.")
            return None
        return Cell.ByShell(shell, tolerance=tolerance, silent=silent)

    def _native_result(shape, require_cell: bool = True):
        """Wrap a native OCCT result, optionally requiring a Cell."""
        if _is_null_shape(shape):
            return None
        try:
            result = Topology.ByOcctShape(shape)
        except Exception:
            return None
        if require_cell and not isinstance(result, Cell):
            return None
        return result

    def _native_tolerance(tolerance: float = 0.0001):
        """Return a finite positive backend tolerance, or None."""
        try:
            value = abs(float(tolerance))
        except Exception:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _orient_to_direction(cell, origin, direction):
        """Rotate a Cell from the +Z axis to the supplied direction."""
        if not isinstance(cell, Cell) or not isinstance(origin, Vertex):
            return None

        try:
            target = [float(direction[0]), float(direction[1]), float(direction[2])]
        except Exception:
            return None

        target_norm = math.sqrt(sum(value * value for value in target))
        if not math.isfinite(target_norm) or target_norm <= 1.0e-12:
            return None

        target = [value / target_norm for value in target]
        source = [0.0, 0.0, 1.0]

        cross = [
            source[1] * target[2] - source[2] * target[1],
            source[2] * target[0] - source[0] * target[2],
            source[0] * target[1] - source[1] * target[0],
        ]
        sin_angle = math.sqrt(sum(value * value for value in cross))
        cos_angle = sum(source[i] * target[i] for i in range(3))

        if sin_angle <= 1.0e-12:
            if cos_angle > 0.0:
                return cell
            axis = [1.0, 0.0, 0.0]
            angle = 180.0
        else:
            axis = [value / sin_angle for value in cross]
            angle = math.degrees(math.atan2(sin_angle, cos_angle))

        try:
            return cell.Rotate(origin, axis[0], axis[1], axis[2], angle)
        except Exception:
            return None

    def _native_thicken_shape(
        topology,
        thickness: float,
        bothSides: bool,
        reverse: bool,
        tolerance: float,
        silent: bool,
        label: str,
    ):
        """Native normal-offset thickening shared by Face and Shell entry points."""
        tol = Cell._native_tolerance(tolerance)
        try:
            thickness = abs(float(thickness))
        except Exception:
            thickness = float("nan")
        if tol is None or not math.isfinite(thickness) or thickness <= tol:
            if not silent:
                print(f"Cell.{label} - Error: Invalid thickness. Returning None.")
            return None
        shape = getattr(topology, "shape", None)
        if _is_null_shape(shape):
            return None
        signed = -thickness if reverse else thickness
        try:
            from OCC.Core.BRepOffsetAPI import (
                BRepOffsetAPI_MakeOffsetShape,
                BRepOffsetAPI_MakeThickSolid,
            )

            base_shape = shape
            if bothSides:
                offsetter = BRepOffsetAPI_MakeOffsetShape()
                offsetter.PerformBySimple(shape, -0.5 * signed)
                if not offsetter.IsDone():
                    return None
                base_shape = offsetter.Shape()
                if _is_null_shape(base_shape):
                    return None

            thickener = BRepOffsetAPI_MakeThickSolid()
            thickener.MakeThickSolidBySimple(base_shape, signed)
            if not thickener.IsDone():
                return None
            result_shape = thickener.Shape()

            # BRepOffsetAPI_MakeThickSolid can return a geometrically valid solid
            # whose accumulated face orientation is inward.  BRepGProp then
            # reports a negative signed volume, and the inverted material sense
            # can also cause incorrect downstream boolean behaviour.  Normalize
            # the result to an outward-oriented solid before wrapping it.
            try:
                from OCC.Core.BRepGProp import brepgprop
                from OCC.Core.GProp import GProp_GProps

                properties = GProp_GProps()
                brepgprop.VolumeProperties(result_shape, properties)
                signed_volume = float(properties.Mass())
                if math.isfinite(signed_volume) and signed_volume < 0.0:
                    result_shape = result_shape.Reversed()
            except Exception:
                # Orientation normalization is defensive.  If OCCT cannot
                # evaluate the signed volume, preserve the successfully built
                # shape and let normal validation/wrapping decide its fate.
                pass
        except Exception:
            if not silent:
                print(f"Cell.{label} - Error: Native OCCT thickening failed. Returning None.")
            return None
        return Cell._native_result(result_shape)

    def ByCylinder(
        radius: float = 0.5,
        height: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical cylinder centred on the origin and +Z axis."""
        tol = Cell._native_tolerance(tolerance)
        try:
            radius = float(radius)
            height = float(height)
        except Exception:
            radius = height = float("nan")
        if (
            tol is None
            or not math.isfinite(radius)
            or not math.isfinite(height)
            or radius <= tol
            or height <= tol
        ):
            if not silent:
                print("Cell.ByCylinder - Error: Invalid radius or height. Returning None.")
            return None
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

            axis = gp_Ax2(gp_Pnt(0.0, 0.0, -0.5 * height), gp_Dir(0.0, 0.0, 1.0))
            shape = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()
        except Exception:
            if not silent:
                print("Cell.ByCylinder - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    def ByCone(
        baseRadius: float = 0.5,
        topRadius: float = 0.0,
        height: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical cone/frustum centred on the origin and +Z axis."""
        tol = Cell._native_tolerance(tolerance)
        try:
            base_radius = abs(float(baseRadius))
            top_radius = abs(float(topRadius))
            height = float(height)
        except Exception:
            base_radius = top_radius = height = float("nan")
        if (
            tol is None
            or not all(math.isfinite(v) for v in (base_radius, top_radius, height))
            or height <= tol
            or max(base_radius, top_radius) <= tol
        ):
            if not silent:
                print("Cell.ByCone - Error: Invalid radii or height. Returning None.")
            return None
        if abs(base_radius - top_radius) <= tol:
            return Cell.ByCylinder(
                radius=0.5 * (base_radius + top_radius),
                height=height,
                tolerance=tol,
                silent=silent,
            )
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

            axis = gp_Ax2(gp_Pnt(0.0, 0.0, -0.5 * height), gp_Dir(0.0, 0.0, 1.0))
            shape = BRepPrimAPI_MakeCone(axis, base_radius, top_radius, height).Shape()
        except Exception:
            if not silent:
                print("Cell.ByCone - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    def BySphere(
        radius: float = 0.5,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical sphere centred on the origin."""
        tol = Cell._native_tolerance(tolerance)
        try:
            radius = abs(float(radius))
        except Exception:
            radius = float("nan")
        if tol is None or not math.isfinite(radius) or radius <= tol:
            if not silent:
                print("Cell.BySphere - Error: Invalid radius. Returning None.")
            return None
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
            from OCC.Core.gp import gp_Pnt

            shape = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, 0.0), radius).Shape()
        except Exception:
            if not silent:
                print("Cell.BySphere - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    def ByTorus(
        majorRadius: float = 0.5,
        minorRadius: float = 0.125,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical torus centred on the origin and +Z axis."""
        tol = Cell._native_tolerance(tolerance)
        try:
            major_radius = abs(float(majorRadius))
            minor_radius = abs(float(minorRadius))
        except Exception:
            major_radius = minor_radius = float("nan")
        if (
            tol is None
            or not all(math.isfinite(v) for v in (major_radius, minor_radius))
            or major_radius <= tol
            or minor_radius <= tol
            or minor_radius >= major_radius
        ):
            if not silent:
                print("Cell.ByTorus - Error: Invalid major/minor radii. Returning None.")
            return None
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

            axis = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
            shape = BRepPrimAPI_MakeTorus(axis, major_radius, minor_radius).Shape()
        except Exception:
            if not silent:
                print("Cell.ByTorus - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    def ByCapsule(
        radius: float = 0.25,
        height: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical capsule with total extent ``height`` along +Z."""
        tol = Cell._native_tolerance(tolerance)
        try:
            radius = abs(float(radius))
            height = float(height)
        except Exception:
            radius = height = float("nan")
        if (
            tol is None
            or not all(math.isfinite(v) for v in (radius, height))
            or radius <= tol
            or height <= tol
        ):
            if not silent:
                print("Cell.ByCapsule - Error: Invalid radius or height. Returning None.")
            return None

        # Preserve the algorithm-layer definition: when there is no positive
        # cylindrical middle section, the capsule degenerates to a sphere.
        cylinder_height = height - 2.0 * radius
        if cylinder_height <= tol:
            return Cell.BySphere(radius=radius, tolerance=tol, silent=silent)

        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeEdge,
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_MakeWire,
            )
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
            from OCC.Core.GC import GC_MakeArcOfCircle
            from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt

            half_middle = 0.5 * cylinder_height
            inv_sqrt2 = 1.0 / math.sqrt(2.0)

            p_bottom = gp_Pnt(0.0, 0.0, -0.5 * height)
            p_bottom_mid = gp_Pnt(
                radius * inv_sqrt2,
                0.0,
                -half_middle - radius * inv_sqrt2,
            )
            p_bottom_eq = gp_Pnt(radius, 0.0, -half_middle)
            p_top_eq = gp_Pnt(radius, 0.0, half_middle)
            p_top_mid = gp_Pnt(
                radius * inv_sqrt2,
                0.0,
                half_middle + radius * inv_sqrt2,
            )
            p_top = gp_Pnt(0.0, 0.0, 0.5 * height)

            bottom_arc = GC_MakeArcOfCircle(p_bottom, p_bottom_mid, p_bottom_eq).Value()
            top_arc = GC_MakeArcOfCircle(p_top_eq, p_top_mid, p_top).Value()

            edges = [
                BRepBuilderAPI_MakeEdge(bottom_arc).Edge(),
                BRepBuilderAPI_MakeEdge(p_bottom_eq, p_top_eq).Edge(),
                BRepBuilderAPI_MakeEdge(top_arc).Edge(),
                BRepBuilderAPI_MakeEdge(p_top, p_bottom).Edge(),
            ]
            wire_maker = BRepBuilderAPI_MakeWire()
            for edge in edges:
                wire_maker.Add(edge)
            if not wire_maker.IsDone():
                return None
            face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire(), True)
            if not face_maker.IsDone():
                return None
            axis = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
            revol = BRepPrimAPI_MakeRevol(face_maker.Face(), axis, 2.0 * math.pi, True)
            shape = revol.Shape()
        except Exception:
            if not silent:
                print("Cell.ByCapsule - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    def ByThickenedFace(
        face,
        thickness: float = 1.0,
        bothSides: bool = True,
        reverse: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth Cell by normally thickening a Face."""
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Cell.ByThickenedFace - Error: Invalid Face. Returning None.")
            return None
        return Cell._native_thicken_shape(
            face,
            thickness,
            bool(bothSides),
            bool(reverse),
            tolerance,
            silent,
            "ByThickenedFace",
        )

    def ByThickenedShell(
        shell,
        thickness: float = 1.0,
        bothSides: bool = True,
        reverse: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth Cell by normally thickening an open Shell."""
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Cell.ByThickenedShell - Error: Invalid Shell. Returning None.")
            return None
        return Cell._native_thicken_shape(
            shell,
            thickness,
            bool(bothSides),
            bool(reverse),
            tolerance,
            silent,
            "ByThickenedShell",
        )

    def ByOffset(
        cell,
        offset: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Offset a Cell using OCCT's native 3-D offset algorithm."""
        if not isinstance(cell, Cell):
            if not silent:
                print("Cell.ByOffset - Error: Invalid Cell. Returning None.")
            return None
        tol = Cell._native_tolerance(tolerance)
        try:
            offset = float(offset)
        except Exception:
            offset = float("nan")
        if tol is None or not math.isfinite(offset):
            return None
        if abs(offset) <= tol:
            return cell
        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return None
        try:
            from OCC.Core.BRepOffset import BRepOffset_Skin
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
            from OCC.Core.GeomAbs import GeomAbs_Arc

            maker = BRepOffsetAPI_MakeOffsetShape()
            maker.PerformByJoin(
                shape,
                offset,
                tol,
                BRepOffset_Skin,
                False,
                False,
                GeomAbs_Arc,
                True,
            )
            if not maker.IsDone():
                return None
            result_shape = maker.Shape()
        except Exception:
            if not silent:
                print("Cell.ByOffset - Error: Native OCCT offset failed. Returning None.")
            return None
        return Cell._native_result(result_shape, require_cell=False)

    def Shells(self, hostTopology=None, shells=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_SHELL
            )
        else:
            result = list(getattr(self, "shells", []) or [])

        if shells is not None:
            shells.extend(result)
            return 0

        return result

    def Faces(self, hostTopology=None, faces=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_FACE
            )
        else:
            result = []

            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Faces())

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

            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Edges())

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

            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Vertices())

        if vertices is not None:
            vertices.extend(result)
            return 0

        return result

    def Cells(self, hostTopology=None, cells=None):
        result = [self]
        if cells is not None:
            cells.extend(result)
            return 0
        return result


def _cell_by_box(width: float = 1.0, length: float = 1.0, height: float = 1.0,
                  origin=None, direction=None, placement: str = "center", tolerance: float = 0.0001):
    """
    Builds an axis-aligned box Cell directly via BRepPrimAPI_MakeBox, then
    orients/places it. This mirrors the algorithm-layer Cell.Box contract
    (width/length/height + origin + placement), but the algorithm-layer
    Cell.Box (src/topologicpy/Cell.py) actually delegates to Cell.Prism, so
    this backend-level Cell.ByBox only matters to callers that go through
    Core.Cell.ByBox directly.
    """
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt
    except Exception:
        return None

    xOffset = yOffset = zOffset = 0.0
    placement = (placement or "center").lower()
    if placement == "center":
        xOffset = -width * 0.5
        yOffset = -length * 0.5
        zOffset = -height * 0.5
    elif placement == "bottom":
        xOffset = -width * 0.5
        yOffset = -length * 0.5

    ox = origin.x if isinstance(origin, Vertex) else 0.0
    oy = origin.y if isinstance(origin, Vertex) else 0.0
    oz = origin.z if isinstance(origin, Vertex) else 0.0

    try:
        occ_box = BRepPrimAPI_MakeBox(
            gp_Pnt(ox + xOffset, oy + yOffset, oz + zOffset),
            float(width), float(length), float(height),
        ).Shape()
    except Exception:
        return None

    result = Topology.ByOcctShape(occ_box)
    if result is None or direction in (None, [0, 0, 1], (0, 0, 1)):
        return result

    # Reorient from the default [0, 0, 1] up-direction to the requested one.
    try:
        from .topology import Topology as _T
        origin_vertex = origin if isinstance(origin, Vertex) else Vertex.ByCoordinates(ox, oy, oz)
        oriented = _reorient(result, origin_vertex, [0, 0, 1], direction, tolerance)
        if oriented is not None:
            return oriented
    except Exception:
        pass
    return result


def _reorient(topology, origin, dirA, dirB, tolerance=0.0001):
    """
    Small local re-implementation of the rotate-to-align-directions step used
    by Topology.Orient (src/topologicpy/Topology.py), for backend-level
    primitive constructors (Cell.ByBox) that need it without depending on the
    algorithm layer.
    """
    import math

    def _normalize(v):
        n = math.sqrt(sum(c * c for c in v))
        if n == 0:
            return [0.0, 0.0, 1.0]
        return [c / n for c in v]

    a = _normalize(dirA)
    b = _normalize(dirB)
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    sin_a = math.sqrt(sum(c * c for c in cross))
    cos_a = sum(a[i] * b[i] for i in range(3))
    if sin_a < 1e-12:
        if cos_a > 0:
            return topology
        # 180 degree flip: pick any axis perpendicular to a.
        axis = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
        cross = [a[1] * axis[2] - a[2] * axis[1], a[2] * axis[0] - a[0] * axis[2], a[0] * axis[1] - a[1] * axis[0]]
        cross = _normalize(cross)
        angle = 180.0
    else:
        cross = [c / sin_a for c in cross]
        angle = math.degrees(math.atan2(sin_a, cos_a))
    return topology.Rotate(origin, cross[0], cross[1], cross[2], angle)


Cell.ByBox = staticmethod(_cell_by_box)


def _cell_by_wires(wires, close: bool = False, tolerance: float = 0.0001, silent: bool = False):
    """Create an exact curve-preserving solid loft through closed section Wires."""
    if close:
        return None
    if not isinstance(wires, (list, tuple)):
        return None
    wire_list = [wire for wire in wires if isinstance(wire, Wire)]
    if len(wire_list) < 2:
        return None
    try:
        tolerance = abs(float(tolerance))
    except Exception:
        return None
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        return None
    try:
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
    except Exception:
        return None
    occ_wires = []
    expected_edge_count = None
    for wire in wire_list:
        shape = getattr(wire, "shape", None)
        if _is_null_shape(shape):
            return None
        try:
            occ_wire = topods.Wire(shape)
        except Exception:
            return None
        if _is_null_shape(occ_wire):
            return None
        try:
            explorer = TopExp_Explorer(occ_wire, TopAbs_EDGE)
            edge_count = 0
            while explorer.More():
                edge_count += 1
                explorer.Next()
        except Exception:
            return None
        if edge_count < 1:
            return None
        if expected_edge_count is None:
            expected_edge_count = edge_count
        elif edge_count != expected_edge_count:
            return None
        occ_wires.append(occ_wire)
    try:
        loft = BRepOffsetAPI_ThruSections(True, True, tolerance)
        loft.CheckCompatibility(False)
        for occ_wire in occ_wires:
            loft.AddWire(occ_wire)
        loft.Build()
        if not loft.IsDone():
            return None
        shape = loft.Shape()
    except Exception:
        return None
    if _is_null_shape(shape):
        return None
    try:
        result = Topology.ByOcctShape(shape)
    except Exception:
        return None
    return result if isinstance(result, Cell) else None


Cell.ByWires = staticmethod(_cell_by_wires)


def _cell_internal_vertex(cell, tolerance: float = 0.0001):
    return CellUtility.InternalVertex(cell, tolerance=tolerance)


Cell.InternalVertex = _cell_internal_vertex


class CellUtility:
    @staticmethod
    def Volume(cell):
        """
        Returns the volume of the input Cell via OCCT's volume properties
        (same brepgprop.VolumeProperties call already used successfully by
        Topology.CenterOfMass in topology.py for Cell/CellComplex/Cluster
        shapes).
        """
        if not isinstance(cell, Cell):
            return None
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
            return None
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            return props.Mass()
        except Exception:
            return None

    @staticmethod
    def Contains(cell, vertex, tolerance: float = 0.0001):
        """
        Classify a vertex against the cell via BRepClass3d_SolidClassifier
        (0=inside, 1=on-boundary, 2=outside). Cells are boundary-representation
        shells, so the classifier returns ON for any point on/within the shell;
        the algorithm layer's Cell.ContainmentStatus compensates with 8 offset
        vertices + majority vote.
        """
        if not isinstance(cell, Cell) or not isinstance(vertex, Vertex):
            return 2
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
            return 2
        try:
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
            classifier = BRepClass3d_SolidClassifier(shape)
            classifier.Perform(gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)), float(tolerance))
            state = classifier.State()
            if state == TopAbs_IN:
                return 0
            if state == TopAbs_ON:
                return 1
            return 2
        except Exception:
            return 2

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """
        Vertex strictly inside the Cell: try CenterOfMass, confirm via
        CellUtility.Contains; for non-convex cells whose centroid is outside or on
        the boundary, sample a small grid inside the bounding box until one tests
        strictly inside.
        """
        if not isinstance(cell, Cell):
            return None
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
            return None

        center = Topology.CenterOfMass(cell)
        if isinstance(center, Vertex) and CellUtility.Contains(cell, center, tolerance) == 0:
            return center

        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
            box = Bnd_Box()
            brepbndlib.Add(shape, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        except Exception:
            return center if isinstance(center, Vertex) else None

        steps = 6
        for i in range(1, steps):
            for j in range(1, steps):
                for k in range(1, steps):
                    x = xmin + (xmax - xmin) * i / steps
                    y = ymin + (ymax - ymin) * j / steps
                    z = zmin + (zmax - zmin) * k / steps
                    candidate = Vertex.ByCoordinates(x, y, z)
                    if CellUtility.Contains(cell, candidate, tolerance) == 0:
                        return candidate

        return center if isinstance(center, Vertex) else None

# ---------------------------------------------------------------------------
# Explicit unsupported Cell API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _cell_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Cell.{name}", return_value)
    return _method


def _cell_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"CellUtility.{name}", return_value)
    return _method


def _make_adjacent(method_name):
    """Return a staticmethod that delegates to topology.method(hostTopology, output)."""
    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)
    return _impl

CellUtility.AdjacentVertices = _make_adjacent("Vertices")
CellUtility.AdjacentEdges = _make_adjacent("Edges")
CellUtility.AdjacentWires = _make_adjacent("Wires")
CellUtility.AdjacentFaces = _make_adjacent("Faces")
CellUtility.AdjacentShells = _make_adjacent("Shells")
CellUtility.AdjacentCells = _make_adjacent("Cells")
CellUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

# Cell.ByBox, Cell.ByWires, Cell.InternalVertex, CellUtility.Volume,
# CellUtility.Contains, and CellUtility.InternalVertex are implemented above
# -- do not clobber them here.
