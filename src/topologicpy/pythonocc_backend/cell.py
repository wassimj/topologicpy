from __future__ import annotations

import math
from dataclasses import dataclass

from .occ_utils import make_occ_cell
from .shell import Shell
from .topology import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_VERTEX,
    Topology,
    _downward_wrappers,
    _is_null_shape,
)
from .vertex import Vertex
from .wire import Wire


@dataclass(eq=False)
class Cell(Topology):
    def __init__(
        self,
        shape=None,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
        shells=None,
    ):
        super().__init__(
            shape=shape,
            dictionary=dictionary,
            contents=contents,
            contexts=contexts,
            apertures=apertures,
        )
        self.shells = list(shells) if shells else []

    @staticmethod
    def ByShell(shell, tolerance: float = 0.0001, silent: bool = False):
        """Create a Cell from a closed Shell."""
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print(
                    "Cell.ByShell - Error: The input shell parameter is not a valid "
                    "topologic shell. Returning None."
                )
            return None

        occ_cell = make_occ_cell(shell)
        if occ_cell is None or _is_null_shape(occ_cell):
            if not silent:
                print(
                    "Cell.ByShell - Error: Could not create an OpenCascade solid. "
                    "Returning None."
                )
            return None

        return Cell(shape=occ_cell, shells=[shell])

    @staticmethod
    def ByFaces(
        faces,
        planarize: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a Cell from a collection of Faces."""
        # ``planarize`` is intentionally retained for backend-call compatibility.
        # It is not applied here: the PythonOCC backend preserves the supplied
        # Face geometry and delegates sewing to Shell.ByFaces.
        _ = planarize

        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print(
                    "Cell.ByFaces - Error: Could not create a shell from the input "
                    "faces. Returning None."
                )
            return None

        return Cell.ByShell(shell, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByWires(wires, tolerance: float = 0.0001):
        """
        Create a curve-preserving Cell by lofting through closed section Wires.

        The section curves are passed directly to OCCT. Corresponding Wires must
        contain the same number of Edges; the backend does not resample or rebuild
        mismatched sections.

        Parameters
        ----------
        wires : list
            Ordered closed section Wires.
        tolerance : float , optional
            Geometric tolerance. Default is 0.0001.

        Returns
        -------
        Cell
            The resulting OCCT solid, or None if construction fails.
        """
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
            loft = BRepOffsetAPI_ThruSections(
                True,   # isSolid
                True,   # ruled
                tolerance,
            )
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

    @staticmethod
    def ByBox(
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        origin=None,
        direction=None,
        placement: str = "center",
        tolerance: float = 0.0001,
    ):
        """Create an OCCT box Cell and optionally orient it to a direction."""
        try:
            width = float(width)
            length = float(length)
            height = float(height)
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if (
            not all(math.isfinite(value) for value in (width, length, height, tolerance))
            or width <= tolerance
            or length <= tolerance
            or height <= tolerance
            or tolerance <= 0.0
        ):
            return None

        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.gp import gp_Pnt
        except Exception:
            return None

        placement = str(placement or "center").lower()
        x_offset = y_offset = z_offset = 0.0

        if placement == "center":
            x_offset = -0.5 * width
            y_offset = -0.5 * length
            z_offset = -0.5 * height
        elif placement == "bottom":
            x_offset = -0.5 * width
            y_offset = -0.5 * length

        ox = float(origin.x) if isinstance(origin, Vertex) else 0.0
        oy = float(origin.y) if isinstance(origin, Vertex) else 0.0
        oz = float(origin.z) if isinstance(origin, Vertex) else 0.0

        try:
            shape = BRepPrimAPI_MakeBox(
                gp_Pnt(ox + x_offset, oy + y_offset, oz + z_offset),
                width,
                length,
                height,
            ).Shape()
        except Exception:
            return None

        if _is_null_shape(shape):
            return None

        result = Topology.ByOcctShape(shape)
        if not isinstance(result, Cell):
            return None

        if direction is None:
            return result

        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
        except Exception:
            return result

        if direction == [0.0, 0.0, 1.0]:
            return result

        origin_vertex = origin if isinstance(origin, Vertex) else Vertex.ByCoordinates(ox, oy, oz)
        oriented = Cell._orient_to_direction(result, origin_vertex, direction)
        return oriented if isinstance(oriented, Cell) else result

    @staticmethod
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

    @staticmethod
    def _native_tolerance(tolerance: float = 0.0001):
        """Return a finite positive backend tolerance, or None."""
        try:
            value = abs(float(tolerance))
        except Exception:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def ByCHS(
        radius: float = 1.0,
        height: float = 1.0,
        thickness: float = 0.25,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth canonical circular hollow section centred on the origin."""
        tol = Cell._native_tolerance(tolerance)
        try:
            radius = abs(float(radius))
            height = float(height)
            thickness = abs(float(thickness))
        except Exception:
            radius = height = thickness = float("nan")
        if (
            tol is None
            or not all(math.isfinite(v) for v in (radius, height, thickness))
            or radius <= tol
            or height <= tol
            or thickness <= tol
            or thickness >= radius - tol
        ):
            if not silent:
                print("Cell.ByCHS - Error: Invalid dimensions. Returning None.")
            return None
        try:
            from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
            from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

            axis = gp_Ax2(gp_Pnt(0.0, 0.0, -0.5 * height), gp_Dir(0.0, 0.0, 1.0))
            outer = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()
            inner = BRepPrimAPI_MakeCylinder(axis, radius - thickness, height).Shape()
            cut = BRepAlgoAPI_Cut(outer, inner)
            cut.Build()
            if not cut.IsDone():
                return None
            shape = cut.Shape()
        except Exception:
            if not silent:
                print("Cell.ByCHS - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    @staticmethod
    def ByRHS(
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        thickness: float = 0.25,
        outerRadius: float = 0.0,
        innerRadius: float = 0.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a canonical rectangular hollow section with optional exact corner arcs."""
        tol = Cell._native_tolerance(tolerance)
        try:
            width = float(width)
            length = float(length)
            height = float(height)
            thickness = abs(float(thickness))
            outer_radius = max(0.0, float(outerRadius))
            inner_radius = max(0.0, float(innerRadius))
        except Exception:
            width = length = height = thickness = outer_radius = inner_radius = float("nan")

        inner_width = width - 2.0 * thickness
        inner_length = length - 2.0 * thickness
        if (
            tol is None
            or not all(
                math.isfinite(v)
                for v in (width, length, height, thickness, outer_radius, inner_radius)
            )
            or width <= tol
            or length <= tol
            or height <= tol
            or thickness <= tol
            or inner_width <= tol
            or inner_length <= tol
            or outer_radius > 0.5 * min(width, length) + tol
            or inner_radius > 0.5 * min(inner_width, inner_length) + tol
        ):
            if not silent:
                print("Cell.ByRHS - Error: Invalid dimensions or fillet radii. Returning None.")
            return None

        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeEdge,
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_MakeWire,
            )
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
            from OCC.Core.GC import GC_MakeArcOfCircle
            from OCC.Core.gp import gp_Pnt, gp_Vec

            def rounded_wire(w, l, r, z):
                hx = 0.5 * w
                hy = 0.5 * l
                r = min(max(0.0, r), hx, hy)
                maker = BRepBuilderAPI_MakeWire()
                if r <= tol:
                    pts = [
                        gp_Pnt(hx, -hy, z),
                        gp_Pnt(-hx, -hy, z),
                        gp_Pnt(-hx, hy, z),
                        gp_Pnt(hx, hy, z),
                    ]
                    for i in range(4):
                        maker.Add(BRepBuilderAPI_MakeEdge(pts[i], pts[(i + 1) % 4]).Edge())
                else:
                    q = r / math.sqrt(2.0)
                    p0 = gp_Pnt(hx - r, -hy, z)
                    p1 = gp_Pnt(-hx + r, -hy, z)
                    p2 = gp_Pnt(-hx, -hy + r, z)
                    p3 = gp_Pnt(-hx, hy - r, z)
                    p4 = gp_Pnt(-hx + r, hy, z)
                    p5 = gp_Pnt(hx - r, hy, z)
                    p6 = gp_Pnt(hx, hy - r, z)
                    p7 = gp_Pnt(hx, -hy + r, z)
                    arcs = [
                        (p1, gp_Pnt(-hx + r - q, -hy + r - q, z), p2),
                        (p3, gp_Pnt(-hx + r - q, hy - r + q, z), p4),
                        (p5, gp_Pnt(hx - r + q, hy - r + q, z), p6),
                        (p7, gp_Pnt(hx - r + q, -hy + r - q, z), p0),
                    ]
                    sequence = [
                        BRepBuilderAPI_MakeEdge(p0, p1).Edge(),
                        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*arcs[0]).Value()).Edge(),
                        BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
                        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*arcs[1]).Value()).Edge(),
                        BRepBuilderAPI_MakeEdge(p4, p5).Edge(),
                        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*arcs[2]).Value()).Edge(),
                        BRepBuilderAPI_MakeEdge(p6, p7).Edge(),
                        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*arcs[3]).Value()).Edge(),
                    ]
                    for edge in sequence:
                        maker.Add(edge)
                return maker.Wire() if maker.IsDone() else None

            z0 = -0.5 * height
            outer_wire = rounded_wire(width, length, outer_radius, z0)
            inner_wire = rounded_wire(inner_width, inner_length, inner_radius, z0)
            if outer_wire is None or inner_wire is None:
                return None

            # Hole wires must have the opposite orientation to the outer boundary.
            inner_wire.Reverse()
            face_maker = BRepBuilderAPI_MakeFace(outer_wire, True)
            face_maker.Add(inner_wire)
            if not face_maker.IsDone():
                return None
            prism = BRepPrimAPI_MakePrism(face_maker.Face(), gp_Vec(0.0, 0.0, height), True, True)
            shape = prism.Shape()
        except Exception:
            if not silent:
                print("Cell.ByRHS - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(shape)

    @staticmethod
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

    @staticmethod
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
        except Exception:
            if not silent:
                print(f"Cell.{label} - Error: Native OCCT thickening failed. Returning None.")
            return None
        return Cell._native_result(result_shape)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def ByPipe(
        edge,
        profile=None,
        radius: float = 0.5,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Sweep a closed profile natively along an Edge and return a smooth Cell."""
        if not Topology.IsInstance(edge, "Edge"):
            if not silent:
                print("Cell.ByPipe - Error: Invalid Edge. Returning None.")
            return None
        tol = Cell._native_tolerance(tolerance)
        try:
            radius = abs(float(radius))
        except Exception:
            radius = float("nan")
        if tol is None or not math.isfinite(radius) or radius <= tol:
            return None
        edge_shape = getattr(edge, "shape", None)
        if _is_null_shape(edge_shape):
            return None

        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
            from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
            from OCC.Core.TopoDS import topods

            spine_maker = BRepBuilderAPI_MakeWire()
            spine_maker.Add(topods.Edge(edge_shape))
            if not spine_maker.IsDone():
                return None
            spine = spine_maker.Wire()

            if profile is None:
                circle = gp_Circ(
                    gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0)),
                    radius,
                )
                profile_edge = BRepBuilderAPI_MakeEdge(circle).Edge()
                profile_maker = BRepBuilderAPI_MakeWire()
                profile_maker.Add(profile_edge)
                if not profile_maker.IsDone():
                    return None
                profile_shape = profile_maker.Wire()
            else:
                profile_shape = getattr(profile, "shape", None)
                if _is_null_shape(profile_shape):
                    return None

            pipe = BRepOffsetAPI_MakePipeShell(spine)
            # Contact + correction lets OCCT place an XY-plane profile on the
            # spine and rotate it orthogonal to the local tangent.
            pipe.Add(profile_shape, True, True)
            if not pipe.IsReady():
                return None
            pipe.Build()
            if not pipe.IsDone():
                return None
            if not pipe.MakeSolid():
                return None
            result_shape = pipe.Shape()
        except Exception:
            if not silent:
                print("Cell.ByPipe - Error: Native OCCT sweep failed. Returning None.")
            return None
        return Cell._native_result(result_shape)

    @staticmethod
    def ByEgg(
        profile,
        height: float = 1.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth body of revolution through the supplied egg profile."""
        tol = Cell._native_tolerance(tolerance)
        try:
            height = abs(float(height))
        except Exception:
            height = float("nan")
        if tol is None or not math.isfinite(height) or height <= tol:
            return None
        if not isinstance(profile, (list, tuple)) or len(profile) < 3:
            if not silent:
                print("Cell.ByEgg - Error: Invalid profile. Returning None.")
            return None

        points = []
        try:
            for item in profile:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    return None
                radius = abs(float(item[0])) * height
                z = float(item[2]) * height
                if not math.isfinite(radius) or not math.isfinite(z):
                    return None
                points.append((radius, z))
        except Exception:
            return None

        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeEdge,
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_MakeWire,
            )
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
            from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
            from OCC.Core.TColgp import TColgp_Array1OfPnt
            from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt

            array = TColgp_Array1OfPnt(1, len(points))
            for index, (radius, z) in enumerate(points, start=1):
                array.SetValue(index, gp_Pnt(radius, 0.0, z))
            curve_builder = GeomAPI_PointsToBSpline(array)
            curve = curve_builder.Curve()
            profile_edge = BRepBuilderAPI_MakeEdge(curve).Edge()

            first = gp_Pnt(points[0][0], 0.0, points[0][1])
            last = gp_Pnt(points[-1][0], 0.0, points[-1][1])
            first_axis = gp_Pnt(0.0, 0.0, points[0][1])
            last_axis = gp_Pnt(0.0, 0.0, points[-1][1])

            maker = BRepBuilderAPI_MakeWire()
            maker.Add(profile_edge)
            if points[-1][0] > tol:
                maker.Add(BRepBuilderAPI_MakeEdge(last, last_axis).Edge())
            maker.Add(BRepBuilderAPI_MakeEdge(last_axis, first_axis).Edge())
            if points[0][0] > tol:
                maker.Add(BRepBuilderAPI_MakeEdge(first_axis, first).Edge())
            if not maker.IsDone():
                return None
            face_maker = BRepBuilderAPI_MakeFace(maker.Wire(), True)
            if not face_maker.IsDone():
                return None
            axis = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
            revol = BRepPrimAPI_MakeRevol(face_maker.Face(), axis, 2.0 * math.pi, True)
            result_shape = revol.Shape()
        except Exception:
            if not silent:
                print("Cell.ByEgg - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(result_shape)

    @staticmethod
    def ByHyperboloid(
        baseRadius: float = 0.5,
        topRadius: float = 0.5,
        height: float = 1.0,
        twist: float = 60.0,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a smooth ruled Cell between phase-shifted circular sections."""
        tol = Cell._native_tolerance(tolerance)
        try:
            base_radius = abs(float(baseRadius))
            top_radius = abs(float(topRadius))
            height = float(height)
            twist = float(twist)
        except Exception:
            base_radius = top_radius = height = twist = float("nan")
        if (
            tol is None
            or not all(math.isfinite(v) for v in (base_radius, top_radius, height, twist))
            or height <= tol
            or max(base_radius, top_radius) <= tol
        ):
            return None
        if base_radius <= tol or top_radius <= tol:
            return Cell.ByCone(
                baseRadius=base_radius,
                topRadius=top_radius,
                height=height,
                tolerance=tol,
                silent=silent,
            )

        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
            from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

            angle = math.radians(twist)
            bottom_axis = gp_Ax2(
                gp_Pnt(0.0, 0.0, -0.5 * height),
                gp_Dir(0.0, 0.0, 1.0),
                gp_Dir(math.cos(angle), math.sin(angle), 0.0),
            )
            top_axis = gp_Ax2(
                gp_Pnt(0.0, 0.0, 0.5 * height),
                gp_Dir(0.0, 0.0, 1.0),
                gp_Dir(1.0, 0.0, 0.0),
            )

            def circle_wire(axis, radius):
                edge = BRepBuilderAPI_MakeEdge(gp_Circ(axis, radius)).Edge()
                maker = BRepBuilderAPI_MakeWire()
                maker.Add(edge)
                return maker.Wire() if maker.IsDone() else None

            bottom_wire = circle_wire(bottom_axis, base_radius)
            top_wire = circle_wire(top_axis, top_radius)
            if bottom_wire is None or top_wire is None:
                return None

            loft = BRepOffsetAPI_ThruSections(True, True, tol)
            loft.CheckCompatibility(False)
            loft.AddWire(bottom_wire)
            loft.AddWire(top_wire)
            loft.Build()
            if not loft.IsDone():
                return None
            result_shape = loft.Shape()
        except Exception:
            if not silent:
                print("Cell.ByHyperboloid - Error: Native OCCT construction failed. Returning None.")
            return None
        return Cell._native_result(result_shape)


    @staticmethod
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

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """Return a Vertex strictly inside the Cell when one can be found."""
        return CellUtility.InternalVertex(cell, tolerance=tolerance)

    def Shells(self, hostTopology=None, shells=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_SHELL)
        else:
            result = list(getattr(self, "shells", []) or [])

        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_FACE)
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
            result = _downward_wrappers(self, TopAbs_EDGE)
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
            result = _downward_wrappers(self, TopAbs_VERTEX)
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


class CellUtility:
    @staticmethod
    def Volume(cell):
        """Return the OCCT volume of a Cell."""
        if not isinstance(cell, Cell):
            return None

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return None

        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps

            properties = GProp_GProps()
            brepgprop.VolumeProperties(shape, properties)
            volume = float(properties.Mass())
        except Exception:
            return None

        return volume if math.isfinite(volume) else None

    @staticmethod
    def Contains(cell, vertex, tolerance: float = 0.0001):
        """
        Classify a Vertex against a Cell.

        Returns 0 for inside, 1 for on the boundary, and 2 for outside or an
        invalid classification.
        """
        if not isinstance(cell, Cell) or not isinstance(vertex, Vertex):
            return 2

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return 2

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return 2

        try:
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt

            classifier = BRepClass3d_SolidClassifier(shape)
            classifier.Perform(
                gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)),
                tolerance,
            )
            state = classifier.State()
        except Exception:
            return 2

        if state == TopAbs_IN:
            return 0
        if state == TopAbs_ON:
            return 1
        return 2

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """Return a Vertex strictly inside a Cell when one can be found."""
        if not isinstance(cell, Cell):
            return None

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
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
            x = xmin + (xmax - xmin) * i / steps
            for j in range(1, steps):
                y = ymin + (ymax - ymin) * j / steps
                for k in range(1, steps):
                    z = zmin + (zmax - zmin) * k / steps
                    candidate = Vertex.ByCoordinates(x, y, z)
                    if CellUtility.Contains(cell, candidate, tolerance) == 0:
                        return candidate

        return center if isinstance(center, Vertex) else None

    @staticmethod
    def AdjacentVertices(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Vertices(hostTopology, output)

    @staticmethod
    def AdjacentEdges(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Edges(hostTopology, output)

    @staticmethod
    def AdjacentWires(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Wires(hostTopology, output)

    @staticmethod
    def AdjacentFaces(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Faces(hostTopology, output)

    @staticmethod
    def AdjacentShells(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Shells(hostTopology, output)

    @staticmethod
    def AdjacentCells(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Cells(hostTopology, output)

    @staticmethod
    def AdjacentCellComplexes(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.CellComplexes(hostTopology, output)
