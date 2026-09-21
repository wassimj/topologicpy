from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
from .topology import (
    Topology,
    _downward_wrappers,
    _shape_from_topology,
    _is_null_shape,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    )
from .wire import Wire
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_face
from .helpers import unique_by_uuid, edge_key


def _face_tolerance(value=0.0001) -> float:
    """Return a finite positive geometric tolerance."""
    try:
        value = abs(float(value))
    except Exception:
        value = 0.0001
    if not math.isfinite(value) or value <= 0.0:
        return 1.0e-12
    return value


def _same_shape(shape_a, shape_b) -> bool:
    """Return True when two OCCT shapes reference the same topological entity."""
    if shape_a is None or shape_b is None:
        return False
    try:
        return bool(shape_a.IsSame(shape_b))
    except Exception:
        return False


def _explore_shapes(shape, shape_type):
    """Return unique OCCT subshapes of the requested type in explorer order."""
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


def _wire_area(occ_wire):
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
    try:
        from OCC.Core.TopAbs import TopAbs_WIRE
        from OCC.Core.TopoDS import topods
        candidates = [topods.Wire(shape) for shape in _explore_shapes(occ_face, TopAbs_WIRE)]
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
    """Return the native internal-boundary wires of a backend Face."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return []
    outer = _outer_wire_shape(face)
    try:
        from OCC.Core.TopAbs import TopAbs_WIRE
        from OCC.Core.TopoDS import topods
        wires = [topods.Wire(shape) for shape in _explore_shapes(occ_face, TopAbs_WIRE)]
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
        if not all(math.isfinite(value) for value in (u0, u1, v0, v1)):
            return None
        return surface, u0, u1, v0, v1
    except Exception:
        return None



def _surface_adaptor(face):
    """Return a location-aware OCCT surface adaptor for a backend Face."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return None
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        return BRepAdaptor_Surface(occ_face, True)
    except Exception:
        return None


def _surface_d1(face, raw_u, raw_v):
    """Return location-aware (point, dU, dV) data at native UV parameters."""
    adaptor = _surface_adaptor(face)
    if adaptor is None:
        return None
    try:
        from OCC.Core.gp import gp_Pnt, gp_Vec
        point = gp_Pnt()
        derivative_u = gp_Vec()
        derivative_v = gp_Vec()
        adaptor.D1(
            float(raw_u),
            float(raw_v),
            point,
            derivative_u,
            derivative_v,
        )
        return point, derivative_u, derivative_v
    except Exception:
        return None


def _surface_and_location(face):
    """Return the unlocated supporting surface and its TopLoc_Location when exposed."""
    occ_face = _as_occ_face(face)
    if occ_face is None:
        return None, None
    try:
        from OCC.Core.BRep import BRep_Tool
        try:
            from OCC.Core.TopLoc import TopLoc_Location
            location = TopLoc_Location()
            surface = BRep_Tool.Surface(occ_face, location)
            if surface is not None:
                return surface, location
        except Exception:
            pass
        surface = BRep_Tool.Surface(occ_face)
        return surface, None
    except Exception:
        return None, None


def _project_point_to_surface(face, vertex, tolerance=0.0001):
    """Project a world-space Vertex to the Face support and return (u, v, distance)."""
    if not isinstance(vertex, Vertex):
        return None
    surface, location = _surface_and_location(face)
    if surface is None:
        return None
    try:
        from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCC.Core.gp import gp_Pnt

        point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))

        if location is not None:
            try:
                if not location.IsIdentity():
                    point.Transform(location.Transformation().Inverted())
            except Exception:
                pass

        projector = GeomAPI_ProjectPointOnSurf(point, surface)
        if projector.NbPoints() < 1:
            return None

        distance = float(projector.LowerDistance())
        if not math.isfinite(distance):
            return None

        raw_u, raw_v = projector.LowerDistanceParameters()
        return float(raw_u), float(raw_v), distance
    except Exception:
        return None


def _normalized_to_raw(face, u, v):
    """Map normalized TopologicPy UV parameters to native surface parameters."""
    data = _surface_and_bounds(face)
    if data is None:
        return None
    surface, u0, u1, v0, v1 = data
    try:
        u = float(u)
        v = float(v)
    except Exception:
        return None
    raw_u = u0 + u * (u1 - u0)
    raw_v = v0 + v * (v1 - v0)
    return surface, raw_u, raw_v, u0, u1, v0, v1


def _raw_to_normalized(face, raw_u, raw_v):
    """Map native surface parameters to normalized TopologicPy UV parameters."""
    data = _surface_and_bounds(face)
    if data is None:
        return None
    _, u0, u1, v0, v1 = data
    du = u1 - u0
    dv = v1 - v0
    if abs(du) <= 1.0e-30 or abs(dv) <= 1.0e-30:
        return None
    return [(float(raw_u) - u0) / du, (float(raw_v) - v0) / dv]


def _normalized_vector(values, tolerance=1.0e-12):
    """Return a normalized finite 3D vector, or None if degenerate."""
    try:
        vector = [float(values[0]), float(values[1]), float(values[2])]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in vector):
        return None
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= _face_tolerance(tolerance):
        return None
    return [value / magnitude for value in vector]


def _expanded_knot_data(values):
    """Convert an expanded knot vector to unique knots and multiplicities."""
    if not isinstance(values, (list, tuple)):
        return None
    try:
        expanded = [float(value) for value in values]
    except Exception:
        return None
    if len(expanded) < 2 or any(not math.isfinite(value) for value in expanded):
        return None
    if any(expanded[index] > expanded[index + 1] for index in range(len(expanded) - 1)):
        return None
    unique_knots = []
    multiplicities = []
    for value in expanded:
        if unique_knots and value == unique_knots[-1]:
            multiplicities[-1] += 1
        else:
            unique_knots.append(value)
            multiplicities.append(1)
    if len(unique_knots) < 2:
        return None
    return unique_knots, multiplicities


@dataclass(eq=False)
class Face(Topology):
    external: Optional[Wire] = None
    internals: list = field(default_factory=list)

    @staticmethod
    def AddInternalBoundaries(
        face,
        wires,
        tolerance: float = 0.0001
    ):
        """
        Add closed internal boundary Wires to an existing Face while preserving
        the Face's native OCCT support surface and existing trims.

        The input Face is copied natively using BRepBuilderAPI_MakeFace.Init().
        The new Wires are copied, supplied with p-curves on the existing support
        surface when necessary, and then added as holes.

        Parameters
        ----------
        face : Face
            The input backend Face.
        wires : list
            Closed backend Wires to add as holes.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Face
            The modified Face, or None if construction fails.
        """
        import math

        if not isinstance(face, Face):
            return None

        if not isinstance(wires, (list, tuple)):
            return None

        wires = [
            wire
            for wire in wires
            if isinstance(wire, Wire)
        ]

        if len(wires) == 0:
            return face

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            return None

        occ_face = _as_occ_face(face)

        if occ_face is None:
            return None

        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_Copy,
                BRepBuilderAPI_MakeFace,
            )
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.ShapeFix import ShapeFix_Edge
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Copy the EXISTING Face.
        #
        # This is the critical operation: the support surface, external trim,
        # existing internal trims, location and orientation are retained.
        # ------------------------------------------------------------------

        try:
            builder = BRepBuilderAPI_MakeFace()
            builder.Init(occ_face)

            if not builder.IsDone():
                return None

            working_face = builder.Face()

            if working_face is None or working_face.IsNull():
                return None

        except Exception:
            return None

        # Existing external boundary orientation.
        external_wire = _outer_wire_shape(face)

        try:
            external_orientation = (
                external_wire.Orientation()
                if external_wire is not None
                else None
            )
        except Exception:
            external_orientation = None

        # ------------------------------------------------------------------
        # Add each new hole.
        # ------------------------------------------------------------------

        for wire in wires:

            occ_wire = _as_occ_wire(wire)

            if occ_wire is None:
                return None

            # Never mutate the input Wire. Copy its topology and exact geometry.
            try:
                copier = BRepBuilderAPI_Copy(
                    occ_wire,
                    True,   # copy geometry
                    False   # triangulation not needed
                )

                copied_shape = copier.Shape()

                if copied_shape is None or copied_shape.IsNull():
                    return None

                hole_wire = topods.Wire(
                    copied_shape
                )

            except Exception:
                return None

            # Internal wires must oppose the external boundary orientation.
            if external_orientation is not None:
                try:
                    if hole_wire.Orientation() == external_orientation:
                        hole_wire = topods.Wire(
                            hole_wire.Reversed()
                        )
                except Exception:
                    return None

            # --------------------------------------------------------------
            # Ensure every Edge has a p-curve on the ORIGINAL support surface.
            #
            # ShapeFix_Edge projects the existing exact 3D curve onto the
            # surface to create the required 2D parametric representation.
            # The 3D curve itself is not chorded or reconstructed.
            # --------------------------------------------------------------

            try:
                edge_fixer = ShapeFix_Edge()

                explorer = TopExp_Explorer(
                    hole_wire,
                    TopAbs_EDGE
                )

                while explorer.More():

                    edge = topods.Edge(
                        explorer.Current()
                    )

                    try:
                        edge_fixer.FixAddPCurve(
                            edge,
                            working_face,
                            False,
                            tolerance
                        )
                    except Exception:
                        return None

                    try:
                        edge_fixer.FixReversed2d(
                            edge,
                            working_face
                        )
                    except Exception:
                        pass

                    try:
                        edge_fixer.FixSameParameter(
                            edge,
                            working_face,
                            tolerance
                        )
                    except Exception:
                        pass

                    try:
                        edge_fixer.FixVertexTolerance(
                            edge,
                            working_face
                        )
                    except Exception:
                        pass

                    explorer.Next()

            except Exception:
                return None

            # Add the complete closed Wire as another trim of the copied Face.
            try:
                builder.Add(
                    hole_wire
                )
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Retrieve and validate the completed Face.
        # ------------------------------------------------------------------

        try:
            if not builder.IsDone():
                return None

            result_shape = builder.Face()

            if (
                result_shape is None
                or result_shape.IsNull()
            ):
                return None

            analyzer = BRepCheck_Analyzer(
                result_shape
            )

            if not analyzer.IsValid():
                return None

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Wrap result and preserve metadata.
        # ------------------------------------------------------------------

        try:
            result = Face.ByOcctShape(
                result_shape
            )
        except Exception:
            result = None

        if not isinstance(result, Face):
            return None

        result = _wrap_metadata(
            face,
            result
        )

        # Preserve the original wrapper-level boundary references where
        # available, and append the new hole Wires.
        try:
            external = getattr(
                face,
                "external",
                None
            )

            if isinstance(external, Wire):
                result.external = external
        except Exception:
            pass

        try:
            existing = getattr(
                face,
                "internals",
                []
            )

            existing = [
                wire
                for wire in (existing or [])
                if isinstance(wire, Wire)
            ]

            result.internals = (
                existing
                + list(wires)
            )

        except Exception:
            pass

        return result

    @staticmethod
    def ByExternalBoundary(wire):
        """Create an exact Face from a closed Wire without chordalising curves."""
        if not isinstance(wire, Wire):
            return None
        try:
            if not wire.IsClosed():
                return None
        except Exception:
            return None

        try:
            shape = make_occ_face(wire)
        except Exception:
            shape = None

        if _is_null_shape(shape):
            return None

        result = Face.ByOcctShape(shape)
        if not isinstance(result, Face):
            return None

        # Preserve the original wrapper so metadata and exact constituent curves
        # remain directly available without re-traversal.
        result.external = wire
        result.internals = []
        return result

    @staticmethod
    def ByWire(wire):
        """Alias for exact curved-safe Face.ByExternalBoundary."""
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
    def ByExternalInternalBoundaries(
        externalBoundary,
        internalBoundaries,
        tolerance: float = 0.0001
    ):
        """
        Creates a Face from an external boundary Wire and optional internal
        boundary Wires.

        Internal boundary wires are added to the native OCCT Face with an
        orientation opposite to that of the external boundary. This is required
        by OCCT for the internal wires to represent holes rather than additional
        positive-area regions.

        Parameters
        ----------
        externalBoundary : Wire
            The external closed boundary Wire.
        internalBoundaries : list
            The internal closed boundary Wires.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Face
            The created Face, or None if construction fails.
        """
        from .wire import Wire

        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.TopoDS import topods
        except Exception:
            return None

        # ------------------------------------------------------------------
        # Validate external boundary
        # ------------------------------------------------------------------

        if not isinstance(
            externalBoundary,
            Wire
        ):
            return None

        if internalBoundaries is None:
            internalBoundaries = []

        if not isinstance(
            internalBoundaries,
            (list, tuple)
        ):
            return None

        if not all(
            isinstance(wire, Wire)
            for wire in internalBoundaries
        ):
            return None

        # ------------------------------------------------------------------
        # Retrieve native external Wire
        # ------------------------------------------------------------------

        external_shape = getattr(
            externalBoundary,
            "shape",
            None
        )

        if external_shape is None:
            try:
                external_shape = externalBoundary.GetOcctShape()
            except Exception:
                return None

        try:
            if external_shape.IsNull():
                return None
        except Exception:
            pass

        try:
            external_wire = topods.Wire(
                external_shape
            )
        except Exception:
            return None

        # ------------------------------------------------------------------
        # Create native Face from external boundary
        # ------------------------------------------------------------------

        try:
            builder = BRepBuilderAPI_MakeFace(
                external_wire,
                True
            )
        except Exception:
            return None

        if not builder.IsDone():
            return None

        # ------------------------------------------------------------------
        # Add internal boundaries.
        #
        # A TopoDS_Wire's FORWARD/REVERSED flag does not describe its signed
        # traversal in the supporting plane. Comparing those flags can invert
        # a Rhino face so that the hole becomes the positive region. Instead,
        # try both directions and retain the candidate whose area is closest to
        # ``outer area - accumulated hole area``.
        # ------------------------------------------------------------------

        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop

            def _face_area(occ_face):
                properties = GProp_GProps()
                brepgprop.SurfaceProperties(occ_face, properties)
                return abs(float(properties.Mass()))

            outer_area = _face_area(builder.Face())
        except Exception:
            outer_area = None

        selected_holes = []
        accumulated_hole_area = 0.0

        def _candidate_face(hole_wires):
            candidate_builder = BRepBuilderAPI_MakeFace(external_wire, True)
            if not candidate_builder.IsDone():
                return None
            for candidate_hole in hole_wires:
                candidate_builder.Add(candidate_hole)
            if not candidate_builder.IsDone():
                return None
            return candidate_builder.Face()

        for internalBoundary in internalBoundaries:
            internal_shape = getattr(internalBoundary, "shape", None)
            if internal_shape is None:
                try:
                    internal_shape = internalBoundary.GetOcctShape()
                except Exception:
                    return None
            try:
                internal_wire = topods.Wire(internal_shape)
                reversed_wire = topods.Wire(internal_wire.Reversed())
            except Exception:
                return None

            hole_area = _wire_area(internal_wire)
            if outer_area is not None and hole_area is not None:
                target_area = outer_area - accumulated_hole_area - hole_area
                choices = []
                for candidate_wire in (internal_wire, reversed_wire):
                    candidate_face = _candidate_face(selected_holes + [candidate_wire])
                    if candidate_face is None:
                        continue
                    try:
                        error = abs(_face_area(candidate_face) - target_area)
                    except Exception:
                        continue
                    choices.append((error, candidate_wire))
                if choices:
                    choices.sort(key=lambda item: item[0])
                    selected_holes.append(choices[0][1])
                    accumulated_hole_area += hole_area
                    continue

            # Defensive fallback when mass properties are unavailable.
            selected_holes.append(reversed_wire)

        builder = BRepBuilderAPI_MakeFace(external_wire, True)
        if not builder.IsDone():
            return None
        for internal_wire in selected_holes:
            try:
                builder.Add(internal_wire)
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Retrieve completed native Face
        # ------------------------------------------------------------------

        try:
            if not builder.IsDone():
                return None

            face_shape = builder.Face()

            if face_shape is None:
                return None

            try:
                if face_shape.IsNull():
                    return None
            except Exception:
                pass

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Wrap native Face
        # ------------------------------------------------------------------

        try:
            result = Face.ByOcctShape(
                face_shape
            )
        except Exception:
            result = None

        if not isinstance(
            result,
            Face
        ):
            try:
                result = Face(
                    shape=face_shape,
                    external=externalBoundary,
                    internals=list(internalBoundaries),
                    dictionary={},
                    contents=[],
                    contexts=[],
                    apertures=[]
                )
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Preserve the input boundary wrappers.
        #
        # Their standalone orientation does not need to be reversed. Only the
        # copies embedded in the native TopoDS_Face need hole orientation.
        # This also preserves any metadata attached to the original Wires.
        # ------------------------------------------------------------------

        try:
            result.external = externalBoundary
        except Exception:
            pass

        try:
            result.internals = list(
                internalBoundaries
            )
        except Exception:
            pass

        return result

    @staticmethod
    def ByNurbsParameters(
        controlPoints,
        weights,
        uKnots,
        vKnots,
        isRational,
        isUPeriodic,
        isVPeriodic,
        uDegree,
        vDegree,
        tolerance: float = 0.0001,
    ):
        """
        Creates a Face from exact OCCT B-spline/NURBS surface parameters.

        The control-point and weight grids use the convention:

            controlPoints[u][v]
            weights[u][v]

        The knot vectors are supplied in expanded form. Repeated knot values
        therefore appear repeatedly in the input lists and are converted here
        to OCCT's unique-knot plus multiplicity representation.

        Parameters
        ----------
        controlPoints : list
            Rectangular two-dimensional grid of backend Vertex objects.
        weights : list
            Rectangular two-dimensional grid of positive weights.
        uKnots : list
            Expanded knot vector in the U direction.
        vKnots : list
            Expanded knot vector in the V direction.
        isRational : bool
            If True, construct a rational NURBS surface.
        isUPeriodic : bool
            If True, the surface is periodic in U.
        isVPeriodic : bool
            If True, the surface is periodic in V.
        uDegree : int
            Degree in the U direction.
        vDegree : int
            Degree in the V direction.
        tolerance : float , optional
            Geometric tolerance used when creating the OCCT Face.
            Default is 0.0001.

        Returns
        -------
        Face
            The created backend Face, or None on failure.

        """
        try:
            tolerance = _face_tolerance(tolerance)

            uDegree = int(uDegree)
            vDegree = int(vDegree)

            isRational = bool(isRational)
            isUPeriodic = bool(isUPeriodic)
            isVPeriodic = bool(isVPeriodic)

        except Exception:
            return None

        if not isinstance(controlPoints, (list, tuple)):
            return None

        controlPoints = [
            list(row)
            for row in controlPoints
            if isinstance(row, (list, tuple))
        ]

        if len(controlPoints) < 2:
            return None

        nU = len(controlPoints)
        nV = len(controlPoints[0])

        if nV < 2:
            return None

        if any(
            len(row) != nV
            for row in controlPoints
        ):
            return None

        if any(
            not isinstance(vertex, Vertex)
            for row in controlPoints
            for vertex in row
        ):
            return None

        if (
            uDegree < 1
            or uDegree >= nU
            or vDegree < 1
            or vDegree >= nV
        ):
            return None

        if not isinstance(weights, (list, tuple)):
            return None

        if len(weights) != nU:
            return None

        try:
            weight_values = [
                [
                    float(value)
                    for value in row
                ]
                for row in weights
            ]
        except Exception:
            return None

        if any(
            len(row) != nV
            for row in weight_values
        ):
            return None

        if any(
            not math.isfinite(value)
            or value <= 0.0
            for row in weight_values
            for value in row
        ):
            return None

        if not isRational:
            weight_values = [
                [1.0] * nV
                for _ in range(nU)
            ]

        # Expanded knot vectors -> OCCT unique knots + multiplicities.
        u_data = _expanded_knot_data(uKnots)
        v_data = _expanded_knot_data(vKnots)

        if u_data is None or v_data is None:
            return None

        unique_u_knots, u_multiplicities = u_data
        unique_v_knots, v_multiplicities = v_data

        # ------------------------------------------------------------------
        # Validate OCCT pole/knot relationships.
        # ------------------------------------------------------------------

        if isUPeriodic:
            valid_u = (
                u_multiplicities[0] == u_multiplicities[-1]
                and all(
                    1 <= multiplicity <= uDegree
                    for multiplicity in u_multiplicities
                )
                and (
                    sum(u_multiplicities)
                    - u_multiplicities[0]
                    == nU
                )
            )
        else:
            valid_u = (
                sum(u_multiplicities)
                == nU + uDegree + 1
                and all(
                    1 <= multiplicity <= uDegree
                    for multiplicity in u_multiplicities[1:-1]
                )
                and 1 <= u_multiplicities[0] <= uDegree + 1
                and 1 <= u_multiplicities[-1] <= uDegree + 1
            )

        if isVPeriodic:
            valid_v = (
                v_multiplicities[0] == v_multiplicities[-1]
                and all(
                    1 <= multiplicity <= vDegree
                    for multiplicity in v_multiplicities
                )
                and (
                    sum(v_multiplicities)
                    - v_multiplicities[0]
                    == nV
                )
            )
        else:
            valid_v = (
                sum(v_multiplicities)
                == nV + vDegree + 1
                and all(
                    1 <= multiplicity <= vDegree
                    for multiplicity in v_multiplicities[1:-1]
                )
                and 1 <= v_multiplicities[0] <= vDegree + 1
                and 1 <= v_multiplicities[-1] <= vDegree + 1
            )

        if not valid_u or not valid_v:
            return None

        try:
            from OCC.Core.gp import gp_Pnt

            from OCC.Core.TColgp import (
                TColgp_Array2OfPnt,
            )

            from OCC.Core.TColStd import (
                TColStd_Array1OfInteger,
                TColStd_Array1OfReal,
                TColStd_Array2OfReal,
            )

            from OCC.Core.Geom import (
                Geom_BSplineSurface,
            )

            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeFace,
            )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Poles.
        # ------------------------------------------------------------------

        try:
            poles = TColgp_Array2OfPnt(
                1,
                nU,
                1,
                nV,
            )

            for u_index in range(nU):
                for v_index in range(nV):
                    vertex = controlPoints[u_index][v_index]

                    poles.SetValue(
                        u_index + 1,
                        v_index + 1,
                        gp_Pnt(
                            float(vertex.x),
                            float(vertex.y),
                            float(vertex.z),
                        ),
                    )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Unique knot arrays and multiplicities.
        # ------------------------------------------------------------------

        try:
            occ_u_knots = TColStd_Array1OfReal(
                1,
                len(unique_u_knots),
            )

            occ_u_mults = TColStd_Array1OfInteger(
                1,
                len(u_multiplicities),
            )

            for index, value in enumerate(
                unique_u_knots,
                start=1,
            ):
                occ_u_knots.SetValue(
                    index,
                    value,
                )

            for index, value in enumerate(
                u_multiplicities,
                start=1,
            ):
                occ_u_mults.SetValue(
                    index,
                    int(value),
                )

            occ_v_knots = TColStd_Array1OfReal(
                1,
                len(unique_v_knots),
            )

            occ_v_mults = TColStd_Array1OfInteger(
                1,
                len(v_multiplicities),
            )

            for index, value in enumerate(
                unique_v_knots,
                start=1,
            ):
                occ_v_knots.SetValue(
                    index,
                    value,
                )

            for index, value in enumerate(
                v_multiplicities,
                start=1,
            ):
                occ_v_mults.SetValue(
                    index,
                    int(value),
                )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Construct the exact native surface.
        # ------------------------------------------------------------------

        try:
            if isRational:
                occ_weights = TColStd_Array2OfReal(
                    1,
                    nU,
                    1,
                    nV,
                )

                for u_index in range(nU):
                    for v_index in range(nV):
                        occ_weights.SetValue(
                            u_index + 1,
                            v_index + 1,
                            weight_values[u_index][v_index],
                        )

                surface = Geom_BSplineSurface(
                    poles,
                    occ_weights,
                    occ_u_knots,
                    occ_v_knots,
                    occ_u_mults,
                    occ_v_mults,
                    uDegree,
                    vDegree,
                    isUPeriodic,
                    isVPeriodic,
                )

            else:
                surface = Geom_BSplineSurface(
                    poles,
                    occ_u_knots,
                    occ_v_knots,
                    occ_u_mults,
                    occ_v_mults,
                    uDegree,
                    vDegree,
                    isUPeriodic,
                    isVPeriodic,
                )

        except Exception:
            return None

        # Build a Face using the natural finite UV bounds of the B-spline
        # surface. No tessellation is introduced.
        try:
            maker = BRepBuilderAPI_MakeFace(
                surface,
                tolerance,
            )

            if not maker.IsDone():
                return None

            occ_face = maker.Face()

            if occ_face is None or occ_face.IsNull():
                return None

            return Face.ByOcctShape(
                occ_face
            )

        except Exception:
            return None

    @staticmethod
    def ByNurbsParametersAndWires(
        controlPoints,
        weights,
        uKnots,
        vKnots,
        isRational,
        isUPeriodic,
        isVPeriodic,
        uDegree,
        vDegree,
        externalBoundary,
        internalBoundaries=None,
        reverse: bool = False,
        tolerance: float = 0.0001,
    ):
        """Create an exact trimmed NURBS face from a support surface and wires.

        The supplied three-dimensional wires are projected by OCCT onto the
        exact NURBS support. No tessellation or control-point fitting occurs.
        """
        if not isinstance(externalBoundary, Wire):
            return None
        internalBoundaries = list(internalBoundaries or [])
        if not all(isinstance(wire, Wire) for wire in internalBoundaries):
            return None

        support_face = Face.ByNurbsParameters(
            controlPoints, weights, uKnots, vKnots, isRational,
            isUPeriodic, isVPeriodic, uDegree, vDegree, tolerance,
        )
        if not isinstance(support_face, Face):
            return None

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepLib import breplib
            from OCC.Core.ShapeFix import ShapeFix_Face
            from OCC.Core.TopoDS import topods

            support_shape = topods.Face(support_face.shape)
            try:
                surface = BRep_Tool.Surface(support_shape)
            except Exception:
                surface = BRep_Tool.Surface_s(support_shape)
            outer_wire = topods.Wire(externalBoundary.shape)
            maker = BRepBuilderAPI_MakeFace(surface, outer_wire, True)
            if not maker.IsDone():
                return None

            outer_orientation = outer_wire.Orientation()
            for boundary in internalBoundaries:
                hole = topods.Wire(boundary.shape)
                if hole.Orientation() == outer_orientation:
                    hole = topods.Wire(hole.Reversed())
                maker.Add(hole)
            if not maker.IsDone():
                return None
            face_shape = maker.Face()
            try:
                breplib.BuildCurves3d(face_shape)
            except Exception:
                pass
            try:
                fixer = ShapeFix_Face(face_shape)
                fixer.SetPrecision(_face_tolerance(tolerance))
                fixer.Perform()
                face_shape = fixer.Face()
            except Exception:
                pass
            if bool(reverse):
                face_shape = topods.Face(face_shape.Reversed())
            result = Face.ByOcctShape(face_shape)
        except Exception:
            return None

        if not isinstance(result, Face):
            return None
        result.external = externalBoundary
        result.internals = internalBoundaries
        return result

    @staticmethod
    def ByOcctShape(
        shape,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None
    ):
        """
        Wraps an existing OCCT face without eagerly constructing its boundary
        wires, edges, or vertices.

        Subtopologies are discovered from the underlying OCCT shape only when
        requested.

        Parameters
        ----------
        shape : OCC.Core.TopoDS.TopoDS_Shape
            The input OCCT face shape.
        dictionary : object , optional
            The dictionary assigned to the face.
        contents : list , optional
            The contents assigned to the face.
        contexts : list , optional
            The contexts assigned to the face.
        apertures : list , optional
            The apertures assigned to the face.

        Returns
        -------
        Face
            The wrapped face, or None if the input cannot be converted to an
            OCCT face.
        """
        try:
            from OCC.Core.TopoDS import topods

            occ_face = topods.Face(shape)

            if occ_face.IsNull():
                return None

        except Exception:
            return None

        return Face(
            shape=occ_face,
            external=None,
            internals=[],
            dictionary=dictionary,
            contents=list(contents) if contents else [],
            contexts=list(contexts) if contexts else [],
            apertures=list(apertures) if apertures else [],
        )

    def ExternalBoundary(self):
        """Return the external boundary Wire of this Face."""
        if _is_null_shape(getattr(self, "shape", None)):
            return self.external if isinstance(self.external, Wire) else None
        outer = _outer_wire_shape(self)
        if outer is None:
            return None
        return Wire.ByOcctShape(outer)


    def Wire(self):
        """Alias for ExternalBoundary."""
        return self.ExternalBoundary()

    def InternalBoundaries(self, wires=None):
        """Return or populate the internal boundary Wires of this Face."""
        if _is_null_shape(getattr(self, "shape", None)):
            result = [wire for wire in (getattr(self, "internals", []) or []) if isinstance(wire, Wire)]
        else:
            result = [Wire.ByOcctShape(shape) for shape in _internal_wire_shapes(self)]
            result = [wire for wire in result if isinstance(wire, Wire)]
        if wires is not None:
            wires.extend(result)
            return 0
        return result


    def Edges(self, hostTopology=None, edges=None):
        """
        Returns the unique edges of the face.

        Parameters
        ----------
        hostTopology : object , optional
            Included for backend API compatibility.
        edges : list , optional
            If supplied, the resulting edges are appended to this list and the
            method returns 0.

        Returns
        -------
        list
            The face edges.
        """
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_EDGE
            )

        else:
            result = []

            if isinstance(self.external, Wire):
                result.extend(
                    self.external.Edges()
                )

            for wire in getattr(self, "internals", []) or []:
                if isinstance(wire, Wire):
                    result.extend(
                        wire.Edges()
                    )

            result = unique_by_uuid(result)

        if edges is not None:
            edges.extend(result)
            return 0

        return result

    def Vertices(self, hostTopology=None, vertices=None):
        """
        Returns the unique vertices of the face.

        Parameters
        ----------
        hostTopology : object , optional
            Included for backend API compatibility.
        vertices : list , optional
            If supplied, the resulting vertices are appended to this list and the
            method returns 0.

        Returns
        -------
        list
            The face vertices.
        """
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

            result = unique_by_uuid(
                [
                    vertex
                    for vertex in result
                    if isinstance(vertex, Vertex)
                ]
            )

        if vertices is not None:
            vertices.extend(result)
            return 0

        return result

    def Wires(self, hostTopology=None, wires=None):
        """Return the external boundary first, followed by internal boundaries."""
        result = []
        external = self.ExternalBoundary()
        if isinstance(external, Wire):
            result.append(external)
        result.extend(self.InternalBoundaries() or [])
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
        """Return the exact OCCT surface area of a Face, including trimming."""
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None
        try:
            from OCC.Core.GProp import GProp_GProps

            props = GProp_GProps()

            # pythonocc-core has exposed SurfaceProperties in two forms across
            # releases. Support both without falling back to polygonal area.
            try:
                from OCC.Core.BRepGProp import brepgprop
                brepgprop.SurfaceProperties(occ_face, props)
            except (ImportError, AttributeError):
                from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
                brepgprop_SurfaceProperties(occ_face, props)

            value = float(props.Mass())
            return value if math.isfinite(value) else None
        except Exception:
            return None

    @staticmethod
    def IsValid(topology):
        """Return the native validity state of a backend topology."""
        shape = _shape_from_topology(topology)
        if _is_null_shape(shape):
            return False
        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            return bool(BRepCheck_Analyzer(shape).IsValid())
        except Exception:
            return False

    @staticmethod
    def Fillet(face, vertexRadii, tolerance: float = 0.0001):
        """Delegate native planar Face filleting to the backend Face API."""
        return Face.Fillet(
            face,
            vertexRadii,
            tolerance=tolerance,
        )

    @staticmethod
    def RemoveCollinearEdges(
        face,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001,
    ):
        """Delegate native support-preserving boundary cleanup."""
        return Face.RemoveCollinearEdges(
            face,
            angTolerance=angTolerance,
            tolerance=tolerance,
        )

    @staticmethod
    def ByShell(shell, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """Unify a same-domain Shell into one native Face."""
        from .shell import Shell

        if not isinstance(shell, Shell):
            return None

        shell_shape = _shape_from_topology(shell)
        if _is_null_shape(shell_shape):
            return None

        try:
            angular_tolerance = math.radians(abs(float(angTolerance)))
            linear_tolerance = _face_tolerance(tolerance)
        except Exception:
            return None

        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods

            unifier = ShapeUpgrade_UnifySameDomain(
                shell_shape,
                True,
                True,
                True,
            )
            try:
                unifier.SetLinearTolerance(linear_tolerance)
            except Exception:
                pass
            try:
                unifier.SetAngularTolerance(angular_tolerance)
            except Exception:
                pass
            try:
                unifier.AllowInternalEdges(False)
            except Exception:
                pass

            unifier.Build()
            unified_shape = unifier.Shape()
            if _is_null_shape(unified_shape):
                return None

            faces = []
            explorer = TopExp_Explorer(unified_shape, TopAbs_FACE)
            while explorer.More():
                candidate = topods.Face(explorer.Current())
                if not _is_null_shape(candidate):
                    faces.append(candidate)
                explorer.Next()

            if len(faces) != 1 or not BRepCheck_Analyzer(faces[0]).IsValid():
                return None

            return _wrap_metadata(shell, Face.ByOcctShape(faces[0]))
        except Exception:
            return None

    @staticmethod
    def ByBoundariesOnSurface(
        face,
        externalBoundary,
        internalBoundaries=None,
        tolerance: float = 0.0001
    ):
        """
        Rebuilds a Face using new boundary Wires while retaining the exact
        supporting surface of the input Face.

        This is a primitive backend operation. It performs no simplification
        itself.
        """
        import math

        if not isinstance(face, Face):
            return None

        if not isinstance(externalBoundary, Wire):
            return None

        internalBoundaries = [
            wire
            for wire in (internalBoundaries or [])
            if isinstance(wire, Wire)
        ]

        tol = _face_tolerance(
            tolerance
        )

        # This primitive is currently used by planar boundary simplification.
        if FaceUtility.IsPlanar(
            face,
            tolerance=tol
        ) is not True:
            return None

        occ_face = _as_occ_face(
            face
        )

        if occ_face is None:
            return None

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_Copy,
                BRepBuilderAPI_MakeFace,
            )
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.ShapeFix import ShapeFix_Edge
            from OCC.Core.TopAbs import TopAbs_EDGE
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Retrieve the actual support surface.
        #
        # The no-location BRep_Tool.Surface(face) overload gives the transformed
        # support surface, so the new world-space boundary Wires can be projected
        # directly onto it.
        # ------------------------------------------------------------------

        try:
            surface = BRep_Tool.Surface(
                occ_face
            )
        except Exception:
            surface = None

        if surface is None:
            return None

        identity_location = TopLoc_Location()

        # ------------------------------------------------------------------
        # Copy one Wire and ensure all its Edges carry p-curves on the retained
        # support surface.
        # ------------------------------------------------------------------

        def prepare_wire(wire):

            occ_wire = _as_occ_wire(
                wire
            )

            if occ_wire is None:
                return None

            try:
                copier = BRepBuilderAPI_Copy(
                    occ_wire,
                    True,
                    False
                )

                copied_shape = copier.Shape()

                if (
                    copied_shape is None
                    or copied_shape.IsNull()
                ):
                    return None

                result_wire = topods.Wire(
                    copied_shape
                )

            except Exception:
                return None

            fixer = ShapeFix_Edge()

            explorer = TopExp_Explorer(
                result_wire,
                TopAbs_EDGE
            )

            while explorer.More():

                try:
                    edge = topods.Edge(
                        explorer.Current()
                    )
                except Exception:
                    return None

                try:
                    fixer.FixAddPCurve(
                        edge,
                        surface,
                        identity_location,
                        False,
                        tol
                    )
                except Exception:
                    return None

                try:
                    fixer.FixSameParameter(
                        edge,
                        tol
                    )
                except Exception:
                    return None

                explorer.Next()

            return result_wire

        outer_wire = prepare_wire(
            externalBoundary
        )

        if outer_wire is None:
            return None

        # ------------------------------------------------------------------
        # Construct a new trim domain on the SAME support surface.
        # ------------------------------------------------------------------

        try:
            maker = BRepBuilderAPI_MakeFace(
                surface,
                outer_wire,
                True
            )

            if not maker.IsDone():
                return None

        except Exception:
            return None

        try:
            outer_orientation = outer_wire.Orientation()
        except Exception:
            outer_orientation = None

        # ------------------------------------------------------------------
        # Add holes.
        # ------------------------------------------------------------------

        for internal in internalBoundaries:

            hole = prepare_wire(
                internal
            )

            if hole is None:
                return None

            if outer_orientation is not None:
                try:
                    if hole.Orientation() == outer_orientation:
                        hole = topods.Wire(
                            hole.Reversed()
                        )
                except Exception:
                    return None

            try:
                maker.Add(
                    hole
                )
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Validate.
        # ------------------------------------------------------------------

        try:
            if not maker.IsDone():
                return None

            result_face = maker.Face()

            if (
                result_face is None
                or result_face.IsNull()
            ):
                return None

            # Preserve original Face orientation.
            if result_face.Orientation() != occ_face.Orientation():
                result_face = topods.Face(
                    result_face.Reversed()
                )

            analyzer = BRepCheck_Analyzer(
                result_face
            )

            if not analyzer.IsValid():
                return None

        except Exception:
            return None

        result = Face.ByOcctShape(
            result_face
        )

        if not isinstance(result, Face):
            return None

        result = _wrap_metadata(
            face,
            result
        )

        # Keep the exact wrapper boundaries available.
        try:
            result.external = externalBoundary
            result.internals = list(
                internalBoundaries
            )
        except Exception:
            pass

        return result

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
    def CurvatureAtParameters(
        face,
        u=0.5,
        v=0.5,
        tolerance: float = 0.0001
    ):
        """Return native OCCT curvature properties at normalized parameters."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None

        surface, raw_u, raw_v, _, _, _, _ = mapped
        tol = _face_tolerance(tolerance)
        occ_face = _as_occ_face(face)
        if occ_face is None:
            return None

        properties = None

        # BRepLProp evaluates the actual located TopoDS_Face and therefore keeps
        # principal directions in world coordinates.
        try:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.BRepLProp import BRepLProp_SLProps
            adaptor = BRepAdaptor_Surface(occ_face, True)
            properties = BRepLProp_SLProps(adaptor, float(raw_u), float(raw_v), 2, tol)
        except Exception:
            properties = None

        # Compatibility fallback for PythonOCC builds without BRepLProp_SLProps.
        if properties is None:
            try:
                from OCC.Core.GeomLProp import GeomLProp_SLProps
                properties = GeomLProp_SLProps(surface, raw_u, raw_v, 2, tol)
            except Exception:
                return None

        try:
            if not properties.IsCurvatureDefined():
                return None

            maximum = float(properties.MaxCurvature())
            minimum = float(properties.MinCurvature())
            mean = float(properties.MeanCurvature())
            gaussian = float(properties.GaussianCurvature())
            is_umbilic = bool(properties.IsUmbilic())

            maximum_direction = None
            minimum_direction = None

            try:
                from OCC.Core.gp import gp_Dir
                max_dir = gp_Dir(1.0, 0.0, 0.0)
                min_dir = gp_Dir(0.0, 1.0, 0.0)
                properties.CurvatureDirections(max_dir, min_dir)
                maximum_direction = [float(max_dir.X()), float(max_dir.Y()), float(max_dir.Z())]
                minimum_direction = [float(min_dir.X()), float(min_dir.Y()), float(min_dir.Z())]
            except Exception:
                maximum_direction = None
                minimum_direction = None

            from OCC.Core.TopAbs import TopAbs_REVERSED
            if occ_face.Orientation() == TopAbs_REVERSED:
                old_maximum = maximum
                old_minimum = minimum
                maximum = -old_minimum
                minimum = -old_maximum
                mean = -mean
                maximum_direction, minimum_direction = minimum_direction, maximum_direction

            if not all(math.isfinite(value) for value in (maximum, minimum, mean, gaussian)):
                return None

            return {
                "maximum": maximum,
                "minimum": minimum,
                "mean": mean,
                "gaussian": gaussian,
                "maximumDirection": maximum_direction,
                "minimumDirection": minimum_direction,
                "isUmbilic": is_umbilic,
            }
        except Exception:
            return None

    @staticmethod
    def Edges(face):
        if isinstance(face, Face):
            return face.Edges()
        return []

    @staticmethod
    def Fillet(
        face,
        vertexRadii,
        tolerance: float = 0.0001
    ):
        """
        Fillet selected vertices of a planar Face using OCCT's native
        BRepFilletAPI_MakeFillet2d operation.

        The Face is modified natively. Its support surface, external boundary,
        internal boundaries, and unaffected curve geometry are preserved.

        Parameters
        ----------
        face : Face
            The backend Face.
        vertexRadii : list
            List of (Vertex, radius) pairs.
        tolerance : float , optional
            The desired tolerance.

        Returns
        -------
        Face
            The resulting Face, or None on failure.
        """
        import math

        occ_face = _as_occ_face(face)

        if occ_face is None:
            return None

        if not isinstance(
            vertexRadii,
            (list, tuple)
        ):
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if (
            not math.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            return None

        # ------------------------------------------------------------------
        # Import OCCT fillet machinery.
        # ------------------------------------------------------------------

        try:
            from OCC.Core.BRepFilletAPI import (
                BRepFilletAPI_MakeFillet2d
            )
            from OCC.Core.BRepCheck import (
                BRepCheck_Analyzer
            )
            from OCC.Core.ChFi2d import (
                ChFi2d_IsDone
            )
            from OCC.Core.TopoDS import topods

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Initialise the native planar Face fillet builder.
        # ------------------------------------------------------------------

        try:
            builder = BRepFilletAPI_MakeFillet2d(
                occ_face
            )
        except Exception:
            return None

        fillet_count = 0

        # ------------------------------------------------------------------
        # Apply every requested fillet to the ORIGINAL Face topology.
        #
        # The vertex passed to AddFillet is the actual TopoDS_Vertex belonging
        # to the Face. There is no coordinate matching or reconstruction.
        # ------------------------------------------------------------------

        for item in vertexRadii:

            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
            ):
                return None

            vertex, radius = item

            if not isinstance(vertex, Vertex):
                return None

            try:
                radius = float(radius)
            except Exception:
                return None

            if (
                not math.isfinite(radius)
                or radius <= tolerance
            ):
                continue

            vertex_shape = _shape_from_topology(
                vertex
            )

            if _is_null_shape(vertex_shape):
                return None

            try:
                occ_vertex = topods.Vertex(
                    vertex_shape
                )
            except Exception:
                return None

            try:
                fillet_edge = builder.AddFillet(
                    occ_vertex,
                    radius
                )
            except Exception:
                return None

            # OCCT explicitly requires checking the construction status after
            # every AddFillet call.
            try:
                if builder.Status() != ChFi2d_IsDone:
                    return None
            except Exception:
                return None

            try:
                if fillet_edge.IsNull():
                    return None
            except Exception:
                return None

            fillet_count += 1

        if fillet_count == 0:
            return face

        # ------------------------------------------------------------------
        # Finalise and validate.
        # ------------------------------------------------------------------

        try:
            builder.Build()

            if not builder.IsDone():
                return None

            shape = builder.Shape()

            if _is_null_shape(shape):
                return None

            analyzer = BRepCheck_Analyzer(
                shape
            )

            if not analyzer.IsValid():
                return None

            occ_result = topods.Face(
                shape
            )

        except Exception:
            return None

        result = Face.ByOcctShape(
            occ_result
        )

        if not isinstance(result, Face):
            return None

        return _wrap_metadata(
            face,
            result
        )

    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5, tolerance: float = 0.0001):
        """Return the oriented world-space unit normal at normalized UV parameters."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None
        _, raw_u, raw_v, _, _, _, _ = mapped
        tol = _face_tolerance(tolerance)

        data = _surface_d1(face, raw_u, raw_v)
        if data is None:
            return None

        _, derivative_u, derivative_v = data

        try:
            normal = derivative_u.Crossed(derivative_v)
            result = [float(normal.X()), float(normal.Y()), float(normal.Z())]

            from OCC.Core.TopAbs import TopAbs_REVERSED
            occ_face = _as_occ_face(face)
            if occ_face is not None and occ_face.Orientation() == TopAbs_REVERSED:
                result = [-value for value in result]

            return _normalized_vector(result, tolerance=tol)
        except Exception:
            return None

    @staticmethod
    def ParametersAtVertex(face, vertex, tolerance: float = 0.0001):
        """Return normalized UV parameters of a world-space Vertex on the Face."""
        if not isinstance(vertex, Vertex):
            return None

        projection = _project_point_to_surface(
            face,
            vertex,
            tolerance=tolerance,
        )
        if projection is None:
            return None

        raw_u, raw_v, distance = projection
        if distance > _face_tolerance(tolerance):
            return None

        return _raw_to_normalized(face, raw_u, raw_v)

    @staticmethod
    def IsInside(face, vertex, tolerance: float = 0.0001):
        """Return True when a Vertex lies on and inside/on the trimmed Face."""
        if not isinstance(vertex, Vertex):
            return False

        occ_face = _as_occ_face(face)
        if occ_face is None:
            return False

        tol = _face_tolerance(tolerance)

        projection = _project_point_to_surface(face, vertex, tolerance=tol)
        if projection is None:
            return False

        raw_u, raw_v, distance = projection
        if distance > tol:
            return False

        try:
            from OCC.Core.BRepClass import BRepClass_FaceClassifier
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt

            point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
            classifier = BRepClass_FaceClassifier()
            performed = False

            for args in (
                (occ_face, point, tol, True, 0.1),
                (occ_face, point, tol, True),
                (occ_face, point, tol),
            ):
                try:
                    classifier.Perform(*args)
                    performed = True
                    break
                except TypeError:
                    continue

            if performed:
                state = classifier.State()
                if state in (TopAbs_IN, TopAbs_ON):
                    return True
        except Exception:
            pass

        try:
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt2d

            classifier = BRepTopAdaptor_FClass2d(occ_face, tol)
            state = classifier.Perform(gp_Pnt2d(float(raw_u), float(raw_v)))
            return state in (TopAbs_IN, TopAbs_ON)
        except Exception:
            return False

    @staticmethod
    def RemoveCollinearEdges(
        face,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001
    ):
        """
        Removes redundant consecutive linear collinear Edges from a Face while
        preserving its native OCCT support surface and curved boundary geometry.

        Curved Edges are explicitly protected. Only vertices joining exclusively
        linear Edges are eligible for removal.
        """
        import math

        if not isinstance(face, Face):
            return None

        occ_face = _as_occ_face(face)

        if occ_face is None:
            return None

        try:
            angTolerance = abs(float(angTolerance))
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if (
            not math.isfinite(angTolerance)
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            return None

        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer
            from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods

            from .edge import EdgeUtility

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Collect the original native Edges.
        # ------------------------------------------------------------------

        original_edges = []

        explorer = TopExp_Explorer(
            occ_face,
            TopAbs_EDGE
        )

        while explorer.More():

            try:
                edge = topods.Edge(
                    explorer.Current()
                )

                if not any(
                    edge.IsSame(existing)
                    for existing in original_edges
                ):
                    original_edges.append(edge)

            except Exception:
                pass

            explorer.Next()

        if len(original_edges) < 1:
            return face

        # ------------------------------------------------------------------
        # Configure OCCT's same-domain unifier.
        #
        # UnifyEdges=True
        # UnifyFaces=False
        # ConcatBSplines=True
        #
        # Faces are NEVER merged or reconstructed. Only compatible boundary
        # Edge chains may be combined.
        # ------------------------------------------------------------------

        try:
            unifier = ShapeUpgrade_UnifySameDomain(
                occ_face,
                True,   # UnifyEdges
                False,  # UnifyFaces
                True    # ConcatBSplines
            )

            unifier.SetSafeInputMode(
                True
            )

            unifier.SetLinearTolerance(
                tolerance
            )

            unifier.SetAngularTolerance(
                math.radians(
                    angTolerance
                )
            )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Protect every Vertex belonging to a genuinely curved Edge.
        #
        # ShapeUpgrade_UnifySameDomain.KeepShape(vertex) prevents connected
        # Edges from being merged through that Vertex.
        #
        # Therefore:
        #
        # linear -- linear
        #          ^
        #          eligible
        #
        # curved -- linear
        #          ^
        #          protected
        #
        # curved -- curved
        #          ^
        #          protected
        # ------------------------------------------------------------------

        curved_edge_count = 0

        for occ_edge in original_edges:

            backend_edge = Edge.ByOcctShape(
                occ_edge
            )

            if not isinstance(
                backend_edge,
                Edge
            ):
                return None

            try:
                linear = EdgeUtility.IsLinear(
                    backend_edge,
                    tolerance=tolerance
                )
            except Exception:
                return None

            if linear:
                continue

            curved_edge_count += 1

            vertex_explorer = TopExp_Explorer(
                occ_edge,
                TopAbs_VERTEX
            )

            while vertex_explorer.More():

                try:
                    vertex = topods.Vertex(
                        vertex_explorer.Current()
                    )

                    unifier.KeepShape(
                        vertex
                    )

                except Exception:
                    return None

                vertex_explorer.Next()

        # ------------------------------------------------------------------
        # Perform native edge unification.
        # ------------------------------------------------------------------

        try:
            unifier.Build()

            result_shape = unifier.Shape()

            if (
                result_shape is None
                or result_shape.IsNull()
            ):
                return None

            result_face = topods.Face(
                result_shape
            )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Validate the resulting BRep.
        # ------------------------------------------------------------------

        try:
            analyzer = BRepCheck_Analyzer(
                result_face
            )

            if not analyzer.IsValid():
                return None

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Count output Edges.
        #
        # If nothing was actually removed, return the ORIGINAL Face. This avoids
        # replacing it with a newly generated but geometrically identical shape.
        # ------------------------------------------------------------------

        result_edges = []

        explorer = TopExp_Explorer(
            result_face,
            TopAbs_EDGE
        )

        while explorer.More():

            try:
                edge = topods.Edge(
                    explorer.Current()
                )

                if not any(
                    edge.IsSame(existing)
                    for existing in result_edges
                ):
                    result_edges.append(edge)

            except Exception:
                pass

            explorer.Next()

        if len(result_edges) >= len(original_edges):
            return face

        # ------------------------------------------------------------------
        # Defensive check: the number of genuinely curved Edges must not have
        # decreased.
        # ------------------------------------------------------------------

        result_curved_count = 0

        for occ_edge in result_edges:

            backend_edge = Edge.ByOcctShape(
                occ_edge
            )

            if not isinstance(
                backend_edge,
                Edge
            ):
                return None

            if not EdgeUtility.IsLinear(
                backend_edge,
                tolerance=tolerance
            ):
                result_curved_count += 1

        if result_curved_count != curved_edge_count:
            return None

        # ------------------------------------------------------------------
        # Wrap the SAME native Face support surface with its simplified trims.
        # ------------------------------------------------------------------

        result = Face.ByOcctShape(
            result_face
        )

        if not isinstance(
            result,
            Face
        ):
            return None

        return _wrap_metadata(
            face,
            result
        )

    @staticmethod
    def TangentsAtParameters(
        face,
        u=0.5,
        v=0.5,
        tolerance: float = 0.0001
    ):
        """Return location-aware world-space U and V unit tangent directions."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None

        _, raw_u, raw_v, _, _, _, _ = mapped
        tol = _face_tolerance(tolerance)
        data = _surface_d1(face, raw_u, raw_v)
        if data is None:
            return None

        _, derivative_u, derivative_v = data

        tangent_u = _normalized_vector(
            [float(derivative_u.X()), float(derivative_u.Y()), float(derivative_u.Z())],
            tolerance=tol,
        )
        tangent_v = _normalized_vector(
            [float(derivative_v.X()), float(derivative_v.Y()), float(derivative_v.Z())],
            tolerance=tol,
        )

        if tangent_u is None or tangent_v is None:
            return None

        return [tangent_u, tangent_v]

    @staticmethod
    def Triangulate(face, deflection, outputFaces):
        """
        Triangulates the input Face using OCCT's native face triangulation.

        The resulting triangular Faces are appended to the input outputFaces list.
        Internal boundaries are respected because triangulation is obtained from
        the complete OCCT TopoDS_Face rather than from its individual wires.

        Parameters
        ----------
        face : Face
            The input PythonOCC backend Face.
        deflection : float
            The desired linear meshing deflection.
        outputFaces : list
            The list to which the resulting triangular Faces are appended.

        Returns
        -------
        int
            Returns 0 on success.

        Raises
        ------
        RuntimeError
            If the input Face cannot be triangulated.
        """
        if not isinstance(face, Face):
            raise RuntimeError(
                "FaceUtility.Triangulate - The input face is not a valid Face."
            )

        if not isinstance(outputFaces, list):
            raise RuntimeError(
                "FaceUtility.Triangulate - The outputFaces parameter is not a list."
            )

        shape = getattr(
            face,
            "shape",
            None
        )

        if shape is None:
            raise RuntimeError(
                "FaceUtility.Triangulate - The input Face has no OCCT shape."
            )

        try:
            if shape.IsNull():
                raise RuntimeError(
                    "FaceUtility.Triangulate - The input Face has a null OCCT shape."
                )
        except AttributeError:
            pass

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_MakePolygon,
            )
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TopAbs import TopAbs_REVERSED
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods

        except Exception as error:
            raise RuntimeError(
                "FaceUtility.Triangulate - Required PythonOCC modules "
                "could not be imported."
            ) from error

        try:
            occ_face = topods.Face(
                shape
            )

        except Exception as error:
            raise RuntimeError(
                "FaceUtility.Triangulate - Could not convert the input "
                "shape to a TopoDS_Face."
            ) from error

        # ------------------------------------------------------------------
        # Mesh the complete Face.
        #
        # A strictly zero deflection is not useful to BRepMesh, so impose a
        # small positive floor. The public Face.Triangulate method historically
        # tries values starting at zero.
        # ------------------------------------------------------------------

        try:
            linear_deflection = max(
                abs(
                    float(
                        deflection
                    )
                ),
                1.0e-6
            )

        except Exception:
            linear_deflection = 1.0e-6

        try:
            mesher = BRepMesh_IncrementalMesh(
                occ_face,
                linear_deflection,
                False,
                0.5,
                True
            )

            try:
                mesher.Perform()
            except Exception:
                pass

            if hasattr(
                mesher,
                "IsDone"
            ):
                if not mesher.IsDone():
                    raise RuntimeError(
                        "FaceUtility.Triangulate - OCCT meshing did not complete."
                    )

        except Exception as error:
            raise RuntimeError(
                "FaceUtility.Triangulate - OCCT could not mesh the input Face."
            ) from error

        # ------------------------------------------------------------------
        # Retrieve the triangulation belonging to the COMPLETE TopoDS_Face.
        #
        # This is important for Faces with holes. OCCT's face triangulation
        # represents the material domain of the Face and excludes its internal
        # boundary regions.
        # ------------------------------------------------------------------

        location = TopLoc_Location()

        try:
            triangulation = BRep_Tool.Triangulation(
                occ_face,
                location
            )

        except Exception as error:
            raise RuntimeError(
                "FaceUtility.Triangulate - Could not retrieve the OCCT "
                "triangulation."
            ) from error

        if triangulation is None:
            raise RuntimeError(
                "FaceUtility.Triangulate - OCCT returned no triangulation."
            )

        try:
            if hasattr(
                triangulation,
                "IsNull"
            ):
                if triangulation.IsNull():
                    raise RuntimeError(
                        "FaceUtility.Triangulate - OCCT returned a null "
                        "triangulation."
                    )
        except RuntimeError:
            raise
        except Exception:
            pass

        try:
            triangle_count = triangulation.NbTriangles()

        except Exception as error:
            raise RuntimeError(
                "FaceUtility.Triangulate - Could not query the OCCT "
                "triangulation."
            ) from error

        if triangle_count < 1:
            raise RuntimeError(
                "FaceUtility.Triangulate - OCCT triangulation contains no triangles."
            )

        # ------------------------------------------------------------------
        # OCCT triangulation nodes are expressed in the triangulation's local
        # coordinate system. Apply its TopLoc_Location transformation before
        # constructing backend Faces.
        # ------------------------------------------------------------------

        try:
            transformation = location.Transformation()
            location_is_identity = location.IsIdentity()

        except Exception:
            transformation = None
            location_is_identity = True

        def world_point(index):
            point = triangulation.Node(
                index
            )

            result = gp_Pnt(
                point.X(),
                point.Y(),
                point.Z()
            )

            if (
                not location_is_identity
                and transformation is not None
            ):
                result.Transform(
                    transformation
                )

            return result

        # ------------------------------------------------------------------
        # Build backend triangular Faces.
        #
        # Build the complete result locally first. Nothing is appended to the
        # caller's list unless the entire triangulation succeeds.
        # ------------------------------------------------------------------

        triangles = []

        reversed_face = (
            occ_face.Orientation()
            == TopAbs_REVERSED
        )

        for index in range(
            1,
            triangle_count + 1
        ):

            try:
                node_a, node_b, node_c = triangulation.Triangle(
                    index
                ).Get()

            except Exception as error:
                raise RuntimeError(
                    "FaceUtility.Triangulate - Could not retrieve an OCCT "
                    "triangle."
                ) from error

            # Preserve the Face orientation.
            if reversed_face:
                node_b, node_c = node_c, node_b

            point_a = world_point(
                node_a
            )

            point_b = world_point(
                node_b
            )

            point_c = world_point(
                node_c
            )

            # --------------------------------------------------------------
            # Reject numerically degenerate triangles.
            # --------------------------------------------------------------

            ab = (
                point_b.X() - point_a.X(),
                point_b.Y() - point_a.Y(),
                point_b.Z() - point_a.Z()
            )

            ac = (
                point_c.X() - point_a.X(),
                point_c.Y() - point_a.Y(),
                point_c.Z() - point_a.Z()
            )

            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0]
            )

            area_squared = (
                cross[0] * cross[0]
                + cross[1] * cross[1]
                + cross[2] * cross[2]
            )

            if area_squared <= 1.0e-24:
                continue

            # --------------------------------------------------------------
            # Build a genuine OCCT triangular Face.
            # --------------------------------------------------------------

            polygon_builder = BRepBuilderAPI_MakePolygon()

            polygon_builder.Add(
                point_a
            )

            polygon_builder.Add(
                point_b
            )

            polygon_builder.Add(
                point_c
            )

            polygon_builder.Close()

            if not polygon_builder.IsDone():
                raise RuntimeError(
                    "FaceUtility.Triangulate - Could not construct a "
                    "triangle boundary."
                )

            face_builder = BRepBuilderAPI_MakeFace(
                polygon_builder.Wire()
            )

            if not face_builder.IsDone():
                raise RuntimeError(
                    "FaceUtility.Triangulate - Could not construct a "
                    "triangular Face."
                )

            triangle_shape = face_builder.Face()

            triangle = None

            try:
                triangle = Face.ByOcctShape(
                    triangle_shape
                )

            except Exception:
                triangle = None

            if triangle is None:
                try:
                    triangle = Face(
                        shape=triangle_shape
                    )

                except Exception:
                    triangle = None

            if not isinstance(
                triangle,
                Face
            ):
                raise RuntimeError(
                    "FaceUtility.Triangulate - Could not wrap a triangular "
                    "OCCT Face."
                )

            triangles.append(
                triangle
            )

        if len(triangles) == 0:
            raise RuntimeError(
                "FaceUtility.Triangulate - No valid triangular Faces were produced."
            )

        outputFaces.extend(
            triangles
        )

        return 0

    @staticmethod
    def IsPlanar(
        face,
        tolerance: float = 0.0001
    ):
        """
        Returns True when the actual supporting surface of the input Face is
        geometrically planar.

        This test recognizes planar B-spline and Bezier surfaces as planar; it
        does not rely only on the OCCT surface type.
        """
        occ_face = _as_occ_face(face)

        if occ_face is None:
            return None

        tol = _face_tolerance(
            tolerance
        )

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomLib import GeomLib_IsPlanarSurface

            surface = BRep_Tool.Surface(
                occ_face
            )

            if surface is None:
                return None

            checker = GeomLib_IsPlanarSurface(
                surface,
                tol,
            )

            return bool(
                checker.IsPlanar()
            )

        except Exception:
            return None

    @staticmethod
    def IsCoplanar(faceA, faceB, tolerance: float = 0.0001):
        """
        Return True when two Faces lie on the same geometric plane.

        The test is based on geometric planarity rather than only the native
        OCCT surface type, so planar B-spline/NURBS and Bezier Faces are handled
        correctly as well as native planar Faces.
        """
        if not isinstance(faceA, Face) or not isinstance(faceB, Face):
            return None

        tol = _face_tolerance(tolerance)

        if FaceUtility.IsPlanar(faceA, tolerance=tol) is not True:
            return False
        if FaceUtility.IsPlanar(faceB, tolerance=tol) is not True:
            return False

        normal_a = FaceUtility.NormalAtParameters(faceA, 0.5, 0.5)
        normal_b = FaceUtility.NormalAtParameters(faceB, 0.5, 0.5)

        if normal_a is None or normal_b is None:
            return None

        try:
            ax, ay, az = [float(value) for value in normal_a]
            bx, by, bz = [float(value) for value in normal_b]

            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx

            if math.sqrt(cx * cx + cy * cy + cz * cz) > tol:
                return False

            point_a = FaceUtility.VertexAtParameters(faceA, 0.5, 0.5)
            point_b = FaceUtility.VertexAtParameters(faceB, 0.5, 0.5)

            if not isinstance(point_a, Vertex) or not isinstance(point_b, Vertex):
                return None

            dx = float(point_b.x) - float(point_a.x)
            dy = float(point_b.y) - float(point_a.y)
            dz = float(point_b.z) - float(point_a.z)

            distance = abs(dx * ax + dy * ay + dz * az)
            return distance <= tol

        except Exception:
            return None

    @staticmethod
    def InternalVertex(face, tolerance=0.0001):
        """Return a Vertex guaranteed to lie on the support and inside the trimmed Face."""
        if not isinstance(face, Face):
            return None

        from .topology import Topology as _Topology

        tol = _face_tolerance(tolerance)
        centroid = _Topology.CenterOfMass(face)

        if isinstance(centroid, Vertex):
            params = FaceUtility.ParametersAtVertex(face, centroid, tolerance=tol)
            if params is not None and FaceUtility.IsInside(face, centroid, tolerance=tol):
                return centroid

        # Evaluate candidate points directly on the exact support so every
        # returned candidate is guaranteed to lie on the surface.
        for v in (0.5, 0.25, 0.75, 0.1, 0.9):
            for u in (0.5, 0.25, 0.75, 0.1, 0.9):
                candidate = FaceUtility.VertexAtParameters(face, u, v)
                if isinstance(candidate, Vertex) and FaceUtility.IsInside(face, candidate, tolerance=tol):
                    return candidate

        return None

    @staticmethod
    def TrimByWire(face,
                   wire,
                   reverse: bool = False,
                   tolerance: float = 0.0001
                   ):
        """
        Trims a Face by a closed Wire on the Face's supporting surface.

        If the input Wire already carries valid p-curves on the Face, it is used
        directly. Otherwise, the Wire is projected normally onto the Face using
        OCCT's native BRepOffsetAPI_NormalProjection.

        The resulting surface Wire is then used to split the original Face with
        BRepFeat_SplitShape. This preserves the original supporting surface,
        including B-spline and NURBS surfaces.

        Parameters
        ----------
        face : Face
            The input Face.
        wire : Wire
            The closed trimming Wire.
        reverse : bool , optional
            If False, returns the portion inside the trimming Wire. If True,
            returns the complementary portion. Default is False.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.

        Returns
        -------
        Face
            The trimmed Face, or None if the operation fails.

        """
        occ_face = _as_occ_face(face)
        occ_wire = _as_occ_wire(wire)

        if occ_face is None or occ_wire is None:
            return None

        tol = _face_tolerance(tolerance)

        try:
            from OCC.Core.BRep import BRep_Tool

            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_MakeWire,
            )

            from OCC.Core.BRepFeat import (
                BRepFeat_SplitShape,
            )

            from OCC.Core.BRepOffsetAPI import (
                BRepOffsetAPI_NormalProjection,
            )

            from OCC.Core.BRepCheck import (
                BRepCheck_Analyzer,
            )

            from OCC.Core.ShapeAnalysis import (
                ShapeAnalysis_Edge,
            )

            from OCC.Core.TopAbs import (
                TopAbs_EDGE,
                TopAbs_FACE,
                TopAbs_WIRE,
            )

            from OCC.Core.TopoDS import (
                topods,
            )

        except Exception:
            return None

        # ------------------------------------------------------------------
        # Retrieve the original supporting surface.
        # ------------------------------------------------------------------

        try:
            surface = BRep_Tool.Surface(occ_face)
        except Exception:
            surface = None

        if surface is None:
            return None

        # ------------------------------------------------------------------
        # Determine whether the supplied Wire already belongs parametrically
        # to this Face.
        # ------------------------------------------------------------------

        def wire_has_pcurves(test_wire):
            try:
                analysis = ShapeAnalysis_Edge()

                edge_shapes = _explore_shapes(
                    test_wire,
                    TopAbs_EDGE,
                )

                if len(edge_shapes) < 1:
                    return False

                for edge_shape in edge_shapes:
                    edge = topods.Edge(edge_shape)

                    try:
                        if not analysis.HasPCurve(
                            edge,
                            occ_face,
                        ):
                            return False
                    except Exception:
                        return False

                return True

            except Exception:
                return False

        # ------------------------------------------------------------------
        # If necessary, project the trimming Wire onto the actual Face.
        #
        # BRepOffsetAPI_NormalProjection returns edges/wires that genuinely
        # belong to the target Face and have the required surface geometry.
        # ------------------------------------------------------------------

        if wire_has_pcurves(occ_wire):

            surface_wire = occ_wire

        else:

            try:
                projector = BRepOffsetAPI_NormalProjection(
                    occ_face
                )

                # Restrict the projection to the bounds of the target Face.
                try:
                    projector.SetLimit(True)
                except Exception:
                    pass

                # Request actual 3D curves as well as their p-curves.
                try:
                    projector.Compute3d(True)
                except Exception:
                    pass

                projector.Add(
                    occ_wire
                )

                projector.Build()

                if not projector.IsDone():
                    return None

                projected_shape = projector.Projection()

            except Exception:
                return None

            if (
                projected_shape is None
                or projected_shape.IsNull()
            ):
                return None

            # NormalProjection normally returns a compound containing one or
            # more oriented Wires.
            try:
                wire_shapes = _explore_shapes(
                    projected_shape,
                    TopAbs_WIRE,
                )
            except Exception:
                wire_shapes = []

            if len(wire_shapes) == 1:

                try:
                    surface_wire = topods.Wire(
                        wire_shapes[0]
                    )
                except Exception:
                    return None

            elif len(wire_shapes) > 1:

                # More than one projection means the projection is ambiguous or
                # fragmented. TrimByWire cannot choose one silently.
                return None

            else:

                # Defensive fallback for PythonOCC builds that expose projected
                # edges without the enclosing Wire.
                try:
                    edge_shapes = _explore_shapes(
                        projected_shape,
                        TopAbs_EDGE,
                    )

                    if len(edge_shapes) < 1:
                        return None

                    wire_maker = BRepBuilderAPI_MakeWire()

                    for edge_shape in edge_shapes:
                        wire_maker.Add(
                            topods.Edge(edge_shape)
                        )

                    if not wire_maker.IsDone():
                        return None

                    surface_wire = wire_maker.Wire()

                except Exception:
                    return None

        if (
            surface_wire is None
            or surface_wire.IsNull()
        ):
            return None

        # ------------------------------------------------------------------
        # Verify that every Edge now has a p-curve on the original Face.
        # ------------------------------------------------------------------

        if not wire_has_pcurves(surface_wire):
            return None

        # ------------------------------------------------------------------
        # Build the trimming region ON THE SAME SUPPORTING SURFACE.
        #
        # This is used only for classifying the resulting split pieces.
        # It is not used as a boolean tool.
        # ------------------------------------------------------------------

        try:
            region_maker = BRepBuilderAPI_MakeFace(
                surface,
                surface_wire,
                True,
            )

            if not region_maker.IsDone():
                return None

            occ_region_face = region_maker.Face()

            if (
                occ_region_face is None
                or occ_region_face.IsNull()
            ):
                return None

            region_face = Face.ByOcctShape(
                occ_region_face
            )

        except Exception:
            return None

        if not isinstance(region_face, Face):
            return None

        # ------------------------------------------------------------------
        # Check that the trimming region itself is valid.
        # ------------------------------------------------------------------

        try:
            analyzer = BRepCheck_Analyzer(
                occ_region_face
            )

            if not analyzer.IsValid():
                return None

        except Exception:
            pass

        # ------------------------------------------------------------------
        # Split the ORIGINAL Face.
        #
        # BRepFeat_SplitShape operates locally on the supplied Face and preserves
        # its supporting surface.
        # ------------------------------------------------------------------

        try:
            splitter = BRepFeat_SplitShape(
                occ_face
            )

            splitter.Add(
                surface_wire,
                occ_face,
            )

            splitter.Build()

            if not splitter.IsDone():
                return None

            split_shape = splitter.Shape()

        except Exception:
            return None

        if (
            split_shape is None
            or split_shape.IsNull()
        ):
            return None

        # ------------------------------------------------------------------
        # Retrieve the resulting pieces.
        # ------------------------------------------------------------------

        try:
            candidate_shapes = _explore_shapes(
                split_shape,
                TopAbs_FACE,
            )
        except Exception:
            candidate_shapes = []

        # A successful closed trim should actually split the Face.
        if len(candidate_shapes) < 2:
            return None

        selected = []

        # ------------------------------------------------------------------
        # Select the inside or outside part geometrically.
        #
        # Do not depend on trimming-wire orientation or SplitShape's "left"
        # convention. Instead classify a strict internal point from every
        # resulting Face against the trimming-region Face.
        # ------------------------------------------------------------------

        for candidate_shape in candidate_shapes:

            try:
                occ_candidate = topods.Face(
                    candidate_shape
                )
            except Exception:
                continue

            try:
                analyzer = BRepCheck_Analyzer(
                    occ_candidate
                )

                if not analyzer.IsValid():
                    continue

            except Exception:
                pass

            candidate = Face.ByOcctShape(
                occ_candidate
            )

            if not isinstance(candidate, Face):
                continue

            representative = FaceUtility.InternalVertex(
                candidate,
                tol,
            )

            if not isinstance(representative, Vertex):
                continue

            inside_trim = FaceUtility.IsInside(
                region_face,
                representative,
                tol,
            )

            if reverse:
                keep = not inside_trim
            else:
                keep = inside_trim

            if keep:
                selected.append(candidate)

        # ------------------------------------------------------------------
        # Face.TrimByWire returns one Face.
        #
        # A simple closed trimming loop should produce exactly one requested
        # result. Do not silently choose between disconnected regions.
        # ------------------------------------------------------------------

        if len(selected) != 1:
            return None

        return _wrap_metadata(
            face,
            selected[0],
        )
    
    @staticmethod
    def VertexAtParameters(face, u=0.5, v=0.5):
        """Return a world-space Vertex at normalized UV parameters."""
        mapped = _normalized_to_raw(face, u, v)
        if mapped is None:
            return None
        _, raw_u, raw_v, _, _, _, _ = mapped

        adaptor = _surface_adaptor(face)
        if adaptor is None:
            return None

        try:
            point = adaptor.Value(float(raw_u), float(raw_v))
            return Vertex.ByCoordinates(point.X(), point.Y(), point.Z())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Compatibility API wiring
# ---------------------------------------------------------------------------





def _face_internal_vertex(self, tolerance=0.0001, silent=False):
    return FaceUtility.InternalVertex(self, tolerance=tolerance)


# Plain instance method, not @staticmethod: must support the instance-bound
# Core.InstanceCall convention (face.InternalVertex(tolerance)), which a
# staticmethod-wrapped lambda would break (see HANDOFF.md item 1).
Face.InternalVertex = _face_internal_vertex




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
