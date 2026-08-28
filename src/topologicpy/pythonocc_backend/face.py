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

        # ------------------------------------------------------------------
        # Expanded knot vector -> unique OCCT knots + multiplicities.
        # ------------------------------------------------------------------

        def knot_data(values):
            try:
                values = [
                    float(value)
                    for value in values
                ]
            except Exception:
                return None

            if len(values) < 2:
                return None

            if any(
                not math.isfinite(value)
                for value in values
            ):
                return None

            if any(
                values[i] > values[i + 1]
                for i in range(len(values) - 1)
            ):
                return None

            unique = []
            multiplicities = []

            for value in values:
                if unique and value == unique[-1]:
                    multiplicities[-1] += 1
                else:
                    unique.append(value)
                    multiplicities.append(1)

            if len(unique) < 2:
                return None

            return unique, multiplicities

        u_data = knot_data(uKnots)
        v_data = knot_data(vKnots)

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

    @staticmethod
    def CurvatureAtParameters(
        face,
        u=0.5,
        v=0.5,
        tolerance: float = 0.0001
    ):
        """
        Returns native OCCT surface-curvature properties at normalized parameters.

        Signed principal and mean curvatures respect the topological orientation
        of the Face. Gaussian curvature is orientation-independent.
        """
        mapped = _normalized_to_raw(
            face,
            u,
            v,
        )

        if mapped is None:
            return None

        surface, raw_u, raw_v, _, _, _, _ = mapped

        tol = _face_tolerance(
            tolerance
        )

        try:
            from OCC.Core.GeomLProp import GeomLProp_SLProps
            from OCC.Core.TopAbs import TopAbs_REVERSED
            from OCC.Core.gp import gp_Dir

            properties = GeomLProp_SLProps(
                surface,
                raw_u,
                raw_v,
                2,
                tol,
            )

            if not properties.IsCurvatureDefined():
                return None

            maximum = float(
                properties.MaxCurvature()
            )

            minimum = float(
                properties.MinCurvature()
            )

            mean = float(
                properties.MeanCurvature()
            )

            gaussian = float(
                properties.GaussianCurvature()
            )

            is_umbilic = bool(
                properties.IsUmbilic()
            )

            maximum_direction = None
            minimum_direction = None

            try:
                max_dir = gp_Dir(
                    1.0,
                    0.0,
                    0.0,
                )

                min_dir = gp_Dir(
                    0.0,
                    1.0,
                    0.0,
                )

                properties.CurvatureDirections(
                    max_dir,
                    min_dir,
                )

                maximum_direction = [
                    float(max_dir.X()),
                    float(max_dir.Y()),
                    float(max_dir.Z()),
                ]

                minimum_direction = [
                    float(min_dir.X()),
                    float(min_dir.Y()),
                    float(min_dir.Z()),
                ]

            except Exception:
                maximum_direction = None
                minimum_direction = None

            occ_face = _as_occ_face(
                face
            )

            # Changing Face orientation reverses the normal. Principal and mean
            # curvature signs therefore reverse. Since kmax >= kmin, negation
            # also swaps which principal curvature is the maximum.
            if (
                occ_face is not None
                and occ_face.Orientation() == TopAbs_REVERSED
            ):
                old_maximum = maximum
                old_minimum = minimum

                maximum = -old_minimum
                minimum = -old_maximum

                mean = -mean

                maximum_direction, minimum_direction = (
                    minimum_direction,
                    maximum_direction,
                )

            values = [
                maximum,
                minimum,
                mean,
                gaussian,
            ]

            if not all(
                math.isfinite(value)
                for value in values
            ):
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

    def AdjacentFaces(self, hostTopology, output=None):
        """Return or populate Faces in a host topology that share an OCCT Edge.

        Adjacency is topological rather than merely geometric: a candidate Face is
        adjacent only when it contains an Edge that is ``IsSame`` as one of this
        Face's Edges.
        """
        occ_face = _as_occ_face(self)
        host_shape = _shape_from_topology(hostTopology)
        result = []
        if occ_face is None or _is_null_shape(host_shape):
            if output is not None:
                return 1
            return result

        try:
            from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
            source_edges = _explore_shapes(occ_face, TopAbs_EDGE)
            candidate_faces = _explore_shapes(host_shape, TopAbs_FACE)
            for candidate_shape in candidate_faces:
                if _same_shape(candidate_shape, occ_face):
                    continue
                candidate_edges = _explore_shapes(candidate_shape, TopAbs_EDGE)
                if any(
                    _same_shape(source_edge, candidate_edge)
                    for source_edge in source_edges
                    for candidate_edge in candidate_edges
                ):
                    candidate = Topology.ByOcctShape(candidate_shape)
                    if isinstance(candidate, Face):
                        result.append(candidate)
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
    def TangentsAtParameters(
        face,
        u=0.5,
        v=0.5,
        tolerance: float = 0.0001
    ):
        """
        Returns the normalized U and V parametric tangent directions at normalized
        surface parameters.
        """
        mapped = _normalized_to_raw(
            face,
            u,
            v,
        )

        if mapped is None:
            return None

        surface, raw_u, raw_v, _, _, _, _ = mapped

        tol = _face_tolerance(
            tolerance
        )

        try:
            from OCC.Core.GeomLProp import GeomLProp_SLProps

            properties = GeomLProp_SLProps(
                surface,
                raw_u,
                raw_v,
                1,
                tol,
            )

            derivative_u = properties.D1U()
            derivative_v = properties.D1V()

            tangent_u = [
                float(derivative_u.X()),
                float(derivative_u.Y()),
                float(derivative_u.Z()),
            ]

            tangent_v = [
                float(derivative_v.X()),
                float(derivative_v.Y()),
                float(derivative_v.Z()),
            ]

            magnitude_u = math.sqrt(
                sum(
                    value * value
                    for value in tangent_u
                )
            )

            magnitude_v = math.sqrt(
                sum(
                    value * value
                    for value in tangent_v
                )
            )

            if (
                magnitude_u <= tol
                or magnitude_v <= tol
            ):
                return None

            tangent_u = [
                value / magnitude_u
                for value in tangent_u
            ]

            tangent_v = [
                value / magnitude_v
                for value in tangent_v
            ]

            return [
                tangent_u,
                tangent_v,
            ]

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
