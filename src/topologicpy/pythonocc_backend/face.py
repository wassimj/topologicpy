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
    shape = getattr(face, "shape", None)
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
    shape = getattr(wire, "shape", None)
    if _is_null_shape(shape):
        return None
    try:
        from OCC.Core.TopoDS import topods
        return topods.Wire(shape)
    except Exception:
        return None

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


def _normalized_vector(values, tolerance=1.0e-12):
    """Return a normalized 3D vector, or None if its magnitude is too small."""
    try:
        vector = [
            float(values[0]),
            float(values[1]),
            float(values[2]),
        ]
    except Exception:
        return None

    if not all(math.isfinite(value) for value in vector):
        return None

    magnitude = math.sqrt(
        sum(value * value for value in vector)
    )

    if magnitude <= _face_tolerance(tolerance):
        return None

    return [
        value / magnitude
        for value in vector
    ]


def _expanded_knot_data(values):
    """
    Convert an expanded knot vector to OCCT unique knots and multiplicities.

    Returns
    -------
    tuple
        ``(unique_knots, multiplicities)``, or None for invalid input.
    """
    if not isinstance(values, (list, tuple)):
        return None

    try:
        expanded = [
            float(value)
            for value in values
        ]
    except Exception:
        return None

    if len(expanded) < 2:
        return None

    if any(
        not math.isfinite(value)
        for value in expanded
    ):
        return None

    if any(
        expanded[index] > expanded[index + 1]
        for index in range(len(expanded) - 1)
    ):
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
        # OCCT requires hole wires to have the opposite orientation to the
        # external boundary. Do not modify the original Wire wrapper or its
        # stored TopoDS_Wire; create an orientation-adjusted TopoDS_Wire copy.
        # ------------------------------------------------------------------

        try:
            external_orientation = external_wire.Orientation()
        except Exception:
            external_orientation = None

        for internalBoundary in internalBoundaries:

            internal_shape = getattr(
                internalBoundary,
                "shape",
                None
            )

            if internal_shape is None:
                try:
                    internal_shape = internalBoundary.GetOcctShape()
                except Exception:
                    return None

            try:
                if internal_shape.IsNull():
                    return None
            except Exception:
                pass

            try:
                internal_wire = topods.Wire(
                    internal_shape
                )
            except Exception:
                return None

            # --------------------------------------------------------------
            # The internal boundary must oppose the external boundary.
            #
            # Use Reversed() rather than Reverse() so the input Wire's native
            # orientation is not mutated.
            # --------------------------------------------------------------

            try:
                internal_orientation = internal_wire.Orientation()

                if (
                    external_orientation is not None
                    and internal_orientation == external_orientation
                ):
                    internal_wire = topods.Wire(
                        internal_wire.Reversed()
                    )

            except Exception:
                # If orientation inspection is unavailable, conservatively
                # reverse the hole. This matches the normal case where both
                # independently created wires initially have FORWARD orientation.
                try:
                    internal_wire = topods.Wire(
                        internal_wire.Reversed()
                    )
                except Exception:
                    return None

            try:
                builder.Add(
                    internal_wire
                )
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
        """
        Returns the external boundary wire of the face.

        Returns
        -------
        Wire
            The external boundary wire, or None if it cannot be determined.
        """
        if _is_null_shape(getattr(self, "shape", None)):
            return self.external if isinstance(self.external, Wire) else None

        try:
            from OCC.Core.BRepTools import breptools
            from OCC.Core.TopoDS import topods

            occ_face = topods.Face(self.shape)
            occ_wire = breptools.OuterWire(occ_face)

            if occ_wire.IsNull():
                return None

            return Topology.ByOcctShape(occ_wire)

        except Exception:
            return None

    def InternalBoundaries(self, wires=None):
        """
        Returns the internal boundary wires of the face.

        Parameters
        ----------
        wires : list , optional
            If supplied, the resulting wires are appended to this list and the
            method returns 0.

        Returns
        -------
        list
            The internal boundary wires.
        """
        if _is_null_shape(getattr(self, "shape", None)):
            result = list(getattr(self, "internals", []) or [])

        else:
            result = []

            try:
                from OCC.Core.BRepTools import breptools
                from OCC.Core.TopAbs import TopAbs_WIRE
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopoDS import topods

                occ_face = topods.Face(self.shape)
                outer = breptools.OuterWire(occ_face)

                explorer = TopExp_Explorer(
                    occ_face,
                    TopAbs_WIRE
                )

                while explorer.More():
                    occ_wire = explorer.Current()

                    is_external = (
                        not outer.IsNull()
                        and occ_wire.IsSame(outer)
                    )

                    if not is_external:
                        wire = Topology.ByOcctShape(
                            occ_wire
                        )

                        if isinstance(wire, Wire):
                            result.append(wire)

                    explorer.Next()

            except Exception:
                result = []

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
        """
        Returns the wires of the face with the external boundary first followed
        by the internal boundaries.

        Parameters
        ----------
        hostTopology : object , optional
            Included for backend API compatibility.
        wires : list , optional
            If supplied, the resulting wires are appended to this list and the
            method returns 0.

        Returns
        -------
        list
            The face wires.
        """
        if _is_null_shape(getattr(self, "shape", None)):
            result = []

            if isinstance(self.external, Wire):
                result.append(self.external)

            result.extend(
                [
                    wire
                    for wire in (getattr(self, "internals", []) or [])
                    if isinstance(wire, Wire)
                ]
            )

        else:
            result = []

            try:
                from OCC.Core.BRepTools import breptools
                from OCC.Core.TopAbs import TopAbs_WIRE
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopoDS import topods

                occ_face = topods.Face(self.shape)
                outer = breptools.OuterWire(occ_face)

                external = None
                internals = []

                explorer = TopExp_Explorer(
                    occ_face,
                    TopAbs_WIRE
                )

                while explorer.More():
                    occ_wire = explorer.Current()

                    wire = Topology.ByOcctShape(
                        occ_wire
                    )

                    if isinstance(wire, Wire):

                        if (
                            not outer.IsNull()
                            and occ_wire.IsSame(outer)
                        ):
                            external = wire
                        else:
                            internals.append(wire)

                    explorer.Next()

                if external is not None:
                    result.append(external)

                result.extend(internals)

            except Exception:
                result = []

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
    def NormalAtParameters(face, u=0.5, v=0.5):
        """
        Returns the unit normal vector of the input face at the specified
        normalized UV parameters.

        Parameters
        ----------
        face : Face
            The input face.
        u : float , optional
            The normalized U parameter in the range [0, 1]. Default is 0.5.
        v : float , optional
            The normalized V parameter in the range [0, 1]. Default is 0.5.

        Returns
        -------
        list
            The normal vector [x, y, z], or None if the normal cannot be computed.
        """
        if not isinstance(face, Face):
            return None

        shape = getattr(face, "shape", None)
        if shape is None:
            return None

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepGProp import BRepGProp_Face
            from OCC.Core.GeomLProp import GeomLProp_SLProps
            from OCC.Core.TopAbs import TopAbs_REVERSED

            # Clamp normalized parameters.
            u = max(0.0, min(1.0, float(u)))
            v = max(0.0, min(1.0, float(v)))

            # Get the actual parametric bounds of the trimmed face.
            uMin, uMax, vMin, vMax = BRepGProp_Face(shape).Bounds()

            # Map TopologicPy's normalized [0, 1] parameters onto the
            # underlying OCCT surface parameter domain.
            actualU = uMin + u * (uMax - uMin)
            actualV = vMin + v * (vMax - vMin)

            surface = BRep_Tool.Surface(shape)
            if surface is None:
                return None

            props = GeomLProp_SLProps(
                surface,
                actualU,
                actualV,
                1,
                1.0e-9,
            )

            if not props.IsNormalDefined():
                return None

            normal = props.Normal()

            nx = float(normal.X())
            ny = float(normal.Y())
            nz = float(normal.Z())

            # GeomLProp returns the orientation of the underlying geometric
            # surface. A reversed TopoDS_Face has the opposite topological normal.
            if shape.Orientation() == TopAbs_REVERSED:
                nx = -nx
                ny = -ny
                nz = -nz

            length = math.sqrt(
                nx * nx +
                ny * ny +
                nz * nz
            )

            if length <= 1.0e-12:
                return None

            return [
                nx / length,
                ny / length,
                nz / length,
            ]

        except Exception:
            return None

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

            tangent_u = _normalized_vector(
                tangent_u,
                tolerance=tol,
            )

            tangent_v = _normalized_vector(
                tangent_v,
                tolerance=tol,
            )

            if tangent_u is None or tangent_v is None:
                return None

            return [
                tangent_u,
                tangent_v,
            ]

        except Exception:
            return None

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

    result = []
    fv_src = face.Vertices()
    candidates = []

    Topology.Shells(
        hostTopology,
        None,
        candidates
    )

    for shell in candidates:

        for shell_face in shell.Faces():

            fv = shell_face.Vertices()

            if (
                len(fv) == len(fv_src)
                and all(
                    any(
                        same_vertex(a, b)
                        for b in fv_src
                    )
                    for a in fv
                )
            ):
                result.append(shell)
                break

    if output is not None:
        output.extend(result)

    return 0

def _adjacent_cells(face, hostTopology, output):
    from .topology import Topology
    from .helpers import same_vertex

    if not isinstance(face, Face) or hostTopology is None:
        return 1

    result = []
    fv_src = face.Vertices()
    candidates = []

    Topology.Cells(
        hostTopology,
        None,
        candidates
    )

    for cell in candidates:

        for cell_face in cell.Faces():

            fv = cell_face.Vertices()

            if (
                len(fv) == len(fv_src)
                and all(
                    any(
                        same_vertex(a, b)
                        for b in fv_src
                    )
                    for a in fv
                )
            ):
                result.append(cell)
                break

    if output is not None:
        output.extend(result)

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




