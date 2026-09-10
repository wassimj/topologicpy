# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import math


class Grid:
    """
    Creates grids on planar and non-planar Faces.

    Grid.OnFace is intentionally the only public method.
    """

    # -------------------------------------------------------------------------
    # General helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _Tolerance(tolerance=0.0001):
        try:
            return max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            return 0.0001

    @staticmethod
    def _Coordinates(vertex):
        from topologicpy.Vertex import Vertex
        if vertex is None:
            return None
        try:
            return [float(x) for x in Vertex.Coordinates(vertex, mantissa=None)]
        except Exception:
            try:
                return [float(x) for x in Vertex.Coordinates(vertex)]
            except Exception:
                return None

    @staticmethod
    def _Add(a, b):
        return [a[i] + b[i] for i in range(3)]

    @staticmethod
    def _Subtract(a, b):
        return [a[i] - b[i] for i in range(3)]

    @staticmethod
    def _Scale(v, s):
        return [v[i] * s for i in range(3)]

    @staticmethod
    def _Dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    @staticmethod
    def _Normalize(v, tolerance=0.0001):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            return None
        try:
            v = [float(x) for x in v]
        except Exception:
            return None
        m = math.sqrt(Grid._Dot(v, v))
        if m <= Grid._Tolerance(tolerance):
            return None
        return [x / m for x in v]

    @staticmethod
    def _Basis(normal, xDirection=None, tolerance=0.0001):
        """Return a stable world-oriented basis for a planar Face."""
        n = Grid._Normalize(normal, tolerance)
        if n is None:
            return None

        candidates = []
        if isinstance(xDirection, (list, tuple)) and len(xDirection) == 3:
            candidates.append(xDirection)
        candidates.extend(([1, 0, 0], [0, 1, 0], [0, 0, 1]))

        u = None
        for candidate in candidates:
            d = Grid._Normalize(candidate, tolerance)
            if d is None:
                continue
            p = Grid._Subtract(d, Grid._Scale(n, Grid._Dot(d, n)))
            u = Grid._Normalize(p, tolerance)
            if u is not None:
                break
        if u is None:
            return None

        # Prefer world Z on wall-like Faces and world Y on horizontal Faces.
        for candidate in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            d = Grid._Normalize(candidate, tolerance)
            p = Grid._Subtract(d, Grid._Scale(n, Grid._Dot(d, n)))
            p = Grid._Subtract(p, Grid._Scale(u, Grid._Dot(p, u)))
            v = Grid._Normalize(p, tolerance)
            if v is not None:
                return u, v
        return None

    @staticmethod
    def _SetDictionary(topology, metadata):
        from topologicpy.Topology import Topology
        metadata = {str(k): v for k, v in metadata.items() if v is not None}
        try:
            result = Topology.SetDictionary(topology, metadata, silent=True)
            return result if result is not None else topology
        except Exception:
            return topology

    @staticmethod
    def _Labels(count, specification):
        if specification is None or specification is False:
            return [None] * count
        if specification is True:
            specification = "numbers"
        if isinstance(specification, (list, tuple)):
            result = [str(x) for x in specification[:count]]
            return result + [None] * (count - len(result))
        if isinstance(specification, str):
            mode = specification.lower()
            if mode in ("numbers", "numeric"):
                return [str(i + 1) for i in range(count)]
            if mode in ("letters", "alpha", "alphabetic"):
                result = []
                for index in range(count):
                    n = index + 1
                    label = ""
                    while n:
                        n, r = divmod(n - 1, 26)
                        label = chr(65 + r) + label
                    result.append(label)
                return result
            return [f"{specification}{i + 1}" for i in range(count)]
        return [None] * count

    @staticmethod
    def _Unique(values, tolerance=1.0e-10):
        result = []
        for value in sorted(float(x) for x in values):
            if not result or abs(value - result[-1]) > tolerance:
                result.append(value)
        return result

    # -------------------------------------------------------------------------
    # Placement helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _RegularDistances(length, spacing, alignment, includeBoundary, tolerance):
        """Return regularly spaced physical distances in [0, length]."""
        tol = Grid._Tolerance(tolerance)
        try:
            length = float(length)
            spacing = abs(float(spacing))
        except Exception:
            return None, None
        if length <= tol or spacing <= tol:
            return None, None

        alignment = str(alignment).lower()
        if alignment == "center":
            datum = 0.5 * length
        elif alignment in ("start", "left", "bottom", "lower"):
            datum = 0.0
        elif alignment in ("end", "right", "top", "upper"):
            datum = length
        else:
            return None, None

        values = [datum]
        k = 1
        while datum - k * spacing >= -tol:
            values.append(datum - k * spacing)
            k += 1
        k = 1
        while datum + k * spacing <= length + tol:
            values.append(datum + k * spacing)
            k += 1

        if includeBoundary:
            values.extend((0.0, length))

        values = [min(length, max(0.0, x)) for x in values
                  if -tol <= x <= length + tol]
        return Grid._Unique(values, max(tol, 1.0e-10)), datum

    @staticmethod
    def _ExplicitValues(values, includeBoundary):
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            values = [values]
        result = []
        for value in values:
            try:
                value = min(1.0, max(0.0, float(value)))
                if math.isfinite(value):
                    result.append(value)
            except Exception:
                pass
        if includeBoundary:
            result.extend((0.0, 1.0))
        return Grid._Unique(result)

    @staticmethod
    def _DivisionValues(divisions, includeBoundary):
        try:
            divisions = int(divisions)
        except Exception:
            return None
        if divisions < 1:
            return None
        values = [i / divisions for i in range(divisions + 1)]
        return values if includeBoundary else values[1:-1]

    # -------------------------------------------------------------------------
    # Surface metric
    # -------------------------------------------------------------------------

    @staticmethod
    def _SurfaceMetric(face, axis, samples):
        """
        Return (normalized parameters, cumulative physical lengths) along the
        central reference curve.

        axis="u" varies U at V=0.5.
        axis="v" varies V at U=0.5.
        """
        from topologicpy.Face import Face

        samples = max(16, min(int(samples), 4096))
        parameters = [i / samples for i in range(samples + 1)]
        points = []

        for t in parameters:
            try:
                vertex = Face.VertexByParameters(
                    face,
                    u=t if axis == "u" else 0.5,
                    v=0.5 if axis == "u" else t,
                )
            except Exception:
                return None
            point = Grid._Coordinates(vertex)
            if point is None:
                return None
            points.append(point)

        lengths = [0.0]
        for a, b in zip(points[:-1], points[1:]):
            d = Grid._Subtract(b, a)
            lengths.append(lengths[-1] + math.sqrt(Grid._Dot(d, d)))

        return (parameters, lengths) if lengths[-1] > 1.0e-12 else None

    @staticmethod
    def _ParameterAtLength(distance, metric):
        parameters, lengths = metric
        total = lengths[-1]
        distance = min(total, max(0.0, float(distance)))
        if distance <= 0:
            return 0.0
        if distance >= total:
            return 1.0

        lo, hi = 0, len(lengths) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if lengths[mid] <= distance:
                lo = mid
            else:
                hi = mid

        span = lengths[hi] - lengths[lo]
        if span <= 1.0e-15:
            return parameters[lo]
        f = (distance - lengths[lo]) / span
        return parameters[lo] + f * (parameters[hi] - parameters[lo])

    @staticmethod
    def _LengthAtParameter(parameter, metric):
        parameters, lengths = metric
        t = min(1.0, max(0.0, float(parameter)))
        if t <= 0:
            return 0.0
        if t >= 1:
            return lengths[-1]

        x = t * (len(parameters) - 1)
        i = min(int(math.floor(x)), len(parameters) - 2)
        f = x - i
        return lengths[i] + f * (lengths[i + 1] - lengths[i])

    @staticmethod
    def _SurfacePositions(face, axis, spacing, divisions, values, alignment,
                          includeBoundary, distribution, samples, tolerance):
        explicit = Grid._ExplicitValues(values, includeBoundary)
        metric = Grid._SurfaceMetric(face, axis, samples)
        if metric is None:
            return None

        if explicit is not None:
            return explicit, metric, None

        if divisions is not None:
            fractions = Grid._DivisionValues(divisions, includeBoundary)
            if fractions is None:
                return None
            if distribution == "parameter":
                positions = fractions
            else:
                total = metric[1][-1]
                positions = [Grid._ParameterAtLength(f * total, metric)
                             for f in fractions]
            return Grid._Unique(positions), metric, None

        distances, datum = Grid._RegularDistances(
            metric[1][-1], spacing, alignment, includeBoundary, tolerance
        )
        if distances is None:
            return None
        positions = [Grid._ParameterAtLength(d, metric) for d in distances]
        return Grid._Unique(positions), metric, datum

    # -------------------------------------------------------------------------
    # Planar mode
    # -------------------------------------------------------------------------

    @staticmethod
    def _PlanarFrame(face, xDirection, samples, tolerance):
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        try:
            normal = Face.NormalAtParameters(
                face, u=0.5, v=0.5, mantissa=12, silent=True
            )
        except Exception:
            try:
                normal = Face.Normal(face, mantissa=12)
            except Exception:
                return None

        basis = Grid._Basis(normal, xDirection, tolerance)
        if basis is None:
            return None
        u_dir, v_dir = basis

        try:
            origin = Grid._Coordinates(Face.VertexByParameters(face, u=0.5, v=0.5))
        except Exception:
            origin = None

        points = []
        for edge in Topology.Edges(face, silent=True) or []:
            try:
                linear = Edge.IsLinear(edge, silent=True)
            except Exception:
                linear = False
            n = 1 if linear else max(8, min(64, int(samples) // 4))
            for i in range(n + 1):
                try:
                    point = Grid._Coordinates(Edge.VertexByParameter(edge, i / n))
                except Exception:
                    point = None
                if point is not None:
                    points.append(point)

        if not points:
            return None
        if origin is None:
            origin = points[0]

        us, vs = [], []
        for point in points:
            delta = Grid._Subtract(point, origin)
            us.append(Grid._Dot(delta, u_dir))
            vs.append(Grid._Dot(delta, v_dir))

        u_bounds = [min(us), max(us)]
        v_bounds = [min(vs), max(vs)]
        tol = Grid._Tolerance(tolerance)
        if u_bounds[1] - u_bounds[0] <= tol or v_bounds[1] - v_bounds[0] <= tol:
            return None
        return origin, u_dir, v_dir, u_bounds, v_bounds

    @staticmethod
    def _PlanarPositions(bounds, spacing, divisions, values, alignment,
                         includeBoundary, tolerance):
        start, end = bounds
        length = end - start

        explicit = Grid._ExplicitValues(values, includeBoundary)
        if explicit is not None:
            return [start + t * length for t in explicit], None

        if divisions is not None:
            fractions = Grid._DivisionValues(divisions, includeBoundary)
            if fractions is None:
                return None
            return [start + t * length for t in fractions], None

        distances, datum = Grid._RegularDistances(
            length, spacing, alignment, includeBoundary, tolerance
        )
        if distances is None:
            return None
        return [start + d for d in distances], start + datum

    # -------------------------------------------------------------------------
    # Surface isocurve construction
    # -------------------------------------------------------------------------

    @staticmethod
    def _NativeSurface(face):
        """Return (Geom_Surface, uMin, uMax, vMin, vMax), or None."""
        from topologicpy.Topology import Topology

        try:
            shape = Topology.OCCTShape(face)
            if shape is None:
                return None
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

            adaptor = BRepAdaptor_Surface(shape, True)
            bounds = [
                float(adaptor.FirstUParameter()),
                float(adaptor.LastUParameter()),
                float(adaptor.FirstVParameter()),
                float(adaptor.LastVParameter()),
            ]
            if not all(math.isfinite(x) for x in bounds):
                return None

            try:
                surface = BRep_Tool.Surface(shape)
            except Exception:
                surface = BRep_Tool.Surface_s(shape)

            return (surface, *bounds)
        except Exception:
            return None

    @staticmethod
    def _NativeIso(face, axis, parameter, nativeSurface):
        """Create one exact OCCT constant-U or constant-V Edge."""
        from topologicpy.Topology import Topology
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

        surface, u0, u1, v0, v1 = nativeSurface
        try:
            if axis == "u":
                u = u0 + float(parameter) * (u1 - u0)
                maker = BRepBuilderAPI_MakeEdge(surface.UIso(u), v0, v1)
            else:
                v = v0 + float(parameter) * (v1 - v0)
                maker = BRepBuilderAPI_MakeEdge(surface.VIso(v), u0, u1)

            if hasattr(maker, "IsDone") and not maker.IsDone():
                return None
            return Topology.ByOCCTShape(maker.Edge())
        except Exception:
            return None

    @staticmethod
    def _Edges(result):
        from topologicpy.Topology import Topology
        if result is None:
            return []
        if Topology.IsInstance(result, "Edge"):
            return [result]
        try:
            return Topology.Edges(result, silent=True) or []
        except Exception:
            return []

    @staticmethod
    def _Clip(edge, face, tolerance):
        """
        Clip an exact isocurve/line to the trimmed Face.

        The fallback returns the original Edge only if several interior samples
        confirm that the entire tested curve lies in the Face.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        if edge is None:
            return []

        try:
            clipped = Topology.Intersect(
                edge, face, tolerance=tolerance, silent=True
            )
            edges = Grid._Edges(clipped)
            if edges:
                return edges
        except Exception:
            pass

        # Useful for rectangular UV patches where Common occasionally returns
        # no 1D result although the isocurve is wholly on the Face.
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            try:
                vertex = Edge.VertexByParameter(edge, t)
                inside = Vertex.IsInternal(vertex, face, tolerance=tolerance)
            except TypeError:
                try:
                    inside = Vertex.IsInternal(vertex, face)
                except Exception:
                    return []
            except Exception:
                return []
            if inside is False:
                return []
        return [edge]

    @staticmethod
    def _ApproximateIso(face, axis, parameter, samples, tolerance):
        """Explicit polyline fallback; used only when approximate=True."""
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex

        samples = max(8, min(int(samples), 4096))
        result = []

        for i in range(samples):
            a = i / samples
            b = (i + 1) / samples
            m = (a + b) * 0.5
            try:
                if axis == "u":
                    va = Face.VertexByParameters(face, u=parameter, v=a)
                    vb = Face.VertexByParameters(face, u=parameter, v=b)
                    vm = Face.VertexByParameters(face, u=parameter, v=m)
                else:
                    va = Face.VertexByParameters(face, u=a, v=parameter)
                    vb = Face.VertexByParameters(face, u=b, v=parameter)
                    vm = Face.VertexByParameters(face, u=m, v=parameter)
            except Exception:
                continue

            try:
                inside = Vertex.IsInternal(vm, face, tolerance=tolerance)
            except TypeError:
                try:
                    inside = Vertex.IsInternal(vm, face)
                except Exception:
                    inside = True
            except Exception:
                inside = True

            if inside is False:
                continue

            try:
                edge = Edge.ByVertices(
                    [va, vb], tolerance=tolerance, silent=True
                )
            except Exception:
                edge = None
            if edge is not None:
                result.append(edge)

        return result

    @staticmethod
    def _Append(target, resultEdges, metadata):
        for segment, edge in enumerate(resultEdges):
            md = dict(metadata)
            md["grid_segment"] = segment
            edge = Grid._SetDictionary(edge, md)
            if edge is not None:
                target.append(edge)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @staticmethod
    def OnFace(
        face,
        spacing: float = 1.0,
        uSpacing=None,
        vSpacing=None,
        uDivisions=None,
        vDivisions=None,
        uValues=None,
        vValues=None,
        mode: str = "auto",
        distribution: str = "length",
        xDirection=None,
        uAlignment: str = "center",
        vAlignment: str = "center",
        includeBoundary: bool = False,
        uLabels=None,
        vLabels=None,
        approximate: bool = False,
        samples: int = 128,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a grid on a planar, curved, or NURBS Face.

        ``mode="auto"`` uses an intuitive world-oriented planar grid for planar
        Faces and an intrinsic UV surface grid for non-planar Faces.

        In surface mode, U-family lines are constant-U surface isocurves and
        V-family lines are constant-V isocurves. Under PythonOCC these are exact
        OCCT curves trimmed by the input Face, so a grid on a curved/NURBS Face
        contains genuinely curved Edges rather than projected chords.

        Each family can be specified independently. Precedence is:

        1. ``uValues`` / ``vValues``: explicit normalized parameters in [0, 1].
        2. ``uDivisions`` / ``vDivisions``: number of bays.
        3. ``uSpacing`` / ``vSpacing``: physical model-unit spacing; if omitted,
           ``spacing`` is used.

        For a curved surface, there is no unique globally constant separation
        between isocurves. Physical U spacing is therefore measured along the
        central V=0.5 reference curve, and physical V spacing along the central
        U=0.5 reference curve. ``distribution="length"`` uses this same metric
        for division placement; ``distribution="parameter"`` uses equal UV
        increments.

        ``mode="planar"`` can be forced for a planar Face and accepts
        ``xDirection`` to control its U direction. ``mode="surface"`` can be
        forced on any Face to use its native parameterization.

        Exact non-planar surface isocurves require PythonOCC. By default this
        method returns None rather than flattening them if exact native curves
        are unavailable. Set ``approximate=True`` explicitly to permit a
        sampled polyline fallback.

        Parameters
        ----------
        face : topologicpy.Face
            Input Face.
        spacing : float , optional
            Common physical spacing. Default is 1.
        uSpacing, vSpacing : float , optional
            Per-family physical spacing overrides.
        uDivisions, vDivisions : int , optional
            Number of bays in each family.
        uValues, vValues : float or list , optional
            Explicit normalized U/V positions in [0, 1].
        mode : str , optional
            "auto", "planar", or "surface". Default is "auto".
        distribution : str , optional
            "length" or "parameter" for surface division placement.
        xDirection : list , optional
            Preferred planar U direction. Ignored in surface mode.
        uAlignment, vAlignment : str , optional
            "center", "start", or "end" for spacing placement.
        includeBoundary : bool , optional
            Include U/V extrema as grid lines. Default is False.
        uLabels, vLabels : str or list , optional
            "letters", "numbers", a prefix string, or explicit labels.
        approximate : bool , optional
            Permit sampled polylines when native surface isocurves are
            unavailable. Default is False.
        samples : int , optional
            Samples used for physical surface metrics and fallback geometry.
            Does not control PythonOCC output curve geometry. Default is 128.
        mantissa : int , optional
            Decimal precision used in metadata. Default is 6.
        tolerance : float , optional
            Geometric tolerance. Default is 0.0001.
        silent : bool , optional
            Suppress errors/warnings. Default is False.

        Returns
        -------
        topologicpy.Cluster
            Cluster of clipped grid Edges, or None.

        Notes
        -----
        ``grid_axis="u"`` denotes a constant-U line (which runs in the surface
        V direction), and vice versa.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        tol = Grid._Tolerance(tolerance)

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Grid.OnFace - Error: face is not a valid Face. Returning None.")
            return None

        try:
            samples = max(16, min(int(samples), 4096))
            mantissa = max(0, int(mantissa))
            spacing = abs(float(spacing))
            uSpacing = spacing if uSpacing is None else abs(float(uSpacing))
            vSpacing = spacing if vSpacing is None else abs(float(vSpacing))
        except Exception:
            if not silent:
                print("Grid.OnFace - Error: Invalid numeric input. Returning None.")
            return None

        if uValues is None and uDivisions is None and uSpacing <= tol:
            if not silent:
                print("Grid.OnFace - Error: uSpacing is too small. Returning None.")
            return None
        if vValues is None and vDivisions is None and vSpacing <= tol:
            if not silent:
                print("Grid.OnFace - Error: vSpacing is too small. Returning None.")
            return None

        mode = str(mode).lower()
        if mode not in ("auto", "planar", "surface"):
            if not silent:
                print("Grid.OnFace - Error: mode must be auto, planar, or surface. Returning None.")
            return None

        distribution = str(distribution).lower()
        if distribution in ("distance", "physical"):
            distribution = "length"
        elif distribution in ("parametric", "uv"):
            distribution = "parameter"
        if distribution not in ("length", "parameter"):
            if not silent:
                print("Grid.OnFace - Error: distribution must be length or parameter. Returning None.")
            return None

        try:
            planar = Face.IsPlanar(face, silent=True)
        except Exception:
            planar = Topology.IsPlanar(face, tolerance=tol, silent=True)

        actualMode = "planar" if mode == "auto" and planar is True else (
            "surface" if mode == "auto" else mode
        )

        if actualMode == "planar" and planar is False:
            if not silent:
                print("Grid.OnFace - Error: mode='planar' requires a planar Face. Returning None.")
            return None

        resultEdges = []

        # ---------------------------------------------------------------------
        # Planar grid
        # ---------------------------------------------------------------------
        if actualMode == "planar":
            frame = Grid._PlanarFrame(face, xDirection, samples, tol)
            if frame is None:
                if not silent:
                    print("Grid.OnFace - Error: Could not derive planar frame. Returning None.")
                return None

            origin, uDir, vDir, uBounds, vBounds = frame
            uData = Grid._PlanarPositions(
                uBounds, uSpacing, uDivisions, uValues,
                uAlignment, includeBoundary, tol
            )
            vData = Grid._PlanarPositions(
                vBounds, vSpacing, vDivisions, vValues,
                vAlignment, includeBoundary, tol
            )
            if uData is None or vData is None:
                if not silent:
                    print("Grid.OnFace - Error: Could not derive planar positions. Returning None.")
                return None

            uPositions, uDatum = uData
            vPositions, vDatum = vData
            uLabelsResolved = Grid._Labels(len(uPositions), uLabels)
            vLabelsResolved = Grid._Labels(len(vPositions), vLabels)

            families = (
                ("u", uPositions, uBounds, vBounds, uDir, vDir, uDatum, uLabelsResolved),
                ("v", vPositions, vBounds, uBounds, vDir, uDir, vDatum, vLabelsResolved),
            )

            for axis, positions, primaryBounds, secondaryBounds, primaryDir, secondaryDir, datum, labels in families:
                pMin, pMax = primaryBounds
                sMin, sMax = secondaryBounds
                for index, position in enumerate(positions):
                    a = Grid._Add(
                        origin,
                        Grid._Add(
                            Grid._Scale(primaryDir, position),
                            Grid._Scale(secondaryDir, sMin),
                        ),
                    )
                    b = Grid._Add(
                        origin,
                        Grid._Add(
                            Grid._Scale(primaryDir, position),
                            Grid._Scale(secondaryDir, sMax),
                        ),
                    )
                    edge = Edge.ByVertices(
                        [Vertex.ByCoordinates(*a), Vertex.ByCoordinates(*b)],
                        tolerance=tol,
                        silent=True,
                    )
                    clipped = Grid._Clip(edge, face, tol)
                    parameter = (position - pMin) / (pMax - pMin)
                    Grid._Append(
                        resultEdges,
                        clipped,
                        {
                            "grid_source": "OnFace",
                            "grid_mode": "planar",
                            "grid_geometry": "line",
                            "grid_axis": axis,
                            "grid_index": index,
                            "grid_parameter": round(parameter, mantissa),
                            "grid_distance": round(position - pMin, mantissa),
                            "grid_label": labels[index],
                            "grid_is_boundary": abs(parameter) <= 1e-10 or abs(parameter - 1) <= 1e-10,
                            "grid_is_datum": datum is not None and abs(position - datum) <= tol,
                        },
                    )

        # ---------------------------------------------------------------------
        # Intrinsic surface grid
        # ---------------------------------------------------------------------
        else:
            uData = Grid._SurfacePositions(
                face, "u", uSpacing, uDivisions, uValues,
                uAlignment, includeBoundary, distribution, samples, tol
            )
            vData = Grid._SurfacePositions(
                face, "v", vSpacing, vDivisions, vValues,
                vAlignment, includeBoundary, distribution, samples, tol
            )
            if uData is None or vData is None:
                if not silent:
                    print("Grid.OnFace - Error: Could not derive surface positions. Returning None.")
                return None

            nativeSurface = Grid._NativeSurface(face)
            if nativeSurface is None and not approximate:
                if not silent:
                    print(
                        "Grid.OnFace - Error: Exact surface isocurves are unavailable "
                        "with the active backend. Set approximate=True for a polyline "
                        "fallback. Returning None."
                    )
                return None

            for axis, data, labelsSpec in (
                ("u", uData, uLabels),
                ("v", vData, vLabels),
            ):
                positions, metric, datum = data
                labels = Grid._Labels(len(positions), labelsSpec)

                for index, parameter in enumerate(positions):
                    if nativeSurface is not None:
                        raw = Grid._NativeIso(face, axis, parameter, nativeSurface)
                        clipped = Grid._Clip(raw, face, tol)
                        geometry = "isocurve"
                    else:
                        clipped = Grid._ApproximateIso(
                            face, axis, parameter, samples, tol
                        )
                        geometry = "polyline"

                    distance = Grid._LengthAtParameter(parameter, metric)
                    Grid._Append(
                        resultEdges,
                        clipped,
                        {
                            "grid_source": "OnFace",
                            "grid_mode": "surface",
                            "grid_geometry": geometry,
                            "grid_axis": axis,
                            "grid_index": index,
                            "grid_parameter": round(parameter, mantissa),
                            "grid_distance": round(distance, mantissa),
                            "grid_label": labels[index],
                            "grid_is_boundary": abs(parameter) <= 1e-10 or abs(parameter - 1) <= 1e-10,
                            "grid_is_datum": datum is not None and abs(distance - datum) <= tol,
                        },
                    )

        if not resultEdges:
            if not silent:
                print("Grid.OnFace - Warning: No grid Edges were created. Returning None.")
            return None

        return Cluster.ByTopologies(resultEdges, silent=True)
