# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations


class Grid:
    """Utility methods for creating architectural and design grids."""

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _Tolerance(tolerance=0.0001):
        try:
            value = abs(float(tolerance))
        except Exception:
            value = 0.0001
        return max(value, 1e-12)

    @staticmethod
    def _IsFace(face):
        from topologicpy.Topology import Topology
        return Topology.IsInstance(face, "Face")

    @staticmethod
    def _IsVertex(vertex):
        from topologicpy.Topology import Topology
        return Topology.IsInstance(vertex, "Vertex")

    @staticmethod
    def _Coordinates(vertex, mantissa=6):
        from topologicpy.Vertex import Vertex
        return [
            Vertex.X(vertex, mantissa=mantissa),
            Vertex.Y(vertex, mantissa=mantissa),
            Vertex.Z(vertex, mantissa=mantissa),
        ]

    @staticmethod
    def _Add(a, b):
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    @staticmethod
    def _Subtract(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    @staticmethod
    def _Scale(v, s):
        return [v[0] * s, v[1] * s, v[2] * s]

    @staticmethod
    def _Dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _Cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _Magnitude(v):
        return Grid._Dot(v, v) ** 0.5

    @staticmethod
    def _Normalize(v, tolerance=0.0001):
        mag = Grid._Magnitude(v)
        if mag <= Grid._Tolerance(tolerance):
            return None
        return [v[0] / mag, v[1] / mag, v[2] / mag]

    @staticmethod
    def _ProjectedDirection(direction, normal, tolerance=0.0001):
        """Project a direction into the plane perpendicular to normal."""
        d = Grid._Normalize(direction, tolerance=tolerance)
        n = Grid._Normalize(normal, tolerance=tolerance)
        if d is None or n is None:
            return None
        projected = Grid._Subtract(d, Grid._Scale(n, Grid._Dot(d, n)))
        return Grid._Normalize(projected, tolerance=tolerance)

    @staticmethod
    def _Basis(normal=None, xDirection=None, tolerance=0.0001):
        """Return a stable, architecture-friendly local u/v basis.

        The default u direction follows world X where possible. The default v
        direction prefers world Z on vertical/sloping faces and world Y on
        horizontal faces. This keeps wall grids horizontal/vertical and floor
        grids aligned to world X/Y without depending on face-normal orientation.
        """
        if normal is None:
            normal = [0, 0, 1]
        n = Grid._Normalize(normal, tolerance=tolerance)
        if n is None:
            return None, None

        u_candidates = []
        if isinstance(xDirection, (list, tuple)) and len(xDirection) == 3:
            u_candidates.append(list(xDirection))
        u_candidates.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        u_dir = None
        for candidate in u_candidates:
            u_dir = Grid._ProjectedDirection(candidate, n, tolerance=tolerance)
            if u_dir is not None:
                break
        if u_dir is None:
            return None, None

        # Prefer an intuitive "up" direction on walls and world Y on floors.
        for candidate in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            projected = Grid._ProjectedDirection(candidate, n, tolerance=tolerance)
            if projected is None:
                continue
            orthogonal = Grid._Subtract(projected, Grid._Scale(u_dir, Grid._Dot(projected, u_dir)))
            v_dir = Grid._Normalize(orthogonal, tolerance=tolerance)
            if v_dir is not None:
                return u_dir, v_dir

        # Numerical fallback.
        v_dir = Grid._Normalize(Grid._Cross(n, u_dir), tolerance=tolerance)
        if v_dir is None:
            return None, None
        return u_dir, v_dir

    @staticmethod
    def _OriginCoordinates(origin=None, mantissa=6):
        if Grid._IsVertex(origin):
            return Grid._Coordinates(origin, mantissa=mantissa)
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _Point(origin, u_dir, v_dir, u, v, mantissa=6):
        from topologicpy.Vertex import Vertex
        p = Grid._Add(origin, Grid._Add(Grid._Scale(u_dir, u), Grid._Scale(v_dir, v)))
        return Vertex.ByCoordinates(
            round(p[0], mantissa),
            round(p[1], mantissa),
            round(p[2], mantissa),
        )

    @staticmethod
    def _SetDictionary(topology, metadata):
        from topologicpy.Topology import Topology

        if topology is None or not isinstance(metadata, dict):
            return topology

        metadata = {str(key): value for key, value in metadata.items() if value is not None}
        if len(metadata) < 1:
            return topology

        try:
            result = Topology.SetDictionary(topology, metadata, silent=True)
            return result if result is not None else topology
        except Exception:
            return topology

    @staticmethod
    def _Value(topology, key, default=None):
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        try:
            dictionary = Topology.Dictionary(topology, silent=True)
            return Dictionary.ValueAtKey(dictionary, key, default)
        except Exception:
            return default

    @staticmethod
    def _PlacementExtents(width, length, placement="center"):
        placement = str(placement).lower().replace("_", "").replace("-", "")

        if placement == "center":
            return [-0.5 * width, 0.5 * width], [-0.5 * length, 0.5 * length]
        if placement in ["lowerleft", "bottomleft"]:
            return [0.0, width], [0.0, length]
        if placement in ["upperleft", "topleft"]:
            return [0.0, width], [-length, 0.0]
        if placement in ["lowerright", "bottomright"]:
            return [-width, 0.0], [0.0, length]
        if placement in ["upperright", "topright"]:
            return [-width, 0.0], [-length, 0.0]
        return None, None

    @staticmethod
    def _UniqueSorted(values, tolerance=0.0001):
        tol = Grid._Tolerance(tolerance)
        out = []
        for value in sorted(float(v) for v in values):
            if not out or abs(value - out[-1]) > tol:
                out.append(value)
        return out

    @staticmethod
    def _RegularCoordinates(bounds, spacing, alignment="center", includeBoundary=True, tolerance=0.0001):
        """Return regularly spaced coordinates within bounds."""
        tol = Grid._Tolerance(tolerance)
        try:
            spacing = abs(float(spacing))
        except Exception:
            return None, None
        if spacing <= tol:
            return None, None

        a = float(min(bounds))
        b = float(max(bounds))
        if b - a <= tol:
            return None, None

        alignment = str(alignment).lower()
        if alignment == "center":
            datum = 0.5 * (a + b)
        elif alignment in ["start", "left", "bottom", "lower"]:
            datum = a
            alignment = "start"
        elif alignment in ["end", "right", "top", "upper"]:
            datum = b
            alignment = "end"
        else:
            return None, None

        values = [datum]
        k = 1
        while datum - k * spacing >= a - tol:
            values.append(datum - k * spacing)
            k += 1
        k = 1
        while datum + k * spacing <= b + tol:
            values.append(datum + k * spacing)
            k += 1

        if includeBoundary:
            values.extend([a, b])

        values = [min(b, max(a, v)) for v in values if a - tol <= v <= b + tol]
        return Grid._UniqueSorted(values, tolerance=tol), datum

    @staticmethod
    def _DivisionCoordinates(bounds, divisions, tolerance=0.0001):
        tol = Grid._Tolerance(tolerance)
        try:
            divisions = int(divisions)
        except Exception:
            return None
        if divisions < 1:
            return None

        a = float(min(bounds))
        b = float(max(bounds))
        if b - a <= tol:
            return None
        step = (b - a) / divisions
        return [a + i * step for i in range(divisions + 1)]

    @staticmethod
    def _AlphaLabel(index):
        """Return Excel-style A..Z, AA.. labels for zero-based index."""
        try:
            index = int(index)
        except Exception:
            return None
        if index < 0:
            return None
        value = index + 1
        label = ""
        while value > 0:
            value, rem = divmod(value - 1, 26)
            label = chr(65 + rem) + label
        return label

    @staticmethod
    def _Labels(count, specification=None):
        if specification is None or specification is False:
            return [None] * count
        if specification is True:
            specification = "numbers"
        if isinstance(specification, str):
            mode = specification.lower()
            if mode in ["letters", "alpha", "alphabetic"]:
                return [Grid._AlphaLabel(i) for i in range(count)]
            if mode in ["numbers", "numeric"]:
                return [str(i + 1) for i in range(count)]
            return [f"{specification}{i + 1}" for i in range(count)]
        if isinstance(specification, (list, tuple)):
            labels = [str(v) for v in specification[:count]]
            labels += [None] * (count - len(labels))
            return labels
        return [None] * count

    @staticmethod
    def _AppendEdgeResult(result, edges, metadata):
        from topologicpy.Topology import Topology

        if result is None:
            return

        candidates = []
        if Topology.IsInstance(result, "Edge"):
            candidates = [result]
        else:
            try:
                candidates = Topology.Edges(result) or []
            except Exception:
                candidates = []

        for segment, edge in enumerate(candidates):
            if not Topology.IsInstance(edge, "Edge"):
                continue
            md = dict(metadata)
            md["grid_segment"] = segment
            edge = Grid._SetDictionary(edge, md)
            if edge is not None:
                edges.append(edge)

    @staticmethod
    def _CreateOrthogonal(origin,
                          u_dir,
                          v_dir,
                          uBounds,
                          vBounds,
                          uCoordinates,
                          vCoordinates,
                          source,
                          role="axis",
                          uSpacing=None,
                          vSpacing=None,
                          uAlignment=None,
                          vAlignment=None,
                          uDatum=None,
                          vDatum=None,
                          uLabels=None,
                          vLabels=None,
                          clipFace=None,
                          extension=0.0,
                          commonMetadata=None,
                          uItemMetadata=None,
                          vItemMetadata=None,
                          mantissa=6,
                          tolerance=0.0001):
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        tol = Grid._Tolerance(tolerance)
        try:
            extension = max(0.0, float(extension))
        except Exception:
            extension = 0.0

        uCoordinates = Grid._UniqueSorted(uCoordinates or [], tolerance=tol)
        vCoordinates = Grid._UniqueSorted(vCoordinates or [], tolerance=tol)
        if len(uCoordinates) < 1 and len(vCoordinates) < 1:
            return None

        if uLabels is None:
            uLabels = [None] * len(uCoordinates)
        if vLabels is None:
            vLabels = [None] * len(vCoordinates)
        if uItemMetadata is None:
            uItemMetadata = [{} for _ in uCoordinates]
        if vItemMetadata is None:
            vItemMetadata = [{} for _ in vCoordinates]
        commonMetadata = commonMetadata or {}

        uMin, uMax = min(uBounds), max(uBounds)
        vMin, vMax = min(vBounds), max(vBounds)
        edges = []

        for i, u in enumerate(uCoordinates):
            start = Grid._Point(origin, u_dir, v_dir, u, vMin - extension, mantissa=mantissa)
            end = Grid._Point(origin, u_dir, v_dir, u, vMax + extension, mantissa=mantissa)
            edge = Edge.ByVertices([start, end], tolerance=tol)
            if edge is None:
                continue
            if Grid._IsFace(clipFace):
                try:
                    edge = Topology.Intersect(edge, clipFace, tolerance=tol, silent=True)
                except Exception:
                    edge = None

            metadata = {
                "grid_type": "orthogonal",
                "grid_source": source,
                "grid_role": role,
                "grid_axis": "u",
                "grid_index": i,
                "grid_coordinate": round(float(u), mantissa),
                "grid_label": uLabels[i] if i < len(uLabels) else None,
                "grid_spacing": uSpacing,
                "grid_alignment": uAlignment,
                "grid_is_boundary": abs(u - uMin) <= tol or abs(u - uMax) <= tol,
                "grid_is_datum": uDatum is not None and abs(u - uDatum) <= tol,
            }
            metadata.update(commonMetadata)
            if i < len(uItemMetadata):
                metadata.update(uItemMetadata[i] or {})
            Grid._AppendEdgeResult(edge, edges, metadata)

        for i, v in enumerate(vCoordinates):
            start = Grid._Point(origin, u_dir, v_dir, uMin - extension, v, mantissa=mantissa)
            end = Grid._Point(origin, u_dir, v_dir, uMax + extension, v, mantissa=mantissa)
            edge = Edge.ByVertices([start, end], tolerance=tol)
            if edge is None:
                continue
            if Grid._IsFace(clipFace):
                try:
                    edge = Topology.Intersect(edge, clipFace, tolerance=tol, silent=True)
                except Exception:
                    edge = None

            metadata = {
                "grid_type": "orthogonal",
                "grid_source": source,
                "grid_role": role,
                "grid_axis": "v",
                "grid_index": i,
                "grid_coordinate": round(float(v), mantissa),
                "grid_label": vLabels[i] if i < len(vLabels) else None,
                "grid_spacing": vSpacing,
                "grid_alignment": vAlignment,
                "grid_is_boundary": abs(v - vMin) <= tol or abs(v - vMax) <= tol,
                "grid_is_datum": vDatum is not None and abs(v - vDatum) <= tol,
            }
            metadata.update(commonMetadata)
            if i < len(vItemMetadata):
                metadata.update(vItemMetadata[i] or {})
            Grid._AppendEdgeResult(edge, edges, metadata)

        return Cluster.ByTopologies(edges) if len(edges) > 0 else None

    @staticmethod
    def _FaceFrame(face, xDirection=None, mantissa=6, tolerance=0.0001):
        """Return a centred local frame and extents for a planar face."""
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Grid._IsFace(face):
            return None
        try:
            if Topology.IsPlanar(face, mantissa=mantissa, tolerance=tolerance, silent=True) is not True:
                return None
        except Exception:
            return None

        try:
            normal = Face.Normal(face)
        except Exception:
            return None
        u_dir, v_dir = Grid._Basis(normal=normal, xDirection=xDirection, tolerance=tolerance)
        if u_dir is None or v_dir is None:
            return None

        try:
            vertices = Topology.Vertices(face) or []
        except Exception:
            vertices = []
        if len(vertices) < 3:
            return None

        ref = Grid._Coordinates(vertices[0], mantissa=mantissa)
        u_values = []
        v_values = []
        for vertex in vertices:
            p = Grid._Coordinates(vertex, mantissa=mantissa)
            delta = Grid._Subtract(p, ref)
            u_values.append(Grid._Dot(delta, u_dir))
            v_values.append(Grid._Dot(delta, v_dir))

        uMin, uMax = min(u_values), max(u_values)
        vMin, vMax = min(v_values), max(v_values)
        uCenter = 0.5 * (uMin + uMax)
        vCenter = 0.5 * (vMin + vMax)
        origin = Grid._Add(ref, Grid._Add(Grid._Scale(u_dir, uCenter), Grid._Scale(v_dir, vCenter)))
        uBounds = [uMin - uCenter, uMax - uCenter]
        vBounds = [vMin - vCenter, vMax - vCenter]
        return origin, u_dir, v_dir, uBounds, vBounds

    @staticmethod
    def _TileJointCoordinates(bounds,
                              tileSize,
                              grout=0.0,
                              centerMode="best",
                              tolerance=0.0001):
        """Return centred tile-joint coordinates and the selected centring mode."""
        tol = Grid._Tolerance(tolerance)
        try:
            tileSize = float(tileSize)
            grout = max(0.0, float(grout))
        except Exception:
            return None, None, None
        if tileSize <= tol:
            return None, None, None

        pitch = tileSize + grout
        if pitch <= tol:
            return None, None, None

        a = float(min(bounds))
        b = float(max(bounds))
        c = 0.5 * (a + b)

        def candidate(mode):
            if mode == "joint":
                base = c
            else:
                base = c + 0.5 * pitch

            values = []
            kMin = int((a - base) // pitch) - 2
            kMax = int((b - base) // pitch) + 2
            for k in range(kMin, kMax + 1):
                value = base + k * pitch
                if a + tol < value < b - tol:
                    values.append(value)
            values = Grid._UniqueSorted(values, tolerance=tol)

            if len(values) < 1:
                score = b - a
            else:
                leftCut = max(0.0, values[0] - a - 0.5 * grout)
                rightCut = max(0.0, b - values[-1] - 0.5 * grout)
                score = min(leftCut, rightCut)
            return values, score

        mode = str(centerMode).lower().replace("_", "").replace("-", "")
        if mode in ["joint", "grout", "line"]:
            values, _ = candidate("joint")
            return values, "joint", pitch
        if mode in ["tile", "panel"]:
            values, _ = candidate("tile")
            return values, "tile", pitch
        if mode != "best":
            return None, None, None

        jointValues, jointScore = candidate("joint")
        tileValues, tileScore = candidate("tile")
        if tileScore >= jointScore:
            return tileValues, "tile", pitch
        return jointValues, "joint", pitch

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    @staticmethod
    def Square(origin=None,
               size: float = 10.0,
               spacing: float = 1.0,
               placement: str = "center",
               direction: list = [0, 0, 1],
               xDirection=None,
               alignment: str = "center",
               includeBoundary: bool = True,
               uLabels=None,
               vLabels=None,
               mantissa: int = 6,
               tolerance: float = 0.0001,
               silent: bool = False):
        """
        Creates a square orthogonal grid using physical model-unit spacing.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the grid. Default is the world origin.
        size : float , optional
            The width and length of the square grid. Default is 10.
        spacing : float , optional
            The spacing between grid lines in both directions. Default is 1.
        placement : str , optional
            The relationship of the grid extents to the origin. Supported values are
            "center", "lowerleft", "upperleft", "lowerright", and "upperright".
            Default is "center".
        direction : list , optional
            The normal direction of the grid plane. Default is [0, 0, 1].
        xDirection : list , optional
            A preferred direction for the local u axis. It is projected onto the
            grid plane. Default is None, which uses a stable world-axis direction.
        alignment : str , optional
            How the regular spacing is anchored within the extents. Supported values
            are "center", "start", and "end". Default is "center".
        includeBoundary : bool , optional
            If True, grid lines are added at the outer square boundary even when the
            boundary does not coincide with the regular spacing. Default is True.
        uLabels : str or list , optional
            Labels for u grid lines. Use "letters", "numbers", a prefix string, or
            an explicit list. Default is None.
        vLabels : str or list , optional
            Labels for v grid lines. Default is None.
        mantissa : int , optional
            The number of decimal places used for generated coordinates. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If True, error messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            A cluster of grid edges with semantic dictionaries.
        """
        return Grid.Rectangular(
            origin=origin,
            width=size,
            length=size,
            spacing=spacing,
            placement=placement,
            direction=direction,
            xDirection=xDirection,
            uAlignment=alignment,
            vAlignment=alignment,
            includeBoundary=includeBoundary,
            uLabels=uLabels,
            vLabels=vLabels,
            mantissa=mantissa,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Rectangular(origin=None,
                    width: float = 10.0,
                    length: float = 10.0,
                    spacing: float = 1.0,
                    uSpacing=None,
                    vSpacing=None,
                    placement: str = "center",
                    direction: list = [0, 0, 1],
                    xDirection=None,
                    uAlignment: str = "center",
                    vAlignment: str = "center",
                    includeBoundary: bool = True,
                    uLabels=None,
                    vLabels=None,
                    mantissa: int = 6,
                    tolerance: float = 0.0001,
                    silent: bool = False):
        """
        Creates a rectangular orthogonal grid using physical model-unit spacing.

        ``spacing`` provides a convenient common spacing. ``uSpacing`` and
        ``vSpacing`` can override it independently.
        """
        tol = Grid._Tolerance(tolerance)
        try:
            width = abs(float(width))
            length = abs(float(length))
            spacing = abs(float(spacing))
            uSpacing = spacing if uSpacing is None else abs(float(uSpacing))
            vSpacing = spacing if vSpacing is None else abs(float(vSpacing))
        except Exception:
            if not silent:
                print("Grid.Rectangular - Error: The input dimensions or spacing are not valid. Returning None.")
            return None
        if width <= tol or length <= tol or uSpacing <= tol or vSpacing <= tol:
            if not silent:
                print("Grid.Rectangular - Error: The input dimensions and spacing must be greater than the tolerance. Returning None.")
            return None

        uBounds, vBounds = Grid._PlacementExtents(width, length, placement=placement)
        if uBounds is None or vBounds is None:
            if not silent:
                print("Grid.Rectangular - Error: The input placement is not supported. Returning None.")
            return None

        u_dir, v_dir = Grid._Basis(normal=direction, xDirection=xDirection, tolerance=tol)
        if u_dir is None or v_dir is None:
            if not silent:
                print("Grid.Rectangular - Error: Could not derive a valid grid plane. Returning None.")
            return None

        origin_coords = Grid._OriginCoordinates(origin=origin, mantissa=mantissa)
        uCoords, uDatum = Grid._RegularCoordinates(uBounds, uSpacing, alignment=uAlignment,
                                                   includeBoundary=includeBoundary, tolerance=tol)
        vCoords, vDatum = Grid._RegularCoordinates(vBounds, vSpacing, alignment=vAlignment,
                                                   includeBoundary=includeBoundary, tolerance=tol)
        if uCoords is None or vCoords is None:
            if not silent:
                print("Grid.Rectangular - Error: The input alignment is not supported. Returning None.")
            return None

        return Grid._CreateOrthogonal(
            origin=origin_coords,
            u_dir=u_dir,
            v_dir=v_dir,
            uBounds=uBounds,
            vBounds=vBounds,
            uCoordinates=uCoords,
            vCoordinates=vCoords,
            source="Rectangular",
            role="axis",
            uSpacing=uSpacing,
            vSpacing=vSpacing,
            uAlignment=str(uAlignment).lower(),
            vAlignment=str(vAlignment).lower(),
            uDatum=uDatum,
            vDatum=vDatum,
            uLabels=Grid._Labels(len(uCoords), uLabels),
            vLabels=Grid._Labels(len(vCoords), vLabels),
            mantissa=mantissa,
            tolerance=tol,
        )

    @staticmethod
    def ByDivisions(origin=None,
                    width: float = 10.0,
                    length: float = 10.0,
                    uDivisions: int = 10,
                    vDivisions: int = 10,
                    placement: str = "center",
                    direction: list = [0, 0, 1],
                    xDirection=None,
                    uLabels=None,
                    vLabels=None,
                    mantissa: int = 6,
                    tolerance: float = 0.0001,
                    silent: bool = False):
        """
        Creates a rectangular grid by an exact number of bays/divisions.

        ``uDivisions`` and ``vDivisions`` refer to the number of spaces between
        grid lines, not the number of grid lines. For example, 4 divisions create
        5 grid lines including the two boundaries.
        """
        tol = Grid._Tolerance(tolerance)
        try:
            width = abs(float(width))
            length = abs(float(length))
            uDivisions = int(uDivisions)
            vDivisions = int(vDivisions)
        except Exception:
            if not silent:
                print("Grid.ByDivisions - Error: The input dimensions or divisions are not valid. Returning None.")
            return None
        if width <= tol or length <= tol or uDivisions < 1 or vDivisions < 1:
            if not silent:
                print("Grid.ByDivisions - Error: Dimensions must be positive and divisions must be at least 1. Returning None.")
            return None

        uBounds, vBounds = Grid._PlacementExtents(width, length, placement=placement)
        if uBounds is None or vBounds is None:
            if not silent:
                print("Grid.ByDivisions - Error: The input placement is not supported. Returning None.")
            return None

        u_dir, v_dir = Grid._Basis(normal=direction, xDirection=xDirection, tolerance=tol)
        if u_dir is None or v_dir is None:
            if not silent:
                print("Grid.ByDivisions - Error: Could not derive a valid grid plane. Returning None.")
            return None

        uCoords = Grid._DivisionCoordinates(uBounds, uDivisions, tolerance=tol)
        vCoords = Grid._DivisionCoordinates(vBounds, vDivisions, tolerance=tol)
        uSpacing = width / uDivisions
        vSpacing = length / vDivisions

        return Grid._CreateOrthogonal(
            origin=Grid._OriginCoordinates(origin=origin, mantissa=mantissa),
            u_dir=u_dir,
            v_dir=v_dir,
            uBounds=uBounds,
            vBounds=vBounds,
            uCoordinates=uCoords,
            vCoordinates=vCoords,
            source="ByDivisions",
            role="axis",
            uSpacing=uSpacing,
            vSpacing=vSpacing,
            uAlignment="fit",
            vAlignment="fit",
            uDatum=0.5 * (uBounds[0] + uBounds[1]),
            vDatum=0.5 * (vBounds[0] + vBounds[1]),
            uLabels=Grid._Labels(len(uCoords), uLabels),
            vLabels=Grid._Labels(len(vCoords), vLabels),
            commonMetadata={
                "grid_u_divisions": uDivisions,
                "grid_v_divisions": vDivisions,
            },
            mantissa=mantissa,
            tolerance=tol,
        )

    @staticmethod
    def Structural(origin=None,
                   uBays=None,
                   vBays=None,
                   placement: str = "center",
                   direction: list = [0, 0, 1],
                   xDirection=None,
                   extension: float = 0.0,
                   uLabels="letters",
                   vLabels="numbers",
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Creates an architectural/structural setting-out grid from bay widths.

        Parameters
        ----------
        uBays : list
            Consecutive bay widths along the local u axis, e.g. [6, 6, 7.5, 6].
        vBays : list
            Consecutive bay widths along the local v axis.
        extension : float , optional
            Distance by which each grid axis extends beyond the outermost axes.
            Default is 0.
        uLabels : str or list , optional
            U-axis labels. Default is "letters".
        vLabels : str or list , optional
            V-axis labels. Default is "numbers".

        Returns
        -------
        topologic_core.Cluster
            A cluster of structural grid axes with semantic dictionaries including
            the adjacent bay widths.
        """
        tol = Grid._Tolerance(tolerance)
        if uBays is None:
            uBays = [6.0, 6.0, 6.0]
        if vBays is None:
            vBays = [6.0, 6.0, 6.0]
        try:
            uBays = [abs(float(v)) for v in uBays]
            vBays = [abs(float(v)) for v in vBays]
        except Exception:
            if not silent:
                print("Grid.Structural - Error: The input uBays or vBays is not valid. Returning None.")
            return None
        if len(uBays) < 1 or len(vBays) < 1 or min(uBays) <= tol or min(vBays) <= tol:
            if not silent:
                print("Grid.Structural - Error: Bay widths must be positive. Returning None.")
            return None

        width = sum(uBays)
        length = sum(vBays)
        uBounds, vBounds = Grid._PlacementExtents(width, length, placement=placement)
        if uBounds is None or vBounds is None:
            if not silent:
                print("Grid.Structural - Error: The input placement is not supported. Returning None.")
            return None

        uCoords = [uBounds[0]]
        for bay in uBays:
            uCoords.append(uCoords[-1] + bay)
        vCoords = [vBounds[0]]
        for bay in vBays:
            vCoords.append(vCoords[-1] + bay)

        u_dir, v_dir = Grid._Basis(normal=direction, xDirection=xDirection, tolerance=tol)
        if u_dir is None or v_dir is None:
            if not silent:
                print("Grid.Structural - Error: Could not derive a valid grid plane. Returning None.")
            return None

        uItemMetadata = []
        for i in range(len(uCoords)):
            uItemMetadata.append({
                "grid_bay_before": uBays[i - 1] if i > 0 else None,
                "grid_bay_after": uBays[i] if i < len(uBays) else None,
            })
        vItemMetadata = []
        for i in range(len(vCoords)):
            vItemMetadata.append({
                "grid_bay_before": vBays[i - 1] if i > 0 else None,
                "grid_bay_after": vBays[i] if i < len(vBays) else None,
            })

        return Grid._CreateOrthogonal(
            origin=Grid._OriginCoordinates(origin=origin, mantissa=mantissa),
            u_dir=u_dir,
            v_dir=v_dir,
            uBounds=uBounds,
            vBounds=vBounds,
            uCoordinates=uCoords,
            vCoordinates=vCoords,
            source="Structural",
            role="structural_axis",
            uAlignment="bays",
            vAlignment="bays",
            uDatum=0.5 * (uBounds[0] + uBounds[1]),
            vDatum=0.5 * (vBounds[0] + vBounds[1]),
            uLabels=Grid._Labels(len(uCoords), uLabels),
            vLabels=Grid._Labels(len(vCoords), vLabels),
            extension=extension,
            uItemMetadata=uItemMetadata,
            vItemMetadata=vItemMetadata,
            commonMetadata={
                "grid_width": width,
                "grid_length": length,
            },
            mantissa=mantissa,
            tolerance=tol,
        )

    @staticmethod
    def OnFace(face,
               spacing: float = 1.0,
               uSpacing=None,
               vSpacing=None,
               xDirection=None,
               uAlignment: str = "center",
               vAlignment: str = "center",
               includeBoundary: bool = False,
               uLabels=None,
               vLabels=None,
               mantissa: int = 6,
               tolerance: float = 0.0001,
               silent: bool = False):
        """
        Creates an orthogonal grid on and clipped to a planar face.

        The grid is defined in physical model units. It does not depend on the
        face's UV parameterization. By default, the local u axis is the world X
        direction projected onto the face plane, with a stable fallback when
        necessary. Supply ``xDirection`` to control the grid orientation.

        ``spacing`` is a convenient common spacing. ``uSpacing`` and ``vSpacing``
        can override it independently.
        """
        tol = Grid._Tolerance(tolerance)
        if not Grid._IsFace(face):
            if not silent:
                print("Grid.OnFace - Error: The input face is not a valid Face. Returning None.")
            return None

        try:
            spacing = abs(float(spacing))
            uSpacing = spacing if uSpacing is None else abs(float(uSpacing))
            vSpacing = spacing if vSpacing is None else abs(float(vSpacing))
        except Exception:
            if not silent:
                print("Grid.OnFace - Error: The input spacing is not valid. Returning None.")
            return None
        if uSpacing <= tol or vSpacing <= tol:
            if not silent:
                print("Grid.OnFace - Error: Spacing must be greater than the tolerance. Returning None.")
            return None

        frame = Grid._FaceFrame(face, xDirection=xDirection, mantissa=mantissa, tolerance=tol)
        if frame is None:
            if not silent:
                print("Grid.OnFace - Error: Could not derive a valid planar frame from the face. Returning None.")
            return None
        origin, u_dir, v_dir, uBounds, vBounds = frame

        uCoords, uDatum = Grid._RegularCoordinates(uBounds, uSpacing, alignment=uAlignment,
                                                   includeBoundary=includeBoundary, tolerance=tol)
        vCoords, vDatum = Grid._RegularCoordinates(vBounds, vSpacing, alignment=vAlignment,
                                                   includeBoundary=includeBoundary, tolerance=tol)
        if uCoords is None or vCoords is None:
            if not silent:
                print("Grid.OnFace - Error: The input alignment is not supported. Returning None.")
            return None

        return Grid._CreateOrthogonal(
            origin=origin,
            u_dir=u_dir,
            v_dir=v_dir,
            uBounds=uBounds,
            vBounds=vBounds,
            uCoordinates=uCoords,
            vCoordinates=vCoords,
            source="OnFace",
            role="axis",
            uSpacing=uSpacing,
            vSpacing=vSpacing,
            uAlignment=str(uAlignment).lower(),
            vAlignment=str(vAlignment).lower(),
            uDatum=uDatum,
            vDatum=vDatum,
            uLabels=Grid._Labels(len(uCoords), uLabels),
            vLabels=Grid._Labels(len(vCoords), vLabels),
            clipFace=face,
            commonMetadata={"grid_clipped": True},
            mantissa=mantissa,
            tolerance=tol,
        )

    @staticmethod
    def TileLayout(face,
                   tileWidth: float = 0.3,
                   tileHeight=None,
                   groutWidth: float = 0.0,
                   groutHeight=None,
                   xDirection=None,
                   uCenterMode: str = "best",
                   vCenterMode: str = "best",
                   mantissa: int = 6,
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Creates a symmetric tile-joint layout on a planar face.

        The layout starts from the centre of the face and works outwards. For each
        axis, ``centerMode`` can be:

        - ``"tile"``: centre a tile on the face centreline.
        - ``"joint"``: centre a tile joint/grout line on the face centreline.
        - ``"best"``: test both symmetric alternatives and choose the one that
          maximises the smallest cut tile at the two edges.

        This captures the common setting-out rule used for wall and floor tiling:
        preserve symmetry while avoiding unnecessarily small edge cuts.

        Returns
        -------
        topologic_core.Cluster
            A cluster of clipped tile-joint edges. Each edge dictionary records the
            axis, index, coordinate, pitch, tile size, grout size, and selected
            centring mode.
        """
        tol = Grid._Tolerance(tolerance)
        if not Grid._IsFace(face):
            if not silent:
                print("Grid.TileLayout - Error: The input face is not a valid Face. Returning None.")
            return None

        if tileHeight is None:
            tileHeight = tileWidth
        if groutHeight is None:
            groutHeight = groutWidth

        try:
            tileWidth = float(tileWidth)
            tileHeight = float(tileHeight)
            groutWidth = max(0.0, float(groutWidth))
            groutHeight = max(0.0, float(groutHeight))
        except Exception:
            if not silent:
                print("Grid.TileLayout - Error: Tile or grout dimensions are not valid. Returning None.")
            return None
        if tileWidth <= tol or tileHeight <= tol:
            if not silent:
                print("Grid.TileLayout - Error: Tile dimensions must be greater than the tolerance. Returning None.")
            return None

        frame = Grid._FaceFrame(face, xDirection=xDirection, mantissa=mantissa, tolerance=tol)
        if frame is None:
            if not silent:
                print("Grid.TileLayout - Error: Could not derive a valid planar frame from the face. Returning None.")
            return None
        origin, u_dir, v_dir, uBounds, vBounds = frame

        uCoords, selectedU, uPitch = Grid._TileJointCoordinates(
            uBounds, tileWidth, grout=groutWidth, centerMode=uCenterMode, tolerance=tol
        )
        vCoords, selectedV, vPitch = Grid._TileJointCoordinates(
            vBounds, tileHeight, grout=groutHeight, centerMode=vCenterMode, tolerance=tol
        )
        if uCoords is None or vCoords is None:
            if not silent:
                print("Grid.TileLayout - Error: The input centre mode is not supported. Returning None.")
            return None

        return Grid._CreateOrthogonal(
            origin=origin,
            u_dir=u_dir,
            v_dir=v_dir,
            uBounds=uBounds,
            vBounds=vBounds,
            uCoordinates=uCoords,
            vCoordinates=vCoords,
            source="TileLayout",
            role="tile_joint",
            uSpacing=uPitch,
            vSpacing=vPitch,
            uAlignment="center",
            vAlignment="center",
            uDatum=0.0 if selectedU == "joint" else None,
            vDatum=0.0 if selectedV == "joint" else None,
            clipFace=face,
            commonMetadata={
                "grid_clipped": True,
                "grid_tile_width": tileWidth,
                "grid_tile_height": tileHeight,
                "grid_grout_width": groutWidth,
                "grid_grout_height": groutHeight,
                "grid_u_center_mode": selectedU,
                "grid_v_center_mode": selectedV,
            },
            mantissa=mantissa,
            tolerance=tol,
        )

    @staticmethod
    def Vertices(grid,
                 mantissa: int = 6,
                 tolerance: float = 0.0001,
                 silent: bool = False):
        """
        Returns the semantic intersection vertices of an orthogonal grid.

        Only intersections between u and v grid families are returned. The grid
        edge semantics are transferred into explicit vertex keys such as
        ``grid_u_index``, ``grid_v_index``, ``grid_u_coordinate``,
        ``grid_v_coordinate``, ``grid_u_label``, and ``grid_v_label``.

        Parameters
        ----------
        grid : topologic_core.Topology
            A grid returned by this class.
        mantissa : int , optional
            The number of decimal places used to deduplicate coincident vertices.
            Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If True, error messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            A cluster of semantic grid-intersection vertices.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        tol = Grid._Tolerance(tolerance)
        try:
            edges = Topology.Edges(grid) or []
        except Exception:
            edges = []
        if len(edges) < 2:
            if not silent:
                print("Grid.Vertices - Error: The input grid does not contain enough edges. Returning None.")
            return None

        uEdges = [e for e in edges if Grid._Value(e, "grid_axis", None) == "u"]
        vEdges = [e for e in edges if Grid._Value(e, "grid_axis", None) == "v"]
        if len(uEdges) < 1 or len(vEdges) < 1:
            if not silent:
                print("Grid.Vertices - Error: The input does not contain both u and v grid families. Returning None.")
            return None

        vertices = []
        seen = set()
        for uEdge in uEdges:
            for vEdge in vEdges:
                try:
                    result = Topology.Intersect(uEdge, vEdge, tolerance=tol, silent=True)
                except Exception:
                    result = None
                if result is None:
                    continue

                candidates = []
                if Topology.IsInstance(result, "Vertex"):
                    candidates = [result]
                else:
                    try:
                        candidates = Topology.Vertices(result) or []
                    except Exception:
                        candidates = []

                for vertex in candidates:
                    if not Topology.IsInstance(vertex, "Vertex"):
                        continue
                    xyz = Grid._Coordinates(vertex, mantissa=mantissa)
                    key = tuple(round(float(c), mantissa) for c in xyz)
                    if key in seen:
                        continue
                    seen.add(key)

                    metadata = {
                        "grid_type": "orthogonal",
                        "grid_source": Grid._Value(uEdge, "grid_source", Grid._Value(vEdge, "grid_source", None)),
                        "grid_role": "intersection",
                        "grid_u_index": Grid._Value(uEdge, "grid_index", None),
                        "grid_v_index": Grid._Value(vEdge, "grid_index", None),
                        "grid_u_coordinate": Grid._Value(uEdge, "grid_coordinate", None),
                        "grid_v_coordinate": Grid._Value(vEdge, "grid_coordinate", None),
                        "grid_u_label": Grid._Value(uEdge, "grid_label", None),
                        "grid_v_label": Grid._Value(vEdge, "grid_label", None),
                    }
                    vertex = Grid._SetDictionary(vertex, metadata)
                    if vertex is not None:
                        vertices.append(vertex)

        if len(vertices) < 1:
            return None
        return Cluster.ByTopologies(vertices)
