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

from topologicpy.Core import Core
import math

class CellComplex():
    @staticmethod
    def Box(origin= None,
            width: float = 1.0, length: float = 1.0, height: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a box with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. Default is None which results in the box being placed at (0, 0, 0).
        width : float , optional
            The width of the box. Default is 1.
        length : float , optional
            The length of the box. Default is 1.
        height : float , optional
            The height of the box.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int, optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the box. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the box. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created box.

        """
        return CellComplex.Prism(origin=origin,
                                 width=width, length=length, height=height,
                                 uSides=uSides, vSides=vSides, wSides=wSides,
                                 direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def ByCells(cells: list, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a CellComplex by assembling the input Cells.

        The active backend is asked to construct the CellComplex directly from
        the input Cells. The input Cells are converted to Faces only as a
        compatibility fallback for backends that do not expose a native
        ``CellComplex.ByCells`` constructor.

        Parameters
        ----------
        cells : list
            The input list of Cells.
        transferDictionaries : bool, optional
            If True, dictionaries from the source Cells are transferred to the
            corresponding Cells in the result. Default is False.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex, or None on failure.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not isinstance(cells, list):
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter is not a valid list. Returning None.")
            return None

        cells = [cell for cell in cells if Topology.IsInstance(cell, "Cell")]
        if len(cells) < 1:
            if not silent:
                print("CellComplex.ByCells - Error: The input cells parameter does not contain any valid Cells. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.ByCells - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if tolerance <= 0.0:
            if not silent:
                print("CellComplex.ByCells - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        cell_complex = None
        method = getattr(Core.CellComplex, "ByCells", None)
        if callable(method):
            attempts = (
                lambda: method(cells, tolerance=tolerance),
                lambda: method(cells, tolerance),
                lambda: method(cells, tolerance, False),
            )
            for attempt in attempts:
                try:
                    cell_complex = attempt()
                except (TypeError, AttributeError):
                    continue
                except Exception:
                    cell_complex = None
                    break
                if Topology.IsInstance(cell_complex, "CellComplex"):
                    break

        # Compatibility fallback: preserve the exact source Faces and ask the
        # backend to assemble those. This does not polygonise curved geometry.
        if not Topology.IsInstance(cell_complex, "CellComplex"):
            faces = []
            for cell in cells:
                cell_faces = Topology.Faces(cell)
                if isinstance(cell_faces, list):
                    faces.extend(cell_faces)
            cell_complex = CellComplex._ByFaces(faces, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(cell_complex, "CellComplex"):
            if not silent:
                print("CellComplex.ByCells - Error: Could not create a CellComplex from the input Cells. Returning None.")
            return None

        if transferDictionaries:
            try:
                source_cluster = Cluster.ByTopologies(cells)
                result_cells = CellComplex.Cells(cell_complex, silent=True)
                if isinstance(result_cells, list):
                    for result_cell in result_cells:
                        selector = Topology.InternalVertex(result_cell, tolerance=tolerance)
                        if not Topology.IsInstance(selector, "Vertex"):
                            continue
                        enclosing_cells = Vertex.EnclosingCells(
                            selector,
                            source_cluster,
                            tolerance=tolerance,
                        )
                        if not isinstance(enclosing_cells, list) or len(enclosing_cells) == 0:
                            continue
                        dictionaries = [Topology.Dictionary(cell) for cell in enclosing_cells]
                        dictionary = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                        if dictionary is not None:
                            Topology.SetDictionary(result_cell, dictionary, silent=True)
            except Exception:
                if not silent:
                    print("CellComplex.ByCells - Warning: The CellComplex was created, but one or more dictionaries could not be transferred.")

        return cell_complex
    
    @staticmethod
    def ByCellsCluster(cluster, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cellcomplex by merging the cells within the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of cells.
        transferDictionaries : bool , optional
            If set to True, any dictionaries in the faces are transferred to the faces of the created CellComplex.
            Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("CellComplex.ByCellsCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cells = Topology.Cells(cluster)
        return CellComplex.ByCells(cells, transferDictionaries=transferDictionaries, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByDisjointedFaces(faces: list,
                           minOffset: float = 0,
                           maxOffset: float = 1.0,
                           minCells: int = 2,
                           maxCells: int = 10,
                           maxAttempts: int = 100,
                           patience: int = 5,
                           transferDictionaries: bool = False,
                           exclusive: bool = True,
                           tolerance: float = 0.0001,
                           silent: bool = False):
        """
        Creates a CellComplex from a list of disjoint Faces by progressively
        offsetting the Faces until a valid volumetric partition can be built.

        Parameters
        ----------
        faces : list of topologic_core.Face
            The input Faces.
        minOffset : float, optional
            The minimum initial Face offset to try. Default is 0.
        maxOffset : float, optional
            The maximum Face offset to try. Default is 1.0.
        minCells : int, optional
            The minimum number of Cells to create. Default is 2.
        maxCells : int, optional
            The maximum number of Cells to retain. Default is 10.
        maxAttempts : int, optional
            Maximum number of offset attempts. Default is 100.
        patience : int, optional
            Number of consecutive attempts with the same non-zero Cell count
            after which the search stops. Set to 0 to disable this early stop.
            Default is 5.
        transferDictionaries : bool, optional
            If True, Face dictionaries are inherited by result Faces.
            Default is False.
        exclusive : bool, optional
            Used only when transferDictionaries is True. If True, only one
            source Face contributes its dictionary to a target Face.
            Default is True.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The best CellComplex found, or None on failure.
        """
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(faces) < 3:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input list must contain at least three valid Faces. Returning None.")
            return None

        try:
            minOffset = float(minOffset)
            maxOffset = float(maxOffset)
            minCells = int(minCells)
            maxCells = int(maxCells)
            maxAttempts = int(maxAttempts)
            patience = int(patience)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        if tolerance <= 0.0 or minOffset < 0.0 or minOffset > maxOffset:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: Invalid offset or tolerance range. Returning None.")
            return None
        if minCells < 2 or minCells > maxCells:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: minCells must be at least 2 and no greater than maxCells. Returning None.")
            return None
        if maxAttempts < 1 or patience < 0 or patience > maxAttempts:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: Invalid maxAttempts or patience parameter. Returning None.")
            return None

        def _trim(cells, count):
            ranked = []
            for cell in cells:
                volume = Cell.Volume(cell, mantissa=12)
                if volume is not None:
                    ranked.append((float(volume), cell))
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in ranked[:count]]

        if maxAttempts == 1:
            offsets = [minOffset]
        else:
            step = (maxOffset - minOffset) / float(maxAttempts - 1)
            offsets = [minOffset + step * i for i in range(maxAttempts)]

        candidates = []
        recent_counts = []

        for offset in offsets:
            expanded_faces = []
            for face in faces:
                try:
                    expanded = Face.ByOffset(face, offset=-offset, tolerance=tolerance, silent=True)
                except Exception:
                    expanded = None
                if Topology.IsInstance(expanded, "Face"):
                    expanded_faces.append(expanded)

            if len(expanded_faces) < 3:
                recent_counts.append(0)
            else:
                cc = CellComplex.ByFaces(expanded_faces, tolerance=tolerance, silent=True)
                if Topology.IsInstance(cc, "CellComplex"):
                    cells = CellComplex.Cells(cc, silent=True)
                    n_cells = len(cells) if isinstance(cells, list) else 0
                    recent_counts.append(n_cells)

                    if minCells <= n_cells <= maxCells:
                        candidates.append((n_cells, cc))
                    elif n_cells > maxCells:
                        trimmed = _trim(cells, maxCells)
                        if len(trimmed) >= minCells:
                            trimmed_cc = CellComplex.ByCells(trimmed, tolerance=tolerance, silent=True)
                            if Topology.IsInstance(trimmed_cc, "CellComplex"):
                                candidates.append((len(trimmed), trimmed_cc))
                else:
                    recent_counts.append(0)

            if patience > 0 and len(recent_counts) >= patience:
                window = recent_counts[-patience:]
                if len(set(window)) == 1 and window[0] > 0:
                    if not silent:
                        print("CellComplex.ByDisjointedFaces - Warning: Ran out of patience.")
                    break

        if len(candidates) == 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: Could not create a CellComplex. Consider revising the input parameters. Returning None.")
            return None

        candidates.sort(key=lambda item: item[0])
        cell_complex = candidates[-1][1]

        if transferDictionaries:
            try:
                result_faces = CellComplex.Faces(cell_complex, silent=True)
                Topology.Inherit(
                    targets=result_faces,
                    sources=faces,
                    exclusive=exclusive,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                if not silent:
                    print("CellComplex.ByDisjointedFaces - Warning: The CellComplex was created, but dictionaries could not be transferred.")

        return cell_complex

    
    @staticmethod
    def _ByFaces(faces: list, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a CellComplex directly from the input Faces using the active
        backend. No polygonisation or external geometry library is used.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(faces) < 1:
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter does not contain any valid Faces. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.ByFaces - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None
        if tolerance <= 0.0:
            return None

        method = getattr(Core.CellComplex, "ByFaces", None)
        cell_complex = None
        if callable(method):
            attempts = (
                lambda: method(faces, tolerance=tolerance, copyAttributes=False),
                lambda: method(faces, tolerance, False),
                lambda: method(faces, tolerance),
            )
            for attempt in attempts:
                try:
                    cell_complex = attempt()
                except (TypeError, AttributeError):
                    continue
                except Exception:
                    cell_complex = None
                    break
                if Topology.IsInstance(cell_complex, "CellComplex"):
                    return cell_complex

        # Legacy compatibility fallback. Merge the original Faces directly;
        # this preserves their actual Edge/Surface geometry.
        topology = faces[0]
        for i, face in enumerate(faces[1:], start=1):
            try:
                merged = Core.InstanceCall(topology, "Merge", face, False, tolerance)
            except Exception:
                merged = None
            if merged is not None:
                topology = merged
            elif not silent:
                print(f"CellComplex.ByFaces - Warning: Failed to merge Face #{i}. Skipping it.")

        if Topology.IsInstance(topology, "CellComplex"):
            return topology
        if Topology.IsInstance(topology, "Cluster"):
            try:
                complexes = Cluster.CellComplexes(topology)
            except Exception:
                complexes = []
            if isinstance(complexes, list) and len(complexes) > 0:
                return complexes[0]

        if not silent:
            print("CellComplex.ByFaces - Error: The input Faces do not form a CellComplex. Returning None.")
        return None

    @staticmethod
    def ByFacesTopologic(faces, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a CellComplex from the input faces after removing coplanar overlaps
        using only TopologicPy / Topologic boolean operations.

        The method keeps larger coplanar faces first and trims later faces by
        subtracting already accepted coplanar regions. This avoids dissolving
        coplanar subdivisions into a single merged face.

        Parameters
        ----------
        faces : list
            The input list of topologic_core.Face objects.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex or None
            The created CellComplex.
        """

        import math

        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFacesTopologic - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        faces = [f for f in faces if Topology.IsInstance(f, "Face")]

        if len(faces) == 0:
            if not silent:
                print("CellComplex.ByFacesTopologic - Error: The input faces list does not contain any valid faces. Returning None.")
            return None

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _length(v):
            return math.sqrt(_dot(v, v))

        def _normalize(v):
            length = _length(v)
            if length <= tolerance:
                return None
            return [v[0]/length, v[1]/length, v[2]/length]

        def _coords(vertex):
            return [
                Vertex.X(vertex, mantissa=12),
                Vertex.Y(vertex, mantissa=12),
                Vertex.Z(vertex, mantissa=12),
            ]

        def _face_vertices(face):
            try:
                return Topology.Vertices(face)
            except Exception:
                return []

        def _face_normal(face):
            try:
                n = Face.Normal(face)
                if isinstance(n, tuple):
                    n = list(n)
                return _normalize(n)
            except Exception:
                return None

        def _canonical_normal(n):
            """
            Makes opposite normals equivalent for grouping coplanar faces.
            """
            if not n:
                return None

            for c in n:
                if abs(c) > tolerance:
                    if c < 0:
                        return [-n[0], -n[1], -n[2]]
                    return n

            return n

        def _plane_key(face):
            vertices = _face_vertices(face)

            if len(vertices) < 3:
                return None

            n = _face_normal(face)

            if not n:
                return None

            n = _canonical_normal(n)
            p = _coords(vertices[0])
            d = _dot(n, p)

            q = max(tolerance, 1e-9)

            return (
                round(n[0] / q),
                round(n[1] / q),
                round(n[2] / q),
                round(d / q),
            )

        def _aabb(face):
            vertices = _face_vertices(face)

            if len(vertices) == 0:
                return None

            xs = []
            ys = []
            zs = []

            for v in vertices:
                xs.append(Vertex.X(v, mantissa=12))
                ys.append(Vertex.Y(v, mantissa=12))
                zs.append(Vertex.Z(v, mantissa=12))

            return [
                min(xs), min(ys), min(zs),
                max(xs), max(ys), max(zs),
            ]

        def _aabb_overlap(a, b):
            if a is None or b is None:
                return True

            return not (
                a[3] < b[0] - tolerance or b[3] < a[0] - tolerance or
                a[4] < b[1] - tolerance or b[4] < a[1] - tolerance or
                a[5] < b[2] - tolerance or b[5] < a[2] - tolerance
            )

        def _face_area(face):
            try:
                return abs(Face.Area(face))
            except Exception:
                return 0.0

        def _extract_faces(topology):
            if topology is None:
                return []

            if isinstance(topology, list):
                result = []
                for item in topology:
                    result.extend(_extract_faces(item))
                return result

            if Topology.IsInstance(topology, "Face"):
                return [topology]

            try:
                extracted = Topology.Faces(topology)
                return [f for f in extracted if Topology.IsInstance(f, "Face")]
            except Exception:
                return []

        def _boolean(topology_a, topology_b, operation):
            """
            Tries a few common TopologicPy boolean call signatures.
            This keeps the method tolerant of minor API differences between versions.
            """
            try:
                return Topology.Boolean(topology_a, topology_b, operation=operation, tolerance=tolerance, silent=True)
            except TypeError:
                pass
            except Exception:
                return None

            try:
                return Topology.Boolean(topology_a, topology_b, operation=operation, tolerance=tolerance)
            except TypeError:
                pass
            except Exception:
                return None

            try:
                return Topology.Boolean(topology_a, topology_b, operation=operation)
            except TypeError:
                pass
            except Exception:
                return None

            try:
                return Topology.Boolean(topology_a, topology_b, operation)
            except Exception:
                return None

        def _intersects(face_a, face_b):
            if not _aabb_overlap(_aabb(face_a), _aabb(face_b)):
                return False

            intersection = _boolean(face_a, face_b, "Intersect")
            intersection_faces = _extract_faces(intersection)

            if len(intersection_faces) == 0:
                return False

            return sum(_face_area(f) for f in intersection_faces) > tolerance * tolerance

        def _difference(face_a, face_b):
            difference = _boolean(face_a, face_b, "Difference")
            difference_faces = _extract_faces(difference)

            if len(difference_faces) == 0:
                return []

            return [f for f in difference_faces if _face_area(f) > tolerance * tolerance]

        def _remove_overlaps_in_group(group_faces):
            if len(group_faces) <= 1:
                return group_faces

            group_faces = sorted(group_faces, key=_face_area, reverse=True)

            accepted = []
            accepted_aabbs = []

            for face in group_faces:
                pieces = [face]

                for cutter, cutter_aabb in zip(accepted, accepted_aabbs):
                    new_pieces = []

                    for piece in pieces:
                        piece_aabb = _aabb(piece)

                        if not _aabb_overlap(piece_aabb, cutter_aabb):
                            new_pieces.append(piece)
                            continue

                        if not _intersects(piece, cutter):
                            new_pieces.append(piece)
                            continue

                        difference_faces = _difference(piece, cutter)

                        if len(difference_faces) > 0:
                            new_pieces.extend(difference_faces)

                    pieces = new_pieces

                    if len(pieces) == 0:
                        break

                for piece in pieces:
                    if _face_area(piece) > tolerance * tolerance:
                        accepted.append(piece)
                        accepted_aabbs.append(_aabb(piece))

            return accepted

        # -------------------------------------------------------------------------
        # Group faces by approximately identical planes.
        # -------------------------------------------------------------------------

        groups = {}
        passthrough_faces = []

        for face in faces:
            key = _plane_key(face)

            if key is None:
                passthrough_faces.append(face)
            else:
                groups.setdefault(key, []).append(face)

        cleaned_faces = list(passthrough_faces)

        # -------------------------------------------------------------------------
        # Remove coplanar overlaps using Topologic boolean Difference.
        # -------------------------------------------------------------------------

        for group_faces in groups.values():
            cleaned_faces.extend(_remove_overlaps_in_group(group_faces))

        if len(cleaned_faces) == 0:
            if not silent:
                print("CellComplex.ByFacesTopologic - Error: No valid faces remained after overlap removal. Returning None.")
            return None

        return CellComplex._ByFaces(cleaned_faces, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def ByFaces(faces, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a CellComplex directly from the input Faces while preserving
        their native geometry, including curved Edges and Surfaces.

        This is the default Face-based constructor. It does not project Faces to
        2D and does not require Shapely. For difficult coplanar-overlap cases,
        ``ByFacesTopologic`` and ``ByFacesShapely`` remain available explicitly.

        Parameters
        ----------
        faces : list
            The input Faces.
        transferDictionaries : bool, optional
            If True, source Face dictionaries are inherited by result Faces.
            Default is False.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex, or None on failure.
        """
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        source_faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(source_faces) < 1:
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter does not contain any valid Faces. Returning None.")
            return None

        cell_complex = CellComplex._ByFaces(source_faces, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(cell_complex, "CellComplex"):
            if not silent:
                print(
                    "CellComplex.ByFaces - Error: Could not create a CellComplex directly from the input Faces. "
                    "For coplanar-overlap repair, try CellComplex.ByFacesTopologic or CellComplex.ByFacesShapely. Returning None."
                )
            return None

        if transferDictionaries:
            try:
                result_faces = CellComplex.Faces(cell_complex, silent=True)
                Topology.Inherit(
                    targets=result_faces,
                    sources=source_faces,
                    exclusive=False,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                if not silent:
                    print("CellComplex.ByFaces - Warning: The CellComplex was created, but dictionaries could not be transferred.")

        return cell_complex

    @staticmethod
    def ByFacesShapely(faces, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a CellComplex from the input Faces after using Shapely to remove
        coplanar face overlaps.

        This method is intended as a faster pre-processing pathway for cases where
        native CellComplex.ByFaces is unsuitable because the input contains overlapping coplanar
        faces. Non-coplanar faces are passed through unchanged. Curved planar
        boundaries are reconstructed from projected polygon coordinates and may
        therefore become straight-edged approximations.

        Parameters
        ----------
        faces : list
            The input list of topologic_core.Face objects.
        transferDictionaries : bool , optional
            If set to True, any dictionaries in the faces are transferred to the faces of the created CellComplex.
            Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex or None
            The created CellComplex.
        """

        import math

        try:
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.ops import unary_union
            try:
                from shapely.validation import make_valid
            except Exception:
                make_valid = None
        except Exception:
            if not silent:
                print("CellComplex.ByFacesShapely - Error: Shapely is not installed. Please install it using: pip install shapely")
            return None

        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFacesShapely - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        faces = [f for f in faces if Topology.IsInstance(f, "Face")]

        if len(faces) == 0:
            if not silent:
                print("CellComplex.ByFacesShapely - Error: The input faces list does not contain any valid faces. Returning None.")
            return None

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]

        def _sub(a, b):
            return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

        def _add(a, b):
            return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

        def _mul(a, s):
            return [a[0]*s, a[1]*s, a[2]*s]

        def _length(v):
            return math.sqrt(_dot(v, v))

        def _normalize(v):
            l = _length(v)
            if l <= tolerance:
                return None
            return [v[0]/l, v[1]/l, v[2]/l]

        def _coords(vertex):
            return [
                Vertex.X(vertex, mantissa=12),
                Vertex.Y(vertex, mantissa=12),
                Vertex.Z(vertex, mantissa=12),
            ]

        def _face_vertices(face):
            try:
                return Topology.Vertices(face)
            except Exception:
                return []

        def _boundary_vertices(face):
            try:
                eb = Face.ExternalBoundary(face)
                return Topology.Vertices(eb)
            except Exception:
                return _face_vertices(face)

        def _face_normal(face):
            try:
                n = Face.Normal(face)
                if isinstance(n, tuple):
                    n = list(n)
                n = _normalize(n)
                if n:
                    return n
            except Exception:
                pass

            vertices = _boundary_vertices(face)
            if len(vertices) < 3:
                return None

            pts = [_coords(v) for v in vertices]

            p0 = pts[0]
            for i in range(1, len(pts)-1):
                a = _sub(pts[i], p0)
                b = _sub(pts[i+1], p0)
                n = _normalize(_cross(a, b))
                if n:
                    return n

            return None

        def _canonical_normal(n):
            # Make opposite normals group together.
            # Pick the orientation where the first significant component is positive.
            for c in n:
                if abs(c) > tolerance:
                    if c < 0:
                        return [-n[0], -n[1], -n[2]]
                    return n
            return n

        def _plane_key(face):
            vertices = _boundary_vertices(face)
            if len(vertices) < 3:
                return None

            n = _face_normal(face)
            if not n:
                return None

            n = _canonical_normal(n)
            p = _coords(vertices[0])
            d = _dot(n, p)

            q = max(tolerance, 1e-9)

            return (
                round(n[0] / q),
                round(n[1] / q),
                round(n[2] / q),
                round(d / q),
            )

        def _plane_basis(face):
            vertices = _boundary_vertices(face)
            if len(vertices) < 3:
                return None

            n = _face_normal(face)
            if not n:
                return None

            n = _canonical_normal(n)
            origin = _coords(vertices[0])

            # Choose a stable reference vector.
            if abs(n[0]) < 0.9:
                ref = [1.0, 0.0, 0.0]
            else:
                ref = [0.0, 1.0, 0.0]

            u = _normalize(_cross(ref, n))
            if not u:
                return None

            v = _normalize(_cross(n, u))
            if not v:
                return None

            return origin, u, v, n

        def _project_point(p, origin, u, v):
            w = _sub(p, origin)
            return (_dot(w, u), _dot(w, v))

        def _unproject_point(p, origin, u, v):
            return _add(origin, _add(_mul(u, p[0]), _mul(v, p[1])))

        def _ring_to_polygon_coords(vertices, origin, u, v):
            coords = []
            last = None

            for vertex in vertices:
                p = _coords(vertex)
                xy = _project_point(p, origin, u, v)

                if last is None:
                    coords.append(xy)
                    last = xy
                else:
                    if math.dist(last, xy) > tolerance:
                        coords.append(xy)
                        last = xy

            if len(coords) > 1 and math.dist(coords[0], coords[-1]) <= tolerance:
                coords = coords[:-1]

            if len(coords) < 3:
                return None

            return coords

        def _face_to_polygon(face, origin, u, v):
            exterior_vertices = _boundary_vertices(face)
            exterior = _ring_to_polygon_coords(exterior_vertices, origin, u, v)

            if not exterior:
                return None

            holes = []

            try:
                internal_boundaries = Face.InternalBoundaries(face)
            except Exception:
                internal_boundaries = []

            if internal_boundaries:
                for ib in internal_boundaries:
                    try:
                        ib_vertices = Topology.Vertices(ib)
                        hole = _ring_to_polygon_coords(ib_vertices, origin, u, v)
                        if hole and len(hole) >= 3:
                            holes.append(hole)
                    except Exception:
                        continue

            try:
                polygon = Polygon(exterior, holes)
            except Exception:
                return None

            if polygon.is_empty:
                return None

            if not polygon.is_valid:
                if make_valid:
                    polygon = make_valid(polygon)
                else:
                    polygon = polygon.buffer(0)

            if polygon.is_empty:
                return None

            return polygon

        def _polygon_to_faces(polygon, origin, u, v):
            result = []

            if polygon.is_empty:
                return result

            if isinstance(polygon, MultiPolygon):
                for geom in polygon.geoms:
                    result.extend(_polygon_to_faces(geom, origin, u, v))
                return result

            if polygon.geom_type != "Polygon":
                return result

            if polygon.area <= tolerance * tolerance:
                return result

            exterior_coords = list(polygon.exterior.coords)
            if len(exterior_coords) < 4:
                return result

            exterior_vertices = []
            for xy in exterior_coords[:-1]:
                p = _unproject_point(xy, origin, u, v)
                exterior_vertices.append(Vertex.ByCoordinates(p[0], p[1], p[2]))

            if len(exterior_vertices) < 3:
                return result

            try:
                external_wire = Wire.ByVertices(exterior_vertices, close=True, tolerance=tolerance, silent=True)
            except TypeError:
                external_wire = Wire.ByVertices(exterior_vertices, close=True, tolerance=tolerance)

            if not external_wire:
                return result

            internal_wires = []

            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                if len(interior_coords) < 4:
                    continue

                interior_vertices = []
                for xy in interior_coords[:-1]:
                    p = _unproject_point(xy, origin, u, v)
                    interior_vertices.append(Vertex.ByCoordinates(p[0], p[1], p[2]))

                if len(interior_vertices) < 3:
                    continue

                try:
                    iw = Wire.ByVertices(interior_vertices, close=True, tolerance=tolerance, silent=True)
                except TypeError:
                    iw = Wire.ByVertices(interior_vertices, close=True, tolerance=tolerance)

                if iw:
                    internal_wires.append(iw)

            face = None

            if len(internal_wires) > 0:
                try:
                    face = Face.ByWires(external_wire, internal_wires, tolerance=tolerance, silent=True)
                except Exception:
                    face = None

            if not face:
                try:
                    face = Face.ByWire(external_wire, tolerance=tolerance, silent=True)
                except TypeError:
                    face = Face.ByWire(external_wire, tolerance=tolerance)

            if face:
                result.append(face)

            return result

        def _clean_polygon(polygon):
            if polygon is None:
                return None

            if polygon.is_empty:
                return None

            if not polygon.is_valid:
                if make_valid:
                    polygon = make_valid(polygon)
                else:
                    polygon = polygon.buffer(0)

            if polygon.is_empty:
                return None

            if polygon.geom_type == "GeometryCollection":
                polygons = [g for g in polygon.geoms if g.geom_type in ["Polygon", "MultiPolygon"] and not g.is_empty]
                if len(polygons) == 0:
                    return None
                polygon = unary_union(polygons)

            return polygon

        # -------------------------------------------------------------------------
        # 1. Group faces by quantised plane
        # -------------------------------------------------------------------------

        groups = {}
        passthrough_faces = []

        for face in faces:
            key = _plane_key(face)
            if key is None:
                passthrough_faces.append(face)
            else:
                groups.setdefault(key, []).append(face)

        cleaned_faces = list(passthrough_faces)

        # -------------------------------------------------------------------------
        # 2. Resolve coplanar overlaps group-by-group
        # -------------------------------------------------------------------------

        for _, group_faces in groups.items():
            if len(group_faces) == 1:
                cleaned_faces.append(group_faces[0])
                continue

            basis = _plane_basis(group_faces[0])
            if not basis:
                cleaned_faces.extend(group_faces)
                continue

            origin, u, v, _ = basis

            items = []

            for face in group_faces:
                polygon = _face_to_polygon(face, origin, u, v)
                polygon = _clean_polygon(polygon)

                if polygon is None:
                    cleaned_faces.append(face)
                    continue

                items.append((face, polygon))

            if len(items) == 0:
                continue

            # Larger polygons first: this tends to preserve major surfaces and trim
            # smaller/duplicate overlapping fragments.
            items.sort(key=lambda item: item[1].area, reverse=True)

            accepted_polygons = []

            for original_face, polygon in items:
                polygon = _clean_polygon(polygon)

                if polygon is None:
                    continue

                if len(accepted_polygons) > 0:
                    occupied = unary_union(accepted_polygons)
                    polygon = polygon.difference(occupied)
                    polygon = _clean_polygon(polygon)

                if polygon is None:
                    continue

                new_faces = _polygon_to_faces(polygon, origin, u, v)

                if len(new_faces) == 0:
                    continue

                cleaned_faces.extend(new_faces)

                # Store the polygon actually accepted, not necessarily the original.
                accepted_polygons.append(polygon)

        if len(cleaned_faces) == 0:
            if not silent:
                print("CellComplex.ByFacesShapely - Error: No valid faces remained after Shapely processing. Returning None.")
            return None

        cc = CellComplex._ByFaces(cleaned_faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(cc, "cellcomplex"):
            if not silent:
                print("CellComplex.ByFacesShapely - Error: Could not create the CellComplex. Returning None.")
            return None
        
        if transferDictionaries:
            cc_faces = Topology.Faces(cc)
            source_cluster = Cluster.ByTopologies(faces)

            for cc_face in cc_faces:
                internal_vertex = Topology.InternalVertex(cc_face, tolerance=tolerance)
                enclosing_faces = Vertex.EnclosingFaces(internal_vertex,
                                                        source_cluster,
                                                        exclusive=False,
                                                        tolerance=tolerance)
                if isinstance(enclosing_faces, list) and len(enclosing_faces) > 0:
                    dictionaries = [Topology.Dictionary(face) for face in enclosing_faces]
                    merged_dictionary = Dictionary.ByMergedDictionaries(dictionaries, silent=True)
                    Topology.SetDictionary(cc_face, merged_dictionary)
        return cc

    @staticmethod
    def ByFacesCluster(cluster, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cellcomplex by merging the faces within the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of faces.
        transferDictionaries : bool , optional
            If set to True, any dictionaries in the faces are transferred to the faces of the created CellComplex.
            Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("CellComplex.ByFacesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        faces = Topology.Faces(cluster)
        return CellComplex.ByFaces(faces, transferDictionaries=transferDictionaries, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByWires(
        wires: list,
        polyhedron: bool = True,
        triangulate: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a CellComplex by lofting through the input Wires.

        Each consecutive pair of Wires defines one Cell. Intermediate Wires
        therefore become the shared non-manifold Faces between adjacent Cells.

        If polyhedron is True, the existing faceted construction is used.
        If polyhedron is False, the section curves are preserved and each Cell
        interval is constructed natively by the PythonOCC backend.

        Parameters
        ----------
        wires : list
            The ordered list of section Wires. At least two valid Wires are
            required. Corresponding Wires must contain the same number of Edges.
        polyhedron : bool , optional
            If True, constructs a faceted CellComplex. If False, constructs a
            curve-preserving CellComplex using the PythonOCC backend.
            Default is True.
        triangulate : bool , optional
            If polyhedron is True, specifies whether generated Faces are
            triangulated. Ignored when polyhedron is False. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            if not silent:
                print("CellComplex.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
            return None

        wires = [wire for wire in wires if Topology.IsInstance(wire, "Wire")]

        if len(wires) < 2:
            if not silent:
                print("CellComplex.ByWires - Error: The input wires parameter contains fewer than two valid Wires. Returning None.")
            return None

        # All sections must have the same edge count.
        edge_lists = [Topology.Edges(wire) for wire in wires]
        edge_count = len(edge_lists[0])

        if edge_count < 1:
            if not silent:
                print("CellComplex.ByWires - Error: The input Wires do not contain any valid Edges. Returning None.")
            return None

        if any(len(edges) != edge_count for edges in edge_lists[1:]):
            if not silent:
                print("CellComplex.ByWires - Error: The input Wires contain different numbers of Edges. Returning None.")
            return None

        # ------------------------------------------------------------------
        # Curve-preserving native CellComplex.
        # ------------------------------------------------------------------
        if not polyhedron:
            try:
                if Topology._IsTopologicCoreBackend():
                    if not silent:
                        print("CellComplex.ByWires - Error: polyhedron=False requires the PythonOCC backend. Returning None.")
                    return None
            except Exception:
                return None

            method = getattr(Core.CellComplex, "ByWires", None)
            if not callable(method):
                if not silent:
                    print("CellComplex.ByWires - Error: Native backend constructor is unavailable. Returning None.")
                return None

            try:
                cell_complex = method(wires, tolerance=tolerance)
            except Exception:
                cell_complex = None

            if not Topology.IsInstance(cell_complex, "CellComplex"):
                if not silent:
                    print("CellComplex.ByWires - Error: Could not create a curve-preserving CellComplex. Returning None.")
                return None

            return cell_complex

        # ------------------------------------------------------------------
        # Existing faceted construction.
        # ------------------------------------------------------------------
        faces = []

        first_face = Face.ByWire(wires[0], tolerance=tolerance, silent=silent)
        last_face = Face.ByWire(wires[-1], tolerance=tolerance, silent=silent)

        if not Topology.IsInstance(first_face, "Face") or not Topology.IsInstance(last_face, "Face"):
            if not silent:
                print("CellComplex.ByWires - Error: Could not create the end Faces. Returning None.")
            return None

        end_faces = [first_face, last_face]

        if triangulate:
            for face in end_faces:
                if len(Topology.Vertices(face)) > 3:
                    triangles = Face.Triangulate(face, tolerance=tolerance)
                    if isinstance(triangles, list):
                        faces.extend(triangles)
                else:
                    faces.append(face)
        else:
            faces.extend(end_faces)

        def _bridge(vertex_a, vertex_b):
            try:
                return Edge.ByStartVertexEndVertex(
                    vertex_a,
                    vertex_b,
                    tolerance=tolerance,
                    silent=True
                )
            except Exception:
                return None

        for i in range(len(wires) - 1):
            wire1 = wires[i]
            wire2 = wires[i + 1]

            # Every intermediate section is an internal CellComplex boundary.
            if i < len(wires) - 2:
                section_face = Face.ByWire(wire2, tolerance=tolerance, silent=silent)
                if Topology.IsInstance(section_face, "Face"):
                    if triangulate and len(Topology.Vertices(section_face)) > 3:
                        triangles = Face.Triangulate(section_face, tolerance=tolerance)
                        if isinstance(triangles, list):
                            faces.extend(triangles)
                    else:
                        faces.append(section_face)

            w1_edges = edge_lists[i]
            w2_edges = edge_lists[i + 1]

            for j in range(edge_count):
                edge1 = w1_edges[j]
                edge2 = w2_edges[j]

                bridge1 = _bridge(Edge.StartVertex(edge1), Edge.StartVertex(edge2))
                bridge2 = _bridge(Edge.EndVertex(edge1), Edge.EndVertex(edge2))

                face = None

                if bridge1 is not None and bridge2 is not None:
                    wire = Wire.ByEdges(
                        [edge1, bridge2, edge2, bridge1],
                        tolerance=tolerance,
                        silent=True
                    )
                    if Topology.IsInstance(wire, "Wire"):
                        face = Face.ByWire(wire, tolerance=tolerance, silent=True)

                    if not Topology.IsInstance(face, "Face"):
                        wire = Wire.ByEdges(
                            [edge1, bridge1, edge2, bridge2],
                            tolerance=tolerance,
                            silent=True
                        )
                        if Topology.IsInstance(wire, "Wire"):
                            face = Face.ByWire(wire, tolerance=tolerance, silent=True)

                elif bridge1 is not None:
                    wire = Wire.ByEdges(
                        [edge1, bridge1, edge2],
                        tolerance=tolerance,
                        silent=True
                    )
                    if Topology.IsInstance(wire, "Wire"):
                        face = Face.ByWire(wire, tolerance=tolerance, silent=True)

                elif bridge2 is not None:
                    wire = Wire.ByEdges(
                        [edge1, bridge2, edge2],
                        tolerance=tolerance,
                        silent=True
                    )
                    if Topology.IsInstance(wire, "Wire"):
                        face = Face.ByWire(wire, tolerance=tolerance, silent=True)

                if not Topology.IsInstance(face, "Face"):
                    continue

                if triangulate and len(Topology.Vertices(face)) > 3:
                    triangles = Face.Triangulate(face, tolerance=tolerance)
                    if isinstance(triangles, list):
                        faces.extend(triangles)
                else:
                    faces.append(face)

        return CellComplex.ByFaces(
            faces,
            tolerance=tolerance,
            silent=silent
        )

    @staticmethod
    def ByWiresCluster(
        cluster,
        polyhedron: bool = True,
        triangulate: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False
    ):
        """
        Creates a CellComplex by lofting through the Wires in the input Cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input Cluster containing the section Wires.
        polyhedron : bool , optional
            If True, constructs a faceted CellComplex. If False, constructs a
            curve-preserving CellComplex using the PythonOCC backend.
            Default is True.
        triangulate : bool , optional
            If polyhedron is True, specifies whether generated Faces are
            triangulated. Ignored when polyhedron is False. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("CellComplex.ByWiresCluster - Error: The input cluster parameter is not a valid topologic Cluster. Returning None.")
            return None

        wires = Topology.Wires(cluster)

        return CellComplex.ByWires(
            wires,
            polyhedron=polyhedron,
            triangulate=triangulate,
            tolerance=tolerance,
            silent=silent
        )

    @staticmethod
    def Cells(cellComplex, silent: bool = False) -> list:
        """Returns the cells of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Cells - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        result = []
        try:
            Core.InstanceCall(cellComplex, "Cells", None, result)
        except Exception:
            if not silent:
                print("CellComplex.Cells - Error: Could not retrieve the cells. Returning None.")
            return None
        return result

    @staticmethod
    def Cube(origin= None,
            size: float = 1.0,
            uSides: int = 2, vSides: int = 2, wSides: int = 2,
            direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a cube with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the cube. Default is None which results in the cube being placed at (0, 0, 0).
        size : float , optional
            The size of the cube. Default is 1.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int, optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the cube. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the cube. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created cube.

        """
        return CellComplex.Prism(origin=origin,
                                 width=size, length=size, height=size,
                                 uSides=uSides, vSides=vSides, wSides=wSides,
                                 direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Decompose(cellComplex, tiltAngle: float = 10.0, tolerance: float = 0.0001, silent: bool = False) -> dict:
        """
        Decomposes the input cellComplex into its logical components. This method assumes that the positive Z direction is UP.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            the input cellComplex.
        tiltAngle : float , optional
            The threshold tilt angle in degrees to determine if a face is vertical, horizontal, or tilted. The tilt angle is measured from the nearest cardinal direction. Default is 10.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "cells": list of cells
            2. "externalVerticalFaces": list of external vertical faces
            3. "internalVerticalFaces": list of internal vertical faces
            4. "topHorizontalFaces": list of top horizontal faces
            5. "bottomHorizontalFaces": list of bottom horizontal faces
            6. "internalHorizontalFaces": list of internal horizontal faces
            7. "externalInclinedFaces": list of external inclined faces
            8. "internalInclinedFaces": list of internal inclined faces
            9. "externalVerticalApertures": list of external vertical apertures
            10. "internalVerticalApertures": list of internal vertical apertures
            11. "topHorizontalApertures": list of top horizontal apertures
            12. "bottomHorizontalApertures": list of bottom horizontal apertures
            13. "internalHorizontalApertures": list of internal horizontal apertures
            14. "externalInclinedApertures": list of external inclined apertures
            15. "internalInclinedApertures": list of internal inclined apertures

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Decompose - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        
        return Topology.Decompose(topology=cellComplex, tiltAngle = tiltAngle, tolerance = tolerance, silent = silent)
    
    @staticmethod
    def Delaunay(vertices: list = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a 3D Delaunay tetrahedralisation of the input Vertices.

        Parameters
        ----------
        vertices : list, optional
            Input Vertices. If None, the vertices of a unit prism are used.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool, optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The Delaunay CellComplex, or None on failure.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        try:
            import numpy as np
            from scipy.spatial import Delaunay as SCIDelaunay
            from scipy.spatial import QhullError
        except Exception:
            if not silent:
                print("CellComplex.Delaunay - Error: scipy and numpy are required. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None
        if tolerance <= 0.0:
            return None

        if not isinstance(vertices, list):
            seed = Cell.Prism(silent=True)
            vertices = Topology.Vertices(seed) if Topology.IsInstance(seed, "Cell") else []

        vertices = [vertex for vertex in vertices if Topology.IsInstance(vertex, "Vertex")]
        if len(vertices) < 4:
            if not silent:
                print("CellComplex.Delaunay - Error: At least four valid Vertices are required. Returning None.")
            return None

        # Remove coincident points within the requested tolerance while retaining
        # the original Vertex wrappers for topology construction.
        unique_vertices = []
        unique_points = []
        quant = max(tolerance, 1e-12)
        keys = set()
        for vertex in vertices:
            coords = Vertex.Coordinates(vertex, mantissa=15)
            if not isinstance(coords, list) or len(coords) < 3:
                continue
            point = [float(coords[0]), float(coords[1]), float(coords[2])]
            key = tuple(round(value / quant) for value in point)
            if key in keys:
                continue
            keys.add(key)
            unique_vertices.append(vertex)
            unique_points.append(point)

        if len(unique_vertices) < 4:
            if not silent:
                print("CellComplex.Delaunay - Error: Fewer than four unique Vertices remain after tolerance filtering. Returning None.")
            return None

        points = np.asarray(unique_points, dtype=float)
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
            if not silent:
                print("CellComplex.Delaunay - Error: The input Vertices are coplanar or otherwise do not span 3D space. Returning None.")
            return None

        try:
            triangulation = SCIDelaunay(points, furthest_site=False)
        except (QhullError, ValueError, RuntimeError):
            if not silent:
                print("CellComplex.Delaunay - Error: SciPy could not compute a 3D Delaunay tetrahedralisation. Returning None.")
            return None

        def _cell_by_indices(indices):
            v0, v1, v2, v3 = [unique_vertices[int(index)] for index in indices]
            faces = [
                Face.ByVertices([v0, v1, v2], tolerance=tolerance, silent=True),
                Face.ByVertices([v0, v3, v1], tolerance=tolerance, silent=True),
                Face.ByVertices([v1, v3, v2], tolerance=tolerance, silent=True),
                Face.ByVertices([v2, v3, v0], tolerance=tolerance, silent=True),
            ]
            if not all(Topology.IsInstance(face, "Face") for face in faces):
                return None
            return Cell.ByFaces(faces, tolerance=tolerance, silent=True)

        cells = []
        for simplex in triangulation.simplices:
            cell = _cell_by_indices(simplex)
            if not Topology.IsInstance(cell, "Cell"):
                if not silent:
                    print("CellComplex.Delaunay - Error: Could not construct one of the Delaunay tetrahedra. Returning None.")
                return None
            cells.append(cell)

        return CellComplex.ByCells(cells, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def Edges(cellComplex, silent: bool = False) -> list:
        """Returns the edges of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Edges - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        result = []
        try:
            Core.InstanceCall(cellComplex, "Edges", None, result)
        except Exception:
            if not silent:
                print("CellComplex.Edges - Error: Could not retrieve the edges. Returning None.")
            return None
        return result

    @staticmethod
    def ExternalBoundary(cellComplex, silent: bool = False):
        """
        Returns the outer Shell of the input CellComplex.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.ExternalBoundary - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        try:
            boundary = Core.InstanceCall(cellComplex, "ExternalBoundary")
        except Exception:
            boundary = None

        if Topology.IsInstance(boundary, "Shell"):
            return boundary

        # TopologicCore may return the fused external Cell. Normalise the public
        # CellComplex API to an outer Shell on both backends.
        if Topology.IsInstance(boundary, "Topology"):
            try:
                shells = Topology.Shells(boundary)
            except Exception:
                shells = []
            if isinstance(shells, list) and len(shells) > 0:
                return shells[0]

        if not silent:
            print("CellComplex.ExternalBoundary - Error: Could not retrieve the external Shell. Returning None.")
        return None
    
    @staticmethod
    def ExternalFaces(cellComplex, silent: bool = False) -> list:
        """Returns the external Faces of the input CellComplex."""
        from topologicpy.Topology import Topology

        shell = CellComplex.ExternalBoundary(cellComplex, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            return None
        try:
            faces = Topology.Faces(shell)
        except Exception:
            faces = None
        return faces if isinstance(faces, list) else None

    @staticmethod
    def Faces(cellComplex, silent: bool = False) -> list:
        """Returns the faces of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Faces - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        result = []
        try:
            Core.InstanceCall(cellComplex, "Faces", None, result)
        except Exception:
            if not silent:
                print("CellComplex.Faces - Error: Could not retrieve the faces. Returning None.")
            return None
        return result

    @staticmethod
    def InternalFaces(cellComplex, silent: bool = False) -> list:
        """Returns the internal/non-manifold Faces of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.InternalFaces - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None
        faces = []
        try:
            Core.InstanceCall(cellComplex, "InternalBoundaries", faces)
        except Exception:
            if not silent:
                print("CellComplex.InternalFaces - Error: Could not retrieve the internal Faces. Returning None.")
            return None
        return faces
    
    @staticmethod
    def NonManifoldFaces(cellComplex, silent: bool = False) -> list:
        """Returns the non-manifold Faces of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.NonManifoldFaces - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None
        faces = []
        try:
            Core.InstanceCall(cellComplex, "NonManifoldFaces", faces)
        except Exception:
            if not silent:
                print("CellComplex.NonManifoldFaces - Error: Could not retrieve the non-manifold Faces. Returning None.")
            return None
        return faces
    
    @staticmethod
    def Octahedron(origin=None,
                   radius: float = 0.5,
                   direction: list = [0, 0, 1],
                   placement: str = "center",
                   tolerance: float = 0.0001,
                   silent: bool = False):
        """
        Creates an octahedral CellComplex consisting of two Cells separated by
        the equatorial Face.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        try:
            radius = abs(float(radius))
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.Octahedron - Error: Invalid numerical input. Returning None.")
            return None
        if radius <= tolerance or tolerance <= 0.0:
            if not silent:
                print("CellComplex.Octahedron - Error: radius must be greater than tolerance. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("CellComplex.Octahedron - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if sum(value * value for value in direction) ** 0.5 <= tolerance:
            if not silent:
                print("CellComplex.Octahedron - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        placement = str(placement).lower().strip()
        if placement not in ["center", "bottom", "lowerleft"]:
            if not silent:
                print('CellComplex.Octahedron - Error: placement must be "center", "bottom", or "lowerleft". Returning None.')
            return None

        left = Vertex.ByCoordinates(-radius, 0, 0)
        front = Vertex.ByCoordinates(0, -radius, 0)
        right = Vertex.ByCoordinates(radius, 0, 0)
        back = Vertex.ByCoordinates(0, radius, 0)
        top = Vertex.ByCoordinates(0, 0, radius)
        bottom = Vertex.ByCoordinates(0, 0, -radius)

        faces = [
            Face.ByVertices([top, left, front], tolerance=tolerance, silent=True),
            Face.ByVertices([top, front, right], tolerance=tolerance, silent=True),
            Face.ByVertices([top, right, back], tolerance=tolerance, silent=True),
            Face.ByVertices([top, back, left], tolerance=tolerance, silent=True),
            Face.ByVertices([bottom, front, left], tolerance=tolerance, silent=True),
            Face.ByVertices([bottom, right, front], tolerance=tolerance, silent=True),
            Face.ByVertices([bottom, back, right], tolerance=tolerance, silent=True),
            Face.ByVertices([bottom, left, back], tolerance=tolerance, silent=True),
            Face.ByVertices([left, front, right, back], tolerance=tolerance, silent=True),
        ]
        if not all(Topology.IsInstance(face, "Face") for face in faces):
            if not silent:
                print("CellComplex.Octahedron - Error: Could not create the required Faces. Returning None.")
            return None

        octahedron = CellComplex._ByFaces(faces, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(octahedron, "CellComplex"):
            if not silent:
                print("CellComplex.Octahedron - Error: Could not create the CellComplex. Returning None.")
            return None

        source = [0.0, 0.0, 0.0]
        if placement == "bottom":
            source = [0.0, 0.0, -radius]
        elif placement == "lowerleft":
            source = [-radius, -radius, -radius]

        return Topology.OrientAndPlace(
            octahedron,
            originA=Vertex.ByCoordinates(source),
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=silent,
        )
    
    @staticmethod
    def Prism(origin=None,
              width: float = 1.0,
              length: float = 1.0,
              height: float = 1.0,
              uSides: int = 2,
              vSides: int = 2,
              wSides: int = 2,
              direction: list = [0, 0, 1],
              placement: str = "center",
              mantissa: int = 6,
              tolerance: float = 0.0001,
              silent: bool = False):
        """
        Creates a prismatic CellComplex subdivided into a regular ``uSides`` by
        ``vSides`` by ``wSides`` grid of Cells.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        try:
            width = abs(float(width))
            length = abs(float(length))
            height = abs(float(height))
            uSides = int(uSides)
            vSides = int(vSides)
            wSides = int(wSides)
            mantissa = int(mantissa)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.Prism - Error: One or more numerical parameters are invalid. Returning None.")
            return None

        if min(width, length, height) <= tolerance or tolerance <= 0.0:
            if not silent:
                print("CellComplex.Prism - Error: width, length, and height must be greater than tolerance. Returning None.")
            return None
        if uSides < 1 or vSides < 1 or wSides < 1:
            if not silent:
                print("CellComplex.Prism - Error: uSides, vSides, and wSides must each be at least 1. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("CellComplex.Prism - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if sum(value * value for value in direction) ** 0.5 <= tolerance:
            if not silent:
                print("CellComplex.Prism - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        placement = str(placement).lower().strip()
        if placement not in ["center", "bottom", "lowerleft"]:
            if not silent:
                print('CellComplex.Prism - Error: placement must be "center", "bottom", or "lowerleft". Returning None.')
            return None

        dx = width / float(uSides)
        dy = length / float(vSides)
        dz = height / float(wSides)
        cells = []

        for i in range(uSides):
            for j in range(vSides):
                for k in range(wSides):
                    cell_origin = Vertex.ByCoordinates(i * dx, j * dy, k * dz)
                    cell = Cell.Prism(
                        origin=cell_origin,
                        width=dx,
                        length=dy,
                        height=dz,
                        uSides=1,
                        vSides=1,
                        wSides=1,
                        direction=[0, 0, 1],
                        placement="lowerleft",
                        mantissa=mantissa,
                        tolerance=tolerance,
                        silent=True,
                    )
                    if not Topology.IsInstance(cell, "Cell"):
                        if not silent:
                            print("CellComplex.Prism - Error: Could not create one of the constituent Cells. Returning None.")
                        return None
                    cells.append(cell)

        prism = CellComplex.ByCells(cells, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(prism, "CellComplex"):
            if not silent:
                print("CellComplex.Prism - Error: Could not assemble the constituent Cells. Returning None.")
            return None

        source = [0.0, 0.0, 0.0]
        if placement == "center":
            source = [0.5 * width, 0.5 * length, 0.5 * height]
        elif placement == "bottom":
            source = [0.5 * width, 0.5 * length, 0.0]

        return Topology.OrientAndPlace(
            prism,
            originA=Vertex.ByCoordinates(source),
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=silent,
        )


    @staticmethod
    def RemoveCollinearEdges(cellComplex, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = True):
        """Removes geometrically collinear straight Edges from a CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.RemoveCollinearEdges - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        faces = CellComplex.Faces(cellComplex, silent=True)
        if not isinstance(faces, list) or len(faces) == 0:
            if not silent:
                print("CellComplex.RemoveCollinearEdges - Error: Could not retrieve any Faces. Returning None.")
            return None

        clean_faces = []
        for face in faces:
            try:
                clean_face = Topology.RemoveCollinearEdges(
                    face,
                    angTolerance=angTolerance,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                clean_face = None
            clean_faces.append(clean_face if Topology.IsInstance(clean_face, "Face") else face)

        result = CellComplex._ByFaces(clean_faces, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "CellComplex"):
            if not silent:
                print("CellComplex.RemoveCollinearEdges - Error: Could not rebuild the CellComplex. Returning None.")
            return None
        return result
    
    @staticmethod
    def Shells(cellComplex, silent: bool = False) -> list:
        """Returns the Shells of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Shells - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None
        try:
            shells = Topology.Shells(cellComplex)
        except Exception:
            shells = None
        if not isinstance(shells, list):
            if not silent:
                print("CellComplex.Shells - Error: Could not retrieve the Shells. Returning None.")
            return None
        return shells

    @staticmethod
    def _grow_connected_group(seed_idx, group_size, adjacency, visited_global):
        """
        Attempts to grow a group of the given size starting from seed_idx using adjacency.
        Returns a list of indices if successful, else None.
        """
        from collections import deque
        import random

        group = [seed_idx]
        visited = set(group)
        queue = deque([seed_idx])

        while queue and len(group) < group_size:
            current = queue.popleft()
            neighbors = adjacency.get(current, [])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in visited_global:
                    group.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(group) >= group_size:
                        break

        return group if len(group) == group_size else None

    @staticmethod
    def Tetrahedron(origin=None,
                    length: float = 1,
                    depth: int = 1,
                    direction: list = [0, 0, 1],
                    placement: str = "center",
                    mantissa: int = 6,
                    tolerance: float = 0.0001,
                    silent: bool = False):
        """
        Creates a recursively subdivided regular tetrahedral CellComplex.

        Each subdivision level partitions every tetrahedron into eight smaller
        tetrahedra. The four corner tetrahedra and the four tetrahedra obtained
        by splitting the central midpoint octahedron exactly fill the parent
        tetrahedron without overlaps or voids.
        """
        from math import sqrt
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        try:
            length = abs(float(length))
            depth = max(0, int(depth))
            mantissa = int(mantissa)
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("CellComplex.Tetrahedron - Error: One or more numerical parameters are invalid. Returning None.")
            return None
        if length <= tolerance or tolerance <= 0.0:
            if not silent:
                print("CellComplex.Tetrahedron - Error: length must be greater than tolerance. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("CellComplex.Tetrahedron - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None
        try:
            direction = [float(value) for value in direction]
        except Exception:
            return None
        if sum(value * value for value in direction) ** 0.5 <= tolerance:
            if not silent:
                print("CellComplex.Tetrahedron - Error: The input direction vector has zero magnitude. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        placement = str(placement).lower().strip()
        if placement not in ["center", "bottom", "lowerleft"]:
            if not silent:
                print('CellComplex.Tetrahedron - Error: placement must be "center", "bottom", or "lowerleft". Returning None.')
            return None

        h = sqrt(2.0 / 3.0) * length
        root = (
            (0.0, 0.0, 0.0),
            (length, 0.0, 0.0),
            (0.5 * length, 0.5 * sqrt(3.0) * length, 0.0),
            (0.5 * length, sqrt(3.0) * length / 6.0, h),
        )

        def _mid(a, b):
            return tuple((a[i] + b[i]) * 0.5 for i in range(3))

        def _subdivide(tetra):
            a, b, c, d = tetra
            ab = _mid(a, b)
            ac = _mid(a, c)
            ad = _mid(a, d)
            bc = _mid(b, c)
            bd = _mid(b, d)
            cd = _mid(c, d)

            # Four corner tetrahedra plus four tetrahedra filling the central
            # octahedron, split along the opposite-vertex diagonal ab--cd.
            return [
                (a, ab, ac, ad),
                (ab, b, bc, bd),
                (ac, bc, c, cd),
                (ad, bd, cd, d),
                (ab, cd, ac, ad),
                (ab, cd, ad, bd),
                (ab, cd, bd, bc),
                (ab, cd, bc, ac),
            ]

        tetrahedra = [root]
        for _ in range(depth):
            next_level = []
            for tetra in tetrahedra:
                next_level.extend(_subdivide(tetra))
            tetrahedra = next_level

        vertex_cache = {}

        def _vertex(point):
            key = tuple(round(float(value), max(mantissa, 12)) for value in point)
            vertex = vertex_cache.get(key)
            if vertex is None:
                vertex = Vertex.ByCoordinates(point[0], point[1], point[2])
                vertex_cache[key] = vertex
            return vertex

        def _cell(tetra):
            a, b, c, d = [_vertex(point) for point in tetra]
            faces = [
                Face.ByVertices([a, b, c], tolerance=tolerance, silent=True),
                Face.ByVertices([a, d, b], tolerance=tolerance, silent=True),
                Face.ByVertices([b, d, c], tolerance=tolerance, silent=True),
                Face.ByVertices([c, d, a], tolerance=tolerance, silent=True),
            ]
            if not all(Topology.IsInstance(face, "Face") for face in faces):
                return None
            return Cell.ByFaces(faces, tolerance=tolerance, silent=True)

        cells = []
        for tetra in tetrahedra:
            cell = _cell(tetra)
            if not Topology.IsInstance(cell, "Cell"):
                if not silent:
                    print("CellComplex.Tetrahedron - Error: Could not construct one of the tetrahedral Cells. Returning None.")
                return None
            cells.append(cell)

        cell_complex = CellComplex.ByCells(cells, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(cell_complex, "CellComplex"):
            if not silent:
                print("CellComplex.Tetrahedron - Error: Could not assemble the tetrahedral Cells. Returning None.")
            return None

        centroid = [0.5 * length, sqrt(3.0) * length / 6.0, 0.25 * h]
        source = [0.0, 0.0, 0.0]
        if placement == "center":
            source = centroid
        elif placement == "bottom":
            source = [centroid[0], centroid[1], 0.0]

        return Topology.OrientAndPlace(
            cell_complex,
            originA=Vertex.ByCoordinates(source),
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=silent,
        )
    
    @staticmethod
    def Torus(
        origin=None,
        majorRadius: float = 0.5,
        minorRadius: float = 0.125,
        uSides: int = 16,
        vSides: int = 8,
        direction: list = [0, 0, 1],
        placement: str = "center",
        tolerance: float = 0.0001,
        silent: bool = False,
        polyhedron: bool = True
    ):
        """
        Creates a toroidal CellComplex.

        The torus is subdivided into ``uSides`` Cells around its major circle.
        When ``polyhedron`` is True, the historical faceted construction is
        retained. When ``polyhedron`` is False, exact OCCT toroidal sectors are
        constructed by the PythonOCC backend and assembled into a non-manifold
        CellComplex.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The placement origin of the torus. Default is None, which uses the
            global origin.
        majorRadius : float , optional
            The major radius measured from the torus centre to the centreline of
            the tube. Default is 0.5.
        minorRadius : float , optional
            The minor radius of the tube. Default is 0.125.
        uSides : int , optional
            The number of Cells around the major circle. Default is 16.
        vSides : int , optional
            The number of sides used to approximate the tube cross-section when
            ``polyhedron`` is True. Ignored when ``polyhedron`` is False.
            Default is 8.
        direction : list , optional
            The vector representing the up direction of the torus.
            Default is [0, 0, 1].
        placement : str , optional
            The placement of the input origin relative to the torus. This can be
            "center", "bottom", or "lowerleft". It is case insensitive.
            Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.
        polyhedron : bool , optional
            If True, creates the historical faceted toroidal CellComplex. If
            False, creates exact smooth toroidal sectors using the PythonOCC
            backend. Default is True.

        Returns
        -------
        topologic_core.CellComplex
            The created toroidal CellComplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        # ------------------------------------------------------------------
        # Validate common inputs.
        # ------------------------------------------------------------------
        try:
            majorRadius = float(majorRadius)
            minorRadius = float(minorRadius)
            uSides = int(uSides)
            tolerance = float(tolerance)
        except Exception:
            if not silent:
                print("CellComplex.Torus - Error: Invalid numerical input. Returning None.")
            return None

        if majorRadius <= tolerance or minorRadius <= tolerance:
            if not silent:
                print("CellComplex.Torus - Error: majorRadius and minorRadius must be greater than tolerance. Returning None.")
            return None

        if minorRadius >= majorRadius:
            if not silent:
                print("CellComplex.Torus - Error: minorRadius must be smaller than majorRadius. Returning None.")
            return None

        if uSides < 3:
            if not silent:
                print("CellComplex.Torus - Error: uSides must be at least 3. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("CellComplex.Torus - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
            if sum(value * value for value in direction) ** 0.5 <= tolerance:
                if not silent:
                    print("CellComplex.Torus - Error: The input direction vector has zero magnitude. Returning None.")
                return None
        except Exception:
            if not silent:
                print("CellComplex.Torus - Error: The input direction parameter is not a valid numerical vector. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()

        placement = str(placement).lower().strip()
        if placement not in ["center", "bottom", "lowerleft"]:
            if not silent:
                print('CellComplex.Torus - Error: placement must be "center", "bottom", or "lowerleft". Returning None.')
            return None

        # ------------------------------------------------------------------
        # Exact smooth toroidal CellComplex.
        # ------------------------------------------------------------------
        if not polyhedron:
            try:
                if Topology._IsTopologicCoreBackend():
                    if not silent:
                        print("CellComplex.Torus - Error: polyhedron=False requires the PythonOCC backend. Returning None.")
                    return None
            except Exception:
                return None

            method = getattr(Core.CellComplex, "ByTorus", None)
            if not callable(method):
                if not silent:
                    print("CellComplex.Torus - Error: Native backend constructor is unavailable. Returning None.")
                return None

            try:
                torus = method(
                    majorRadius=majorRadius,
                    minorRadius=minorRadius,
                    uSides=uSides,
                    tolerance=tolerance,
                    silent=silent,
                )
            except Exception:
                torus = None

            if not Topology.IsInstance(torus, "CellComplex"):
                if not silent:
                    print("CellComplex.Torus - Error: Could not create the smooth toroidal CellComplex. Returning None.")
                return None

            source = [0.0, 0.0, 0.0]
            if placement == "bottom":
                source = [0.0, 0.0, -minorRadius]
            elif placement == "lowerleft":
                extent = majorRadius + minorRadius
                source = [-extent, -extent, -minorRadius]

            source_vertex = Vertex.ByCoordinates(source[0], source[1], source[2])

            return Topology.OrientAndPlace(
                torus,
                originA=source_vertex,
                originB=origin,
                dirA=[0, 0, 1],
                dirB=direction,
                transferDictionaries=False,
                tolerance=tolerance,
                silent=silent,
            )

        # ------------------------------------------------------------------
        # Historical faceted torus.
        # ------------------------------------------------------------------
        try:
            vSides = int(vSides)
        except Exception:
            if not silent:
                print("CellComplex.Torus - Error: vSides must be a valid integer. Returning None.")
            return None

        if vSides < 3:
            if not silent:
                print("CellComplex.Torus - Error: vSides must be at least 3. Returning None.")
            return None

        c = Wire.Circle(
            origin=Vertex.Origin(),
            radius=minorRadius,
            sides=vSides,
            fromAngle=0,
            toAngle=360,
            close=False,
            direction=[0, 1, 0],
            placement="center",
            tolerance=tolerance,
            silent=silent,
        )

        if not Topology.IsInstance(c, "Wire"):
            if not silent:
                print("CellComplex.Torus - Error: Could not create the torus section Wire. Returning None.")
            return None

        c = Face.ByWire(c, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(c, "Face"):
            if not silent:
                print("CellComplex.Torus - Error: Could not create the torus section Face. Returning None.")
            return None

        # Place the tube cross-section centre on the standard torus major radius.
        c = Topology.Translate(c, majorRadius, 0, 0)
        torus = Topology.Spin(
            c,
            origin=Vertex.Origin(),
            triangulate=False,
            direction=[0, 0, 1],
            angle=360,
            sides=uSides,
            tolerance=tolerance,
            silent=silent,
        )

        if Topology.IsInstance(torus, "Shell"):
            torus = CellComplex.ByFaces(
                Topology.Faces(torus),
                tolerance=tolerance,
                silent=silent,
            )

        if not Topology.IsInstance(torus, "CellComplex"):
            if not silent:
                print("CellComplex.Torus - Error: Could not create the faceted toroidal CellComplex. Returning None.")
            return None

        source = [0.0, 0.0, 0.0]
        if placement == "bottom":
            source = [0.0, 0.0, -minorRadius]
        elif placement == "lowerleft":
            extent = majorRadius + minorRadius
            source = [-extent, -extent, -minorRadius]

        return Topology.OrientAndPlace(
            torus,
            originA=Vertex.ByCoordinates(source[0], source[1], source[2]),
            originB=origin,
            dirA=[0, 0, 1],
            dirB=direction,
            transferDictionaries=False,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Vertices(cellComplex, silent: bool = False) -> list:
        """Returns the vertices of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Vertices - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        result = []
        try:
            Core.InstanceCall(cellComplex, "Vertices", None, result)
        except Exception:
            if not silent:
                print("CellComplex.Vertices - Error: Could not retrieve the vertices. Returning None.")
            return None
        return result

    @staticmethod
    def Volume(cellComplex, mantissa: int = 6, silent: bool = False) -> float:
        """
        Returns the volume of the input CellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input CellComplex.
        mantissa : int, optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool, optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        float
            The volume of the input CellComplex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print(
                    "CellComplex.Volume - Error: The input cellComplex parameter "
                    "is not a valid CellComplex. Returning None."
                )
            return None

        try:
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print(
                    "CellComplex.Volume - Error: The input mantissa parameter "
                    "is not a valid integer. Returning None."
                )
            return None

        cells = CellComplex.Cells(cellComplex)

        if not isinstance(cells, list) or len(cells) == 0:
            if not silent:
                print(
                    "CellComplex.Volume - Error: Could not retrieve any Cells "
                    "from the input CellComplex. Returning None."
                )
            return None

        volume = 0.0

        for cell in cells:
            try:
                cell_volume = Core.CellUtility.Volume(cell)
            except Exception:
                if not silent:
                    print(
                        "CellComplex.Volume - Error: Could not compute the volume "
                        "of one or more Cells. Returning None."
                    )
                return None

            try:
                cell_volume = float(cell_volume)
            except Exception:
                if not silent:
                    print(
                        "CellComplex.Volume - Error: The backend returned an invalid "
                        "Cell volume. Returning None."
                    )
                return None

            volume += cell_volume

        return round(volume, mantissa)
    
    @staticmethod
    def Voronoi(vertices: list = None, cell=None, tolerance: float = 0.0001, silent: bool = False):
        """Partitions a Cell using a 3D Voronoi diagram of the input Vertices."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        try:
            import numpy as np
            from scipy.spatial import Voronoi as SCIVoronoi
            from scipy.spatial import QhullError
        except Exception:
            if not silent:
                print("CellComplex.Voronoi - Error: scipy and numpy are required. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None
        if tolerance <= 0.0:
            return None

        if not Topology.IsInstance(cell, "Cell"):
            if isinstance(vertices, list):
                seed_vertices = [vertex for vertex in vertices if Topology.IsInstance(vertex, "Vertex")]
                if len(seed_vertices) == 0:
                    if not silent:
                        print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any valid Vertices. Returning None.")
                    return None
                try:
                    cell = Topology.BoundingBox(Cluster.ByTopologies(seed_vertices))
                except Exception:
                    cell = None
            else:
                cell = Cell.Prism(uSides=1, vSides=1, wSides=1, silent=True)

        if not Topology.IsInstance(cell, "Cell"):
            if not silent:
                print("CellComplex.Voronoi - Error: Could not determine a valid bounding Cell. Returning None.")
            return None

        if not isinstance(vertices, list):
            vertices = []
        vertices = [vertex for vertex in vertices if Topology.IsInstance(vertex, "Vertex")]

        boundary_vertices = Topology.Vertices(cell)
        if isinstance(boundary_vertices, list):
            vertices = vertices + boundary_vertices

        # Keep only points inside or on the bounding Cell and remove coincident
        # points before calling Qhull.
        accepted = []
        points = []
        keys = set()
        quant = max(tolerance, 1e-12)
        cell_vertices = Topology.Vertices(cell)

        for vertex in vertices:
            try:
                on_boundary = Vertex.Index(vertex, cell_vertices, tolerance=tolerance) is not None
                inside = Vertex.IsInternal(vertex, cell, tolerance=tolerance, silent=True)
            except TypeError:
                inside = Vertex.IsInternal(vertex, cell, tolerance=tolerance)
                on_boundary = Vertex.Index(vertex, cell_vertices, tolerance=tolerance) is not None
            except Exception:
                continue
            if not inside and not on_boundary:
                continue

            coords = Vertex.Coordinates(vertex, mantissa=15)
            if not isinstance(coords, list) or len(coords) < 3:
                continue
            point = [float(coords[0]), float(coords[1]), float(coords[2])]
            key = tuple(round(value / quant) for value in point)
            if key in keys:
                continue
            keys.add(key)
            accepted.append(vertex)
            points.append(point)

        if len(points) < 5:
            if not silent:
                print("CellComplex.Voronoi - Error: At least five unique 3D points inside/on the bounding Cell are required. Returning None.")
            return None

        point_array = np.asarray(points, dtype=float)
        if np.linalg.matrix_rank(point_array - point_array.mean(axis=0)) < 3:
            if not silent:
                print("CellComplex.Voronoi - Error: The input points do not span 3D space. Returning None.")
            return None

        try:
            voronoi = SCIVoronoi(point_array)
        except (QhullError, ValueError, RuntimeError):
            if not silent:
                print("CellComplex.Voronoi - Error: SciPy could not compute the Voronoi diagram. Returning None.")
            return None

        voronoi_vertices = [Vertex.ByCoordinates(list(point)) for point in voronoi.vertices]
        ridge_faces = []
        for region in voronoi.ridge_vertices:
            if -1 in region or len(region) < 3:
                continue
            try:
                ridge_vertices = [voronoi_vertices[index] for index in region]
            except Exception:
                continue
            face = Face.ByVertices(ridge_vertices, tolerance=tolerance, silent=True)
            if Topology.IsInstance(face, "Face"):
                ridge_faces.append(face)

        if len(ridge_faces) == 0:
            if not silent:
                print("CellComplex.Voronoi - Error: The Voronoi diagram produced no finite ridge Faces. Returning None.")
            return None

        cutters = Cluster.ByTopologies(ridge_faces)
        try:
            result = Topology.Slice(cell, cutters, tolerance=tolerance, silent=True)
        except Exception:
            result = None

        if not Topology.IsInstance(result, "CellComplex"):
            if not silent:
                print("CellComplex.Voronoi - Error: Could not partition the bounding Cell. Returning None.")
            return None
        return result
    
    @staticmethod
    def Wires(cellComplex, silent: bool = False) -> list:
        """Returns the wires of the input CellComplex."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Wires - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None

        result = []
        try:
            Core.InstanceCall(cellComplex, "Wires", None, result)
        except Exception:
            if not silent:
                print("CellComplex.Wires - Error: Could not retrieve the wires. Returning None.")
            return None
        return result

