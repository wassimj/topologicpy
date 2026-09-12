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
import os
import warnings

try:
    import numpy as np
except:
    print("CellComplex - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("CellComplex - numpy library installed correctly.")
    except:
        warnings.warn("CellComplex - Error: Could not import numpy.")
try:
    from scipy.spatial import Delaunay
    from scipy.spatial import Voronoi
except:
    print("CellComplex - Install required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial import Voronoi
    except:
        warnings.warn("CellComplex - Error: Could not import scipy.")

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
                        minCells: float = 2,
                        maxCells: float = 10,
                        maxAttempts: int = 100,
                        patience: int = 5,
                        transferDictionaries: bool = False,
                        exclusive: bool = True,
                        tolerance: float = 0.0001,
                        silent: bool = False):
        """
        Creates a CellComplex from a list of disjointed faces. The algorithm expands the faces by an offset to find intersections before building cells.

        Parameters
        ----------
        faces : list of topologic_core.Face
            The linput ist of faces.
        minOffset : float , optional
            The minimum initial face offset to try. Default is 0.
        maxOffset : float , optional
            The final maximum face offset to try. Default is 1.0.
        minCells : int , optional
            The minimum number of cells to create. A CellComplex cannot have less than 2 cells. Default is 2.
        maxCells : int , optional
            The maximum number of cells to create. Default is 10.
        maxAttempts : int , optional
            The desired maximum number of attempts. Default is 100.
        patience : int , optional
            The desired number of attempts to wait with no change in the created number of cells. Default is 5.
        transferDictionaries : bool , optional
            If set to True, face dictionaries are inhertied. Default is False.
        exclusive : bool , optional
            Applies only if transferDictionaries is set to True. If set to True, only one source face contributes its dictionary to a target face. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex

        """

        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        def trim(cells, n):
            volumes = [Cell.Volume(c) for c in cells]
            return_cc = Helper.Sort(cells, volumes)
            return_cc.reverse()
            return return_cc[:n]

        faces = [f for f in faces if Topology.IsInstance(f, "Face")]
        if len(faces) == 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input list of faces does not contain any valid topologic faces. Returning None.")
            return None
        if len(faces) < 3:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input list of faces contains less than three topologic faces. Returning None.")
            return None
        if minOffset < 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input minOffset parameter is less than 0. Returning None.")
            return None
        if minOffset > maxOffset:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input minOffset parameter is greater than the input maxOffset parameter. Returning None.")
            return None
        if minCells < 2:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input minCells parameter is less than 2. Returning None.")
            return None
        if minCells > maxCells:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input minCells parameter is greater than the input maxCells parameter. Returning None.")
            return None
        if maxAttempts <= 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input maxAttempts parameter is not greater than 0. Returning None.")
            return None
        if patience < 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input patience parameter is not greater than or equal to 0. Returning None.")
            return None
        if patience > maxAttempts:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: The input patience parameter is greater than the input maxAttempts parameter. Returning None.")
            return None
        cc = None
        attempts = 0
        increment = float(maxOffset) / float(maxAttempts)
        cellComplexes = [] # List of all possible cellComplexes
        patience_list = []
        offset = minOffset
        
        while attempts < maxAttempts:
            expanded_faces = [Face.ByOffset(f, offset=-offset, silent=silent) for f in faces]
            try:
                cc = CellComplex.ByFaces(expanded_faces, silent=True)
                if Topology.IsInstance(cc, "cellComplex"):
                    cells = Topology.Cells(cc)
                    n_cells = len(cells)
                    if minCells <= n_cells <= maxCells:
                        cellComplexes.append(cc)
                    elif n_cells > maxCells:
                        cells = trim(cells, maxCells)
                        try:
                            new_cc = CellComplex.ByCells(cells)
                            if Topology.IsInstance(new_cc, "CellComplex"):
                                cellComplexes.append(new_cc)
                        except:
                            pass
                    patience_list.append(n_cells)
            except:
                patience_list.append(0)
            
            if len(patience_list) >= patience:
                if len(set(patience_list)) == 1 and not patience_list[0] == 0:
                    if not silent:
                        print("CellComplex.ByDisjointedFaces - Warning: Ran out of patience.")
                    break
                else:
                    patience_list = []
            attempts += 1
            offset += increment

        if len(cellComplexes) == 0:
            if not silent:
                print("CellComplex.ByDisjointedFaces - Error: Could not create a CellComplex. Consider revising the input parameters. Returning None.")
            return None
        n_cells = [len(Topology.Cells(c)) for c in cellComplexes] # Get the number of cells in each cellComplex
        cellComplexes = Helper.Sort(cellComplexes, n_cells) # Sort the cellComplexes by their number of cells
        for cc in cellComplexes:
            cells = Topology.Cells(cc)
        cc = cellComplexes[-1] # Choose the last cellComplex (the one with the most number of cells)
        if transferDictionaries == True:
            cc_faces = Topology.Faces(cc)
            cc_faces = Topology.Inherit(targets=cc_faces, sources=faces, exclusive=exclusive, tolerance=tolerance, silent=silent)
        return cc

    
    @staticmethod
    def _ByFaces(faces: list, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a cellcomplex by merging the input faces.

        Parameters
        ----------
        faces : list
            The input faces.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created cellcomplex.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(faces, list):
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None
        faces = [x for x in faces if Topology.IsInstance(x, "Face")]
        if len(faces) < 1:
            if not silent:
                print("CellComplex.ByFaces - Error: The input faces parameter does not contain any valid faces. Returning None.")
            return None
        try:
            cellComplex = Core.CellComplex.ByFaces(faces, tolerance, False)
        except:
            cellComplex = None
        if not cellComplex:
            if not silent:
                print("CellComplex.ByFaces - Warning: The default method failed. Attempting a workaround.")
            cellComplex = faces[0]
            for i in range(1,len(faces)):
                newCellComplex = None
                try:
                    # newCellComplex = cellComplex.Merge(faces[i], False, tolerance) # H to Core
                    newCellComplex = Core.InstanceCall(cellComplex, "Merge", faces[i], False, tolerance)
                except:
                    if not silent:
                        print("CellComplex.ByFaces - Warning: Failed to merge face #"+str(i)+". Skipping.")
                if newCellComplex:
                    cellComplex = newCellComplex
            if not Topology.Type(cellComplex) == Topology.TypeID("CellComplex"):
                if not silent:
                    print("CellComplex.ByFaces - Warning: The input faces do not form a cellcomplex")
                if Topology.Type(cellComplex) == Topology.TypeID("Cluster"):
                    returnCellComplexes = Cluster.CellComplexes(cellComplex)
                    if len(returnCellComplexes) > 0:
                        return returnCellComplexes[0]
                    else:
                        if not silent:
                            print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                        return None
                else:
                    if not silent:
                        print("CellComplex.ByFaces - Error: Could not create a cellcomplex. Returning None.")
                    return None
        else:
            return cellComplex

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

        from topologicpy.CellComplex import CellComplex
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
        Creates a CellComplex from the input faces after using Shapely to remove
        coplanar face overlaps.

        This method is intended as a faster pre-processing pathway for cases where
        CellComplex.ByFaces is slow because the input contains overlapping coplanar
        faces. Non-coplanar faces are passed through unchanged.

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

        from topologicpy.CellComplex import CellComplex
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
                print("CellComplex.ByFaces - Error: Could not create the CellComplex. Returning None.")
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
                print("Enclosing Faces:", len(enclosing_faces))
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
    def ByWires(wires: list,
                triangulate: bool = True,
                tolerance: float = 0.0001,
                polyhedron: bool = True,
                silent: bool = False):
        """
        Creates a CellComplex by lofting through the input Wires.

        When ``polyhedron`` is True, the existing faceted construction is used.
        When ``polyhedron`` is False, the section curves are preserved and each
        interval is constructed natively by the PythonOCC backend.

        Parameters
        ----------
        wires : list
            The ordered input list of Wires. A minimum of two valid Wires is required.
        triangulate : bool , optional
            If ``polyhedron`` is True, specifies whether generated Faces are
            triangulated. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        polyhedron : bool , optional
            If True, uses the historical faceted construction. If False, requests
            an exact curve-preserving PythonOCC CellComplex. Default is True.
        silent : bool , optional
            If True, suppresses error and warning messages. Default is False.

        Returns
        -------
        topologic_core.CellComplex
            The created CellComplex, or None on failure.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not isinstance(wires, list):
            if not silent:
                print("CellComplex.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
            return None
        wires = [x for x in wires if Topology.IsInstance(x, "Wire")]
        if len(wires) < 2:
            if not silent:
                print("CellComplex.ByWires - Error: The input wires parameter contains fewer than two valid Wires. Returning None.")
            return None

        # Exact curve-preserving path. Leave the historical faceted path below untouched.
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
                result = method(wires, tolerance=tolerance)
            except Exception:
                result = None
            if not Topology.IsInstance(result, "CellComplex"):
                if not silent:
                    print("CellComplex.ByWires - Error: Could not create a curve-preserving CellComplex. Returning None.")
                return None
            return result

        # Historical v0.9.68 faceted construction (kept intentionally intact).
        faces = [Face.ByWire(wires[0], tolerance=tolerance), Face.ByWire(wires[-1], tolerance=tolerance)]
        if triangulate == True:
            triangles = []
            for face in faces:
                if len(Topology.Vertices(face)) > 3:
                    triangles += Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles += [face]
            faces = triangles
        for i in range(len(wires)-1):
            wire1 = wires[i]
            wire2 = wires[i+1]
            f = Face.ByWire(wire2, tolerance=tolerance)
            if triangulate == True:
                if len(Topology.Vertices(face)) > 3:
                    triangles = Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles = [face]
                faces += triangles
            else:
                faces.append(f)
            w1_edges = Topology.Edges(wire1)
            w2_edges = Topology.Edges(wire2)
            if len(w1_edges) != len(w2_edges):
                if not silent:
                    print("CellComplex.ByWires - Error: The input wires parameter contains wires with different number of edges. Returning None.")
                return None

            def _bridge(v_a, v_b):
                try:
                    be = Edge.ByStartVertexEndVertex(v_a, v_b, tolerance=tolerance, silent=True)
                except Exception:
                    be = None
                return be

            for j in range(len(w1_edges)):
                e1 = w1_edges[j]
                e2 = w2_edges[j]
                e3 = _bridge(Edge.StartVertex(e1), Edge.StartVertex(e2))
                e4 = _bridge(Edge.EndVertex(e1), Edge.EndVertex(e2))

                f = None
                if e3 is not None and e4 is not None:
                    f = Face.ByWire(Wire.ByEdges([e1, e4, e2, e3], tolerance=tolerance), tolerance=tolerance)
                    if f is None:
                        f = Face.ByWire(Wire.ByEdges([e1, e3, e2, e4], tolerance=tolerance), tolerance=tolerance)
                elif e3 is not None:
                    f = Face.ByWire(Wire.ByEdges([e1, e3, e2], tolerance=tolerance), tolerance=tolerance)
                elif e4 is not None:
                    f = Face.ByWire(Wire.ByEdges([e1, e4, e2], tolerance=tolerance), tolerance=tolerance)
                if f is None:
                    continue
                if triangulate == True:
                    if len(Topology.Vertices(f)) > 3:
                        faces.extend(Face.Triangulate(f, tolerance=tolerance))
                    else:
                        faces.append(f)
                else:
                    faces.append(f)
        return CellComplex.ByFaces(faces, tolerance=tolerance)

    @staticmethod
    def ByWiresCluster(cluster,
                       triangulate: bool = True,
                       tolerance: float = 0.0001,
                       polyhedron: bool = True,
                       silent: bool = False):
        """Creates a CellComplex by lofting through Wires in the input Cluster."""
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("CellComplex.ByWiresCluster - Error: The input cluster parameter is not a valid Cluster. Returning None.")
            return None
        wires = Topology.Wires(cluster)
        return CellComplex.ByWires(
            wires,
            triangulate=triangulate,
            tolerance=tolerance,
            polyhedron=polyhedron,
            silent=silent,
        )

    @staticmethod
    def Cells(cellComplex) -> list:
        """
        Returns the cells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of cells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Cells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        cells = []
        # _ = cellComplex.Cells(None, cells) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "Cells", None, cells)
        except Exception:
            cells = None
        return cells

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
    def Delaunay(vertices: list = None, tolerance: float = 0.0001):
        """
        Triangulates the input vertices based on the Delaunay method. See https://en.wikipedia.org/wiki/Delaunay_triangulation.

        Parameters
        ----------
        vertices: list , optional 
            The input list of vertices to use for delaunay triangulation. If set to None, the algorithm uses the vertices of the input cell parameter.
            if both are set to none, a unit cube centered around the origin is used.
        tolerance : float , optional
            the desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.CellComplex
            The created delaunay cellComplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from scipy.spatial import Delaunay as SCIDelaunay
        import numpy as np

        if not isinstance(vertices, list):
            cell = Cell.Prism()
            vertices = Topology.Vertices(cell)
        
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 3:
            print("CellComplex/Delaunay - Error: The input vertices parameter does not contain enough valid vertices. Returning None.")
            return None
        # Get the vertices of the input cell
        points = np.array([Vertex.Coordinates(v) for v in vertices])
        # Compute Delaunay triangulation
        triangulation = SCIDelaunay(points, furthest_site=False)

        faces = []
        for simplex in triangulation.simplices:
            tetrahedron_vertices = points[simplex]
            verts = [Vertex.ByCoordinates(list(coord)) for coord in tetrahedron_vertices]
            tri1 = [verts[0], verts[1], verts[2], verts[0]]
            tri2 = [verts[0], verts[2], verts[3], verts[0]]
            tri3 = [verts[0], verts[1], verts[3], verts[0]]
            tri4 = [verts[1], verts[2], verts[3], verts[1]]
            f1 = Face.ByVertices(tri1)
            f2 = Face.ByVertices(tri2)
            f3 = Face.ByVertices(tri3)
            f4 = Face.ByVertices(tri4)
            faces.append(f1)
            faces.append(f2)
            faces.append(f3)
            faces.append(f4)
        cc = Topology.RemoveCoplanarFaces(CellComplex.ByFaces(faces, tolerance=tolerance))
        faces = [Topology.RemoveCollinearEdges(f) for f in Topology.Faces(cc)]
        cc = CellComplex.ByFaces(faces)
        return cc
    
    @staticmethod
    def Edges(cellComplex, silent: bool = False) -> list:
        """
        Returns the edges of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.
        silent: bool , optional
            if set to True, no error or warning messages are printed. Default is False.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Edges - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        edges = []
        # _ = cellComplex.Edges(None, edges) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "Edges", None, edges)
        except Exception:
            if not silent:
                Topology.Show(cellComplex, renderer="browser")
                print(f"CellComplex.Edges - Error: Could not fetch edges. Returning None.")
            edges = None
        return edges

    @staticmethod
    def ExternalBoundary(cellComplex, silent: bool = False):
        """
        Returns the external boundary (shell) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The external boundary of the input cellComplex.

        """
        import inspect
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.ExternalBoundary - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
                print("Incoming Topology:", cellComplex)
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        #cell = cellComplex.ExternalBoundary() # H to Core
        try:
            cell = Core.InstanceCall(cellComplex,"ExternalBoundary")
        except Exception:
            cell = None
        if cell is not None:
            shells = Topology.Shells(cell)
            if isinstance(shells, list):
                if len(shells) > 0:
                    return shells[0]
        return None
    
    @staticmethod
    def ExternalFaces(cellComplex) -> list:
        """
        Returns the external faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of external faces.

        """
        from topologicpy.Topology import Topology
        shell = CellComplex.ExternalBoundary(cellComplex)
        return Topology.Faces(shell)

    @staticmethod
    def Faces(cellComplex) -> list:
        """
        Returns the faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Faces - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        faces = []
        # _ = cellComplex.Faces(None, faces) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "Faces", None, faces)
        except Exception:
            faces = None
        return faces

    @staticmethod
    def InternalFaces(cellComplex) -> list:
        """
        Returns the internal boundaries (faces) of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of internal faces of the input cellComplex.

        """
        faces = []
        # _ = cellComplex.InternalBoundaries(faces) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "InternalBoundaries", faces)
        except Exception:
            faces = []
        return faces
    
    @staticmethod
    def NonManifoldFaces(cellComplex) -> list:
        """
        Returns the non-manifold faces of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of non-manifold faces of the input cellComplex.

        """
        faces = []
        # _ = cellComplex.NonManifoldFaces(faces) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "NonManifoldFaces", faces)
        except Exception:
            faces = None
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
    def Prism(origin= None,
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
        Creates a prismatic cellComplex with internal cells.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the prism. Default is None which results in the prism being placed at (0, 0, 0).
        width : float , optional
            The width of the prism. Default is 1.
        length : float , optional
            The length of the prism. Default is 1.
        height : float , optional
            The height of the prism.
        uSides : int , optional
            The number of sides along the width. Default is 1.
        vSides : int , optional
            The number of sides along the length. Default is 1.
        wSides : int , optional
            The number of sides along the height. Default is 1.
        direction : list , optional
            The vector representing the up direction of the prism. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the prism. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.CellComplex
            The created prism.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        # Reject invalid subdivision counts before calculating offsets.
        division_values = {"uSides": uSides, "vSides": vSides, "wSides": wSides}
        validated_divisions = {}
        for name, value in division_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                if not silent:
                    print(f"CellComplex.Prism - Error: {name} must be an integer greater than or equal to 1. Returning None.")
                return None
            if not float(value).is_integer() or int(value) < 1:
                if not silent:
                    print(f"CellComplex.Prism - Error: {name} must be an integer greater than or equal to 1. Returning None.")
                return None
            validated_divisions[name] = int(value)

        uSides = validated_divisions["uSides"]
        vSides = validated_divisions["vSides"]
        wSides = validated_divisions["wSides"]
        
        def bb(topology):
            vertices = Topology.Vertices(topology)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
                z.append(Vertex.Z(aVertex, mantissa=mantissa))
            x_min = min(x)
            y_min = min(y)
            z_min = min(z)
            maxX = max(x)
            maxY = max(y)
            maxZ = max(z)
            return [x_min, y_min, z_min, maxX, maxY, maxZ]
        
        def slice(topology, uSides, vSides, wSides):
            x_min, y_min, z_min, maxX, maxY, maxZ = bb(topology)
            centroid = Vertex.ByCoordinates(x_min+(maxX-x_min)*0.5, y_min+(maxY-y_min)*0.5, z_min+(maxZ-z_min)*0.5)
            wOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), Vertex.Y(centroid, mantissa=mantissa), z_min)
            wFace = Face.Rectangle(origin=wOrigin, width=(maxX-x_min)*1.1, length=(maxY-y_min)*1.1)
            wFaces = []
            wOffset = (maxZ-z_min)/wSides
            for i in range(wSides-1):
                wFaces.append(Topology.Translate(wFace, 0,0,wOffset*(i+1)))
            uOrigin = Vertex.ByCoordinates(x_min, Vertex.Y(centroid, mantissa=mantissa), Vertex.Z(centroid, mantissa=mantissa))
            uFace = Face.Rectangle(origin=uOrigin, width=(maxZ-z_min)*1.1, length=(maxY-y_min)*1.1, direction=[1,0,0])
            uFaces = []
            uOffset = (maxX-x_min)/uSides
            for i in range(uSides-1):
                uFaces.append(Topology.Translate(uFace, uOffset*(i+1),0,0))
            vOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), y_min, Vertex.Z(centroid, mantissa=mantissa))
            vFace = Face.Rectangle(origin=vOrigin, width=(maxX-x_min)*1.1, length=(maxZ-z_min)*1.1, direction=[0,1,0])
            vFaces = []
            vOffset = (maxY-y_min)/vSides
            for i in range(vSides-1):
                vFaces.append(Topology.Translate(vFace, 0,vOffset*(i+1),0))
            all_faces = uFaces+vFaces+wFaces
            if len(all_faces) > 0:
                f_clus = Cluster.ByTopologies(uFaces+vFaces+wFaces)
                return Topology.Slice(topology, f_clus, tolerance=tolerance, silent=silent)
            else:
                return CellComplex.ByCells([topology])
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)

        c = Cell.Prism(origin=origin, width=width, length=length, height=height, uSides=1, vSides=1, wSides=1, placement=placement, mantissa=mantissa, tolerance=tolerance, silent=silent)
        prism = slice(c, uSides=uSides, vSides=vSides, wSides=wSides)
        if prism:
            prism = Topology.Orient(prism, origin=origin, dirA=[0, 0, 1], dirB=direction)
            return prism
        else:
            if not silent:
                print("CellComplex.Prism - Error: Could not create a prism. Returning None.")
            return None

    @staticmethod
    def RemoveCollinearEdges(cellComplex, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = True):
        """
        Removes any collinear edges in the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is True.

        Returns
        -------
        topologic_core.CellComplex
            The created cellComplex without any collinear edges.

        """

        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                import inspect
                print("CellComplex.RemoveCollinearEdges - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
                print("CellComplex.RemoveCollinearEdges - Inspection:")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print("caller name:", calframe[1][3])
            return None

        faces = CellComplex.Faces(cellComplex)

        if not isinstance(faces, list) or len(faces) == 0:
            if not silent:
                print("CellComplex.RemoveCollinearEdges - Error: Could not retrieve any faces from the input cellComplex. Returning None.")
            return None

        clean_faces = []

        for face in faces:
            try:
                clean_face = Topology.RemoveCollinearEdges(
                    face,
                    angTolerance=angTolerance,
                    tolerance=tolerance,
                    silent=silent
                )
            except TypeError:
                try:
                    clean_face = Topology.RemoveCollinearEdges(
                        face,
                        angTolerance=angTolerance,
                        tolerance=tolerance
                    )
                except:
                    clean_face = None
            except:
                clean_face = None

            if Topology.IsInstance(clean_face, "Face"):
                clean_faces.append(clean_face)
            elif Topology.IsInstance(face, "Face"):
                clean_faces.append(face)

        if len(clean_faces) == 0:
            if not silent:
                print("CellComplex.RemoveCollinearEdges - Error: No valid faces remained after removing collinear edges. Returning None.")
            return None

        return CellComplex.ByFaces(clean_faces, tolerance=tolerance, silent=silent)
    
    @staticmethod
    def Shells(cellComplex) -> list:
        """
        Returns the shells of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of shells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Shells - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        shells = Topology.Shells(cellComplex)
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
    def Torus(origin=None,
              majorRadius: float = 0.5,
              minorRadius: float = 0.125,
              uSides: int = 16,
              vSides: int = 8,
              direction: list = [0, 0, 1],
              placement: str = "center",
              tolerance: float = 0.0001,
              silent: bool = False,
              polyhedron: bool = True):
        """Creates a toroidal CellComplex.

        ``polyhedron=True`` preserves the existing faceted v0.9.68 construction.
        ``polyhedron=False`` creates exact OCCT toroidal sectors on PythonOCC.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("CellComplex.Torus - Error: The input origin parameter is not a valid Vertex. Returning None.")
            return None

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
                    print("CellComplex.Torus - Error: Could not create an exact toroidal CellComplex. Returning None.")
                return None

            xOffset = yOffset = zOffset = 0.0
            if str(placement).lower() == "bottom":
                zOffset = -float(minorRadius)
            elif str(placement).lower() == "lowerleft":
                extent = float(majorRadius) + float(minorRadius)
                xOffset = -extent
                yOffset = -extent
                zOffset = -float(minorRadius)
            return Topology.OrientAndPlace(
                torus,
                originA=Vertex.ByCoordinates(xOffset, yOffset, zOffset),
                originB=origin,
                dirA=[0, 0, 1],
                dirB=direction,
                transferDictionaries=False,
                tolerance=tolerance,
                silent=silent,
            )

        # Historical v0.9.68 faceted construction (kept intentionally intact).
        c = Wire.Circle(origin=Vertex.Origin(), radius=minorRadius, sides=vSides, polyline=polyhedron, fromAngle=0, toAngle=360, close=False, direction=[0, 1, 0], placement="center")
        c = Face.ByWire(c)
        c = Topology.Translate(c, abs(majorRadius-minorRadius), 0, 0)
        torus = Topology.Spin(c, origin=Vertex.Origin(), triangulate=False, direction=[0, 0, 1], angle=360, sides=uSides, tolerance=tolerance)
        if Topology.Type(torus) == Topology.TypeID("Shell"):
            faces = Topology.Faces(torus)
            torus = CellComplex.ByFaces(faces)

        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "bottom":
            zOffset = minorRadius
        elif placement.lower() == "lowerleft":
            xOffset = majorRadius
            yOffset = majorRadius
            zOffset = minorRadius

        torus = Topology.Orient(torus, origin=Vertex.Origin(), dirA=[0, 0, 1], dirB=direction)
        torus = Topology.Place(torus, originA=Vertex.Origin(), originB=origin)
        torus = Topology.OrientAndPlace(torus,
                                        originA=Vertex.ByCoordinates(xOffset, yOffset, zOffset),
                                        originB=origin,
                                        dirA=[0, 0, 1],
                                        dirB=direction,
                                        transferDictionaries = False,
                                        tolerance = tolerance,
                                        silent = silent)
        return torus

    @staticmethod
    def Vertices(cellComplex) -> list:
        """
        Returns the vertices of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Vertices - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        vertices = []
        # _ = cellComplex.Vertices(None, vertices) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "Vertices", None, vertices)
        except Exception:
            vertices = None
        return vertices

    @staticmethod
    def Volume(cellComplex, mantissa: int = 6, silent: bool = False) -> float:
        """Returns the total volume of the input CellComplex."""
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            if not silent:
                print("CellComplex.Volume - Error: The input cellComplex parameter is not a valid CellComplex. Returning None.")
            return None
        cells = CellComplex.Cells(cellComplex)
        if not isinstance(cells, list) or len(cells) == 0:
            if not silent:
                print("CellComplex.Volume - Error: Could not retrieve any Cells. Returning None.")
            return None
        total = 0.0
        for cell in cells:
            value = Cell.Volume(cell, mantissa=None, silent=True)
            if value is None:
                if not silent:
                    print("CellComplex.Volume - Error: Could not compute the volume of one or more Cells. Returning None.")
                return None
            total += float(value)
        if mantissa is None:
            return float(total)
        try:
            return round(total, int(mantissa))
        except Exception:
            if not silent:
                print("CellComplex.Volume - Error: The input mantissa parameter is invalid. Returning None.")
            return None
    
    @staticmethod
    def Voronoi(vertices: list = None, cell= None, tolerance: float = 0.0001):
        """
        Partitions the input cell based on the Voronoi method. See https://en.wikipedia.org/wiki/Voronoi_diagram.

        Parameters
        ----------
        vertices: list , optional 
            The input list of vertices to use for voronoi partitioning. If set to None, the algorithm uses the vertices of the input cell parameter.
            if both are set to none, a unit cube centered around the origin is used.
        cell : topologic_core.Cell , optional
            The input bounding cell. If set to None, an axes-aligned bounding cell is created from the list of vertices. Default is None.
        tolerance : float , optional
            the desired tolerance. Default is 0.0001.
        

        Returns
        -------
        topologic_core.CellComplex
            The created voronoi cellComplex.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from scipy.spatial import Voronoi as SCIVoronoi
        import numpy as np

        def fracture_with_voronoi(points):
            # Compute Voronoi tessellation
            vor = SCIVoronoi(points)
            verts = []
            faces = []
            for v in vor.vertices:
                verts.append(Vertex.ByCoordinates(list(v)))
            for region in vor.ridge_vertices:
                temp_list = []
                if -1 not in region and len(region) > 0:
                    for item in region:
                        temp_list.append(verts[item])
                    f = Face.ByVertices(temp_list)
                    if Topology.IsInstance(f, "Face"):
                        faces.append(f)
            if len(faces) < 1:
                return None
            return Cluster.ByTopologies(faces)
        
        if cell == None:
            if not isinstance(vertices, list):
                cell = Cell.Prism(uSides=2, vSides=2, wSides=2)
                vertices = Topology.Vertices(cell)
                vertices.append(Vertex.Origin())
            else:
                vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
                if len(vertices) < 1:
                    print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                    return None
                cell = Topology.BoundingBox(Cluster.ByTopologies(vertices))
        if not isinstance(vertices, list):
            if not Topology.IsInstance(cell, "Cell"):
                cell = Cell.Prism()
                vertices = Topology.Vertices(cell)
            else:
                vertices = Topology.Vertices(cell)
        else:
            vertices += Topology.Vertices(cell)
        vertices = [v for v in vertices if (Vertex.IsInternal(v, cell) or not Vertex.Index(v, Topology.Vertices(cell), tolerance=tolerance) == None)]
        if len(vertices) < 1:
            print("CellComplex.Voronoi - Error: The input vertices parameter does not contain any vertices that are inside the input cell parameter. Returning None.")
            return None
        voronoi_points = np.array([Vertex.Coordinates(v) for v in vertices])
        cluster = fracture_with_voronoi(voronoi_points)
        if cluster == None:
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        cellComplex = Topology.Slice(cell, cluster)
        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Voronoi - Error: the operation failed. Returning None.")
            return None
        return cellComplex
    
    @staticmethod
    def Wires(cellComplex) -> list:
        """
        Returns the wires of the input cellComplex.

        Parameters
        ----------
        cellComplex : topologic_core.CellComplex
            The input cellComplex.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cellComplex, "CellComplex"):
            print("CellComplex.Wires - Error: The input cellcomplex parameter is not a valid topologic cellcomplex. Returning None.")
            return None
        wires = []
        # _ = cellComplex.Wires(None, wires) # H to Core
        try:
            _ = Core.InstanceCall(cellComplex, "Wires", None, wires)
        except Exception:
            wires = None
        return wires

