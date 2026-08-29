from __future__ import annotations

import math

from dataclasses import dataclass, field
from .topology import Topology, _downward_wrappers
from .cell import Cell
from .wire import Wire
from .cluster import Cluster

try:
    from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
    from OCC.Core.TopAbs import (
        TopAbs_VERTEX,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_CompSolid, topods

except Exception:  # pragma: no cover - allows import without PythonOCC
    BOPAlgo_MakerVolume = None
    BRepAlgoAPI_Fuse = None
    BRepBuilderAPI_Transform = None
    BRepPrimAPI_MakeTorus = None
    gp_Ax1 = None
    gp_Dir = None
    gp_Pnt = None
    gp_Trsf = None

    TopAbs_VERTEX = None
    TopAbs_EDGE = None
    TopAbs_FACE = None
    TopAbs_SHELL = None
    TopAbs_SOLID = None

    TopExp_Explorer = None
    TopTools_ListOfShape = None
    BRep_Builder = None
    TopoDS_CompSolid = None
    topods = None


def _iter_subshapes(shape, shape_type):
    if TopExp_Explorer is None or shape_type is None:
        return []
    result = []
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def _as_compsolid(shape):
    """
    OCCT volume builders may yield a plain COMPOUND
    containing the resulting Solids rather than a COMPSOLID, even when every
    Solid shares faces with its neighbours. topologic_core's CellComplex is
    specifically a COMPSOLID, so rebuild one explicitly from the Solid
    sub-shapes whenever the builder didn't already produce one.
    """
    if shape is None:
        return None
    try:
        from OCC.Core.TopAbs import TopAbs_COMPSOLID
        if shape.ShapeType() == TopAbs_COMPSOLID:
            return shape
    except Exception:
        pass

    solids = _iter_subshapes(shape, TopAbs_SOLID)
    if not solids or BRep_Builder is None:
        return None
    try:
        builder = BRep_Builder()
        compsolid = TopoDS_CompSolid()
        builder.MakeCompSolid(compsolid)
        for solid in solids:
            builder.Add(compsolid, topods.Solid(solid))
        return compsolid
    except Exception:
        return None


def _shape_same(a, b):
    """OCCT shape equality. Two TopoDS_Shape objects are the same
    topology iff IsSame returns True (coincident + same orientation
    class). Used for shared/non-manifold face detection where
    BOPAlgo_MakerVolume emits distinct Python objects for the same
    geometric face in adjacent cells.
    """
    if a is b:
        return True
    try:
        return bool(a.IsSame(b))
    except Exception:
        return False


def _is_null_shape(shape):
    if shape is None:
        return True
    try:
        return bool(shape.IsNull())
    except Exception:
        return True


def _drop_open_shells(shape):
    """
    Rebuild a single SOLID dropping open (zero-enclosed-volume) shells: closed
    lofts can carry a degenerate open shell that
    inflates the Shell count (2 vs core's 1). A genuine cavity shell encloses
    non-zero volume and is kept; compsolids are untouched.
    """
    if shape is None or _is_null_shape(shape):
        return shape
    try:
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL
        if shape.ShapeType() != TopAbs_SOLID:
            return shape
    except Exception:
        return shape
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import TopoDS_Solid
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
    except Exception:
        return shape

    shells = []
    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while explorer.More():
        shell = explorer.Current()
        explorer.Next()
        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(shell, props)
        except Exception:
            continue
        if abs(props.Mass()) > 1e-9:
            shells.append(shell)

    if not shells:
        return shape
    # Count shells actually present to know whether anything was dropped.
    total = 0
    exp2 = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp2.More():
        total += 1
        exp2.Next()
    if total == len(shells):
        return shape

    try:
        builder = BRep_Builder()
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        for shell in shells:
            builder.Add(solid, shell)
        return solid
    except Exception:
        return shape


@dataclass(eq=False)
class CellComplex(Topology):
    cells: list = field(default_factory=list)

    @staticmethod
    def _build_from_shapes(shapes, tolerance=0.0001):
        """
        True non-manifold CellComplex construction: BOPAlgo_MakerVolume partitions the
        Face/Shell/Solid boundary soup into every resulting Solid with shared internal faces
        """
        shapes = [s for s in (shapes or []) if not _is_null_shape(s)]
        if not shapes or BOPAlgo_MakerVolume is None:
            return None

        try:
            args = TopTools_ListOfShape()
            for shape in shapes:
                args.Append(shape)
            maker = BOPAlgo_MakerVolume()
            maker.SetArguments(args)
            maker.SetIntersect(True)
            if tolerance:
                try:
                    maker.SetFuzzyValue(float(tolerance))
                except Exception:
                    pass
            maker.Perform()
            if hasattr(maker, "HasErrors") and maker.HasErrors():
                return None
            result_shape = maker.Shape()
        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None

        solid_count = len(_iter_subshapes(result_shape, TopAbs_SOLID))
        if solid_count >= 2:
            compsolid = _as_compsolid(result_shape)
            if compsolid is not None:
                result_shape = compsolid
        else:
            # A single-solid loft/partition can carry an open, zero-volume
            # shell. topologic_core
            # builds a single clean shell; our BOPAlgo_MakerVolume emits a
            # spurious extra shell on such closed lofts, inflating the Shell
            # count (2 vs 1). Drop zero-volume shells -- a genuine void
            # (cavity) shell encloses non-zero volume and is preserved.
            result_shape = _drop_open_shells(result_shape)

        result = Topology.ByOcctShape(result_shape)
        if isinstance(result, CellComplex):
            return result
        if isinstance(result, Cell):
            # The faces/cells only bounded a single solid: still a valid
            # (single-cell) CellComplex per topologic_core semantics.
            return CellComplex(shape=result_shape, cells=[result])
        return None

    @staticmethod
    def ByCells(cells, tolerance=0.0001):
        cells = [cell for cell in (cells or []) if isinstance(cell, Cell)]
        if len(cells) < 1:
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None
        if tolerance <= 0.0:
            return None

        shapes = []
        for cell in cells:
            shape = getattr(cell, "shape", None)
            if _is_null_shape(shape):
                return None
            shapes.append(shape)

        # A single Cell is still a valid single-cell CellComplex wrapper.
        if len(shapes) == 1:
            return CellComplex(shape=shapes[0], cells=[cells[0]])

        result = CellComplex._build_from_shapes(shapes, tolerance)
        if not isinstance(result, CellComplex):
            # Do not return a shapeless pseudo-CellComplex. A failed OCCT
            # construction must remain visible to the algorithm layer.
            return None
        return result

    @staticmethod
    def ByFaces(faces, tolerance=0.0001, copyAttributes=False):
        from .face import Face

        faces = [face for face in (faces or []) if isinstance(face, Face)]
        if len(faces) < 1:
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None
        if tolerance <= 0.0:
            return None

        shapes = []
        for face in faces:
            shape = getattr(face, "shape", None)
            if _is_null_shape(shape):
                return None
            shapes.append(shape)

        result = CellComplex._build_from_shapes(shapes, tolerance)
        if isinstance(result, CellComplex):
            return result

        # A closed Face set may describe exactly one Cell. Preserve its exact
        # Faces/curves and wrap that Cell as a single-cell CellComplex.
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if not isinstance(cell, Cell):
            return None
        return CellComplex.ByCells([cell], tolerance=tolerance)

    @staticmethod
    def ByWires(wires, tolerance: float = 0.0001):
        """
        Create a curve-preserving CellComplex by lofting between consecutive
        closed section Wires.

        Each consecutive pair of Wires creates one native OCCT Cell using
        Cell.ByWires. The resulting Cells are then assembled into a true
        non-manifold CellComplex so intermediate sections become shared Faces.

        Parameters
        ----------
        wires : list
            Ordered closed section Wires. At least two valid Wires are required.
            Corresponding Wires must contain the same number of Edges.
        tolerance : float , optional
            Geometric tolerance. Default is 0.0001.

        Returns
        -------
        CellComplex
            The resulting curve-preserving CellComplex, or None on failure.
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

        if tolerance <= 0.0:
            return None

        cells = []

        for i in range(len(wire_list) - 1):
            cell = Cell.ByWires(
                [wire_list[i], wire_list[i + 1]],
                tolerance=tolerance
            )
            if not isinstance(cell, Cell):
                return None
            cells.append(cell)

        return CellComplex.ByCells(cells, tolerance=tolerance)

    @staticmethod
    def ByTorus(
        majorRadius=0.5,
        minorRadius=0.125,
        uSides=16,
        tolerance=0.0001,
        silent=False,
    ):
        """Build an exact toroidal CellComplex subdivided around its major circle."""
        try:
            majorRadius = float(majorRadius)
            minorRadius = float(minorRadius)
            uSides = int(uSides)
            tolerance = float(tolerance)
        except Exception:
            return None

        if majorRadius <= tolerance or minorRadius <= tolerance:
            return None
        if minorRadius >= majorRadius or uSides < 3:
            return None
        if (
            BRepPrimAPI_MakeTorus is None
            or BRepBuilderAPI_Transform is None
            or gp_Ax1 is None
            or gp_Pnt is None
            or gp_Dir is None
            or gp_Trsf is None
        ):
            return None

        step = 2.0 * math.pi / float(uSides)

        try:
            # OCCT's torus u-parameter is rotation around +Z. Supplying the
            # angular overload creates one exact toroidal sector bounded by
            # planar circular end Faces.
            maker = BRepPrimAPI_MakeTorus(majorRadius, minorRadius, step)
            sector_shape = maker.Shape()
        except Exception:
            return None

        if _is_null_shape(sector_shape):
            return None

        axis = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
        cells = []

        for i in range(uSides):
            try:
                if i == 0:
                    shape = sector_shape
                else:
                    trsf = gp_Trsf()
                    trsf.SetRotation(axis, step * float(i))
                    shape = BRepBuilderAPI_Transform(sector_shape, trsf, True).Shape()
            except Exception:
                return None

            if _is_null_shape(shape):
                return None

            cell = Topology.ByOcctShape(shape)
            if not isinstance(cell, Cell):
                return None

            cells.append(cell)

        result = CellComplex.ByCells(cells, tolerance=tolerance)
        if not isinstance(result, CellComplex):
            return None

        return result

    def Cells(self, hostTopology=None, cells=None):
        result = []

        if not _is_null_shape(getattr(self, "shape", None)):
            try:
                # A single-cell CellComplex can currently carry a Solid directly.
                if self.shape.ShapeType() == TopAbs_SOLID:
                    cell = Topology.ByOcctShape(self.shape)
                    if isinstance(cell, Cell):
                        result = [cell]
                else:
                    result = _downward_wrappers(
                        self,
                        TopAbs_SOLID
                    )
            except Exception:
                result = []

        # Fallback for shapeless/fallback CellComplex instances.
        if not result:
            result = list(getattr(self, "cells", []) or [])

        if cells is not None:
            cells.extend(result)
            return 0

        return result


    def Shells(self, hostTopology=None, shells=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_SHELL
            )
        else:
            result = []
            for cell in getattr(self, "cells", []) or []:
                result.extend(cell.Shells())

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
            for cell in getattr(self, "cells", []) or []:
                result.extend(cell.Faces())

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
            for cell in getattr(self, "cells", []) or []:
                result.extend(cell.Edges())

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
            for cell in getattr(self, "cells", []) or []:
                result.extend(cell.Vertices())

        if vertices is not None:
            vertices.extend(result)
            return 0

        return result

    def CellComplexes(self, hostTopology=None, cellComplexes=None):
        result = [self]
        if cellComplexes is not None:
            cellComplexes.extend(result)
            return 0
        return result

    def ExternalBoundary(self):
        """
        Returns the outer Shell bounding the union of all of this
        CellComplex's Cells (i.e. its Cells fused together, with internal
        non-manifold boundaries removed).
        """
        cell_shapes = [c.shape for c in self.Cells() if not _is_null_shape(getattr(c, "shape", None))]
        if not cell_shapes or BRepAlgoAPI_Fuse is None:
            return None

        try:
            fused = cell_shapes[0]
            for shape in cell_shapes[1:]:
                op = BRepAlgoAPI_Fuse(fused, shape)
                op.Build()
                if not op.IsDone():
                    return None
                fused = op.Shape()
        except Exception:
            return None

        if _is_null_shape(fused):
            return None

        try:
            explorer = TopExp_Explorer(fused, TopAbs_SHELL)
            if explorer.More():
                outer_shell_shape = explorer.Current()
                return Topology.ByOcctShape(outer_shell_shape)
        except Exception:
            pass
        return Topology.ByOcctShape(fused)

    def InternalBoundaries(self, faces=None):
        result = self.NonManifoldFaces()
        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def NonManifoldFaces(self, faces=None):
        """
        Returns the Faces shared by two or more of this CellComplex's Cells
        (the non-manifold internal boundaries). Identity is decided by OCCT
        shape equality (TopoDS_Shape.IsSame), NOT Python object/identity
        hashing: BOPAlgo_MakerVolume rebuilds the shared face as a
        distinct TopoDS_Face object inside each Cell, so two copies of the
        same geometric face have different Python identities and hashes yet
        are the same topology. IsSame is the only correct test here.
        """
        per_cell = []
        for cell in self.Cells():
            seen = []
            for face in cell.Faces():
                shape = getattr(face, "shape", None)
                if _is_null_shape(shape):
                    continue
                # de-dupe within a single cell first
                if not any(_shape_same(shape, s) for s, _ in seen):
                    seen.append((shape, face))
            if seen:
                per_cell.append(seen)

        result = []
        used = set()
        for i in range(len(per_cell)):
            for j in range(i + 1, len(per_cell)):
                for s_i, f_i in per_cell[i]:
                    for s_j, f_j in per_cell[j]:
                        if _shape_same(s_i, s_j):
                            key = id(f_i)
                            if key not in used:
                                used.add(key)
                                result.append(f_i)
        if faces is not None:
            faces.extend(result)
            return 0
        return result


# ---------------------------------------------------------------------------
# Explicit unsupported CellComplex API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _cell_complex_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"CellComplex.{name}", return_value)
    return _method


CellComplex.ByCellsCluster = staticmethod(
    lambda cluster, transferDictionaries=False, tolerance=0.0001, silent=False: (
        CellComplex.ByCells(
            (Topology.Cells(cluster) if isinstance(cluster, Cluster) else []),
            tolerance=tolerance,
        )
    )
)
# CellComplex.ExternalBoundary and CellComplex.NonManifoldFaces are implemented above.
