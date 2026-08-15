from __future__ import annotations

from dataclasses import dataclass, field
from .topology import Topology, _downward_wrappers
from .cell import Cell
from .cluster import Cluster
from .helpers import unique_by_uuid

try:
    from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder, BOPAlgo_MakerVolume
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
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
    BOPAlgo_CellsBuilder = None
    BOPAlgo_MakerVolume = None
    BRepAlgoAPI_Fuse = None
    TopAbs_VERTEX = TopAbs_EDGE, TopAbs_FACE = TopAbs_SHELL = TopAbs_SOLID = None
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
    BOPAlgo_CellsBuilder.MakeContainers() may yield a plain COMPOUND
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
    lofts (e.g. CellComplex.Torus) can carry a degenerate open shell that
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
        (BOPAlgo_CellsBuilder wrong here: fed pure Faces it builds 2D face-cells).
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
            # shell (e.g. CellComplex.Torus' made-by-Spin solid). topologic_core
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
        cells = [c for c in (cells or []) if isinstance(c, Cell)]
        shapes = [c.shape for c in cells if not _is_null_shape(getattr(c, "shape", None))]
        if len(shapes) < 1:
            return None
        if len(shapes) == 1:
            return CellComplex(shape=shapes[0], cells=cells)
        result = CellComplex._build_from_shapes(shapes, tolerance)
        if result is None:
            return CellComplex(shape=None, cells=cells)
        return result

    @staticmethod
    def ByFaces(faces, tolerance=0.0001, copyAttributes=False):
        from .face import Face
        faces = [f for f in (faces or []) if isinstance(f, Face)]
        shapes = [f.shape for f in faces if not _is_null_shape(getattr(f, "shape", None))]
        if len(shapes) < 1:
            return None
        result = CellComplex._build_from_shapes(shapes, tolerance)
        if result is not None:
            return result
        # BOPAlgo_MakerVolume found no fallback volume (e.g. it errored on a
        # face soup that a simpler single-cell sewing pass would tolerate).
        # Fall back to the single-cell path, same as ByCells does when its
        # own _build_from_shapes call fails.
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if cell is None:
            return None
        return CellComplex.ByCells([cell], tolerance)

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
