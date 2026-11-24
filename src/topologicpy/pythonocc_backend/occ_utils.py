from __future__ import annotations

try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
    )
except Exception:  # pragma: no cover
    gp_Pnt = None
    BRepBuilderAPI_MakeVertex = None
    BRepBuilderAPI_MakeEdge = None
    BRepBuilderAPI_MakeWire = None
    BRepBuilderAPI_MakeFace = None


def make_occ_vertex(x: float, y: float, z: float):
    if gp_Pnt is None or BRepBuilderAPI_MakeVertex is None:
        return None
    try:
        return BRepBuilderAPI_MakeVertex(gp_Pnt(float(x), float(y), float(z))).Vertex()
    except Exception:
        return None


def make_occ_edge(start, end):
    if gp_Pnt is None or BRepBuilderAPI_MakeEdge is None:
        return None
    try:
        return BRepBuilderAPI_MakeEdge(gp_Pnt(start.x, start.y, start.z), gp_Pnt(end.x, end.y, end.z)).Edge()
    except Exception:
        return None


def _positive_tolerance(value):
    import math
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _native_subshapes(shape, kind):
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    seen = TopTools_IndexedMapOfShape()
    result = []
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        sub = explorer.Current()
        if not seen.Contains(sub):
            seen.Add(sub)
            result.append(sub)
        explorer.Next()
    return result


def _copy_with_tolerance(shape, tolerance, *, vertices_only=False):
    """Raise tolerances only on a private copy, preserving exact 3D curves."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopoDS import topods
    copied = BRepBuilderAPI_Copy(shape, True, False).Shape()
    builder = BRep_Builder()
    if not vertices_only:
        for edge in _native_subshapes(copied, TopAbs_EDGE):
            builder.UpdateEdge(topods.Edge(edge), tolerance)
    for vertex in _native_subshapes(copied, TopAbs_VERTEX):
        builder.UpdateVertex(topods.Vertex(vertex), tolerance)
    return copied


def make_occ_wire(edges: list, tolerance=0.0001):
    tolerance = _positive_tolerance(tolerance)
    if BRepBuilderAPI_MakeWire is None or tolerance is None or not edges:
        return None
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods
        shapes = [getattr(edge, "shape", None) for edge in edges]
        if any(s is None or s.IsNull() for s in shapes):
            return None
        for repair in (False, True):
            maker = BRepBuilderAPI_MakeWire()
            for shape in shapes:
                if repair:
                    shape = _copy_with_tolerance(shape, tolerance, vertices_only=True)
                maker.Add(topods.Edge(shape))
                if not maker.IsDone():
                    break
            else:
                wire = maker.Wire()
                explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                count = 0
                while explorer.More():
                    count += 1
                    explorer.Next()
                if count == len(shapes):
                    return wire
    except Exception:
        pass
    return None


def make_occ_face(wire, tolerance=0.0001, only_plane=False):
    return make_occ_face_from_wire(getattr(wire, "shape", None), tolerance, only_plane)


def make_occ_face_from_wire(shape, tolerance=0.0001, only_plane=False):
    """Keep exact construction first; recover near-planar wires within tolerance.

    No triangulation or coordinate projection. The original input is never
    modified by the tolerance-recovery path. FindSurface independently bounds
    the deviation of the full 3D curves, not just their endpoints.
    """
    tolerance = _positive_tolerance(tolerance)
    if BRepBuilderAPI_MakeFace is None or tolerance is None or shape is None:
        return None
    try:
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepLib import BRepLib_FindSurface
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        if shape.IsNull():
            return None
        wire = topods.Wire(shape)
        maker = BRepBuilderAPI_MakeFace(wire, only_plane)
        exact = maker.Face() if maker.IsDone() else None
        if exact is not None and BRepCheck_Analyzer(exact).IsValid():
            return exact
        finder = BRepLib_FindSurface(wire, tolerance, True)
        if finder.Found():
            reached = float(finder.ToleranceReached())
            if 0.0 <= reached <= tolerance:
                # Allow enough tolerance for native p-curves, but never more
                # than the requested modelling tolerance on this recovery path.
                applied = min(tolerance, max(1.e-7, 2.0 * reached))
                copied = topods.Wire(_copy_with_tolerance(wire, applied))
                repaired = BRepBuilderAPI_MakeFace(copied, True)
                if repaired.IsDone():
                    candidate = repaired.Face()
                    if BRepCheck_Analyzer(candidate).IsValid():
                        return candidate
        # Preserve the pre-existing face-healing path for exact curved wires.
        return exact
    except Exception:
        return None


def make_occ_shell(faces: list, tolerance=0.0001):
    """Sew all faces at the caller's tolerance; never discard a face."""
    tolerance = _positive_tolerance(tolerance)
    if not faces or tolerance is None:
        return None
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Shell, topods
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        occ_faces = []
        for face in faces:
            shape = getattr(face, "shape", None)
            if shape is None or shape.IsNull():
                return None
            shape = topods.Face(shape)
            if not BRepCheck_Analyzer(shape).IsValid():
                return None
            occ_faces.append(shape)
        sewer = BRepBuilderAPI_Sewing(tolerance)
        for face in occ_faces:
            sewer.Add(face)
        sewer.Perform()
        sewn = sewer.SewedShape()
        if sewn is not None and not sewn.IsNull():
            if len(_native_subshapes(sewn, TopAbs_FACE)) == len(occ_faces):
                if sewn.ShapeType() == TopAbs_SHELL:
                    return topods.Shell(sewn)
                shells = _native_subshapes(sewn, TopAbs_SHELL)
                if len(shells) == 1 and len(_native_subshapes(shells[0], TopAbs_FACE)) == len(occ_faces):
                    return topods.Shell(shells[0])
                if shells:
                    # Preserve disconnected shells and any remaining faces.
                    # Solid construction independently checks closure and validity.
                    return sewn
        # Keep the historical open-shell fallback for visualization/traversal.
        # It cannot be promoted to a Cell unless genuinely closed and valid.
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        for face in occ_faces:
            builder.Add(shell, face)
        return shell
    except Exception:
        return None


def is_valid_occ_solid(shape):
    """Validate native type, every boundary shell and the entire solid."""
    try:
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        if shape is None or shape.IsNull() or shape.ShapeType() != TopAbs_SOLID:
            return False
        shells = _native_subshapes(shape, TopAbs_SHELL)
        return bool(shells) and all(BRep_Tool.IsClosed(s) for s in shells) and bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception:
        return False


def _orient_valid_solid(solid):
    try:
        from OCC.Core.BRepLib import breplib
        from OCC.Core.TopoDS import topods
        solid = topods.Solid(solid)
        if not breplib.OrientClosedSolid(solid):
            return None
        return solid if is_valid_occ_solid(solid) else None
    except Exception:
        return None


def _build_solid_from_shells(shape):
    """Build one solid only if all shells form a valid closed boundary.

    An outer shell plus legitimate cavity shells is supported. Disconnected
    exterior shells or a leaky shell must not be mislabeled as a Cell.
    """
    try:
        from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE
        from OCC.Core.TopoDS import TopoDS_Solid, topods
        from OCC.Core.BRep import BRep_Builder, BRep_Tool
        shells = _native_subshapes(shape, TopAbs_SHELL)
        if len(shells) < 2 or not all(BRep_Tool.IsClosed(s) for s in shells):
            return None
        builder = BRep_Builder()
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        for shell in shells:
            builder.Add(solid, topods.Shell(shell))
        if len(_native_subshapes(solid, TopAbs_FACE)) != len(_native_subshapes(shape, TopAbs_FACE)):
            return None
        return _orient_valid_solid(solid)
    except Exception:
        return None


def make_occ_cell(shell, tolerance=0.0001):
    tolerance = _positive_tolerance(tolerance)
    if shell is None or tolerance is None:
        return None
    try:
        from OCC.Core.TopAbs import TopAbs_SHELL
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        shape = getattr(shell, "shape", None)
        if shape is None or shape.IsNull():
            shape = make_occ_shell(getattr(shell, "faces", None), tolerance=tolerance)
        if shape is None or shape.IsNull():
            return None
        if shape.ShapeType() != TopAbs_SHELL:
            return _build_solid_from_shells(shape)
        if not BRep_Tool.IsClosed(shape):
            return None
        maker = BRepBuilderAPI_MakeSolid(topods.Shell(shape))
        if not maker.IsDone():
            return None
        return _orient_valid_solid(maker.Solid())
    except Exception:
        return None
