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


def make_occ_wire(edges: list):
    if BRepBuilderAPI_MakeWire is None:
        return None
    try:
        maker = BRepBuilderAPI_MakeWire()
        for edge in edges:
            if getattr(edge, "shape", None) is not None:
                maker.Add(edge.shape)
        if maker.IsDone():
            return maker.Wire()
    except Exception:
        pass
    return None


def make_occ_face(wire):
    if BRepBuilderAPI_MakeFace is None:
        return None
    try:
        if getattr(wire, "shape", None) is not None:
            maker = BRepBuilderAPI_MakeFace(wire.shape)
            if maker.IsDone():
                return maker.Face()
    except Exception:
        pass
    return None


def make_occ_shell(faces: list):
    if not faces:
        return None
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Shell, topods
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
    except Exception:
        return None

    builder = BRep_Builder()
    occ_shell = TopoDS_Shell()
    builder.MakeShell(occ_shell)
    added = 0

    for face in faces:
        occ_face = getattr(face, "shape", None)
        if occ_face is None:
            continue
        try:
            occ_face = topods.Face(occ_face)
        except Exception:
            continue
        try:
            builder.Add(occ_shell, occ_face)
            added += 1
        except Exception:
            continue

    if added == 0:
        return None

    try:
        analyzer = BRepCheck_Analyzer(occ_shell)
        if not analyzer.IsValid():
            return occ_shell
    except Exception:
        pass

    return occ_shell


def make_occ_cell(shell):
    if shell is None:
        return None
    try:
        from OCC.Core.TopoDS import TopoDS_Shell, topods
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.BRepLib import breplib
    except Exception:
        return None

    occ_shell = getattr(shell, "shape", None)
    if occ_shell is None:
        faces = getattr(shell, "faces", None)
        occ_shell = make_occ_shell(faces) if faces else None
    if occ_shell is None:
        return None

    try:
        occ_shell = topods.Shell(occ_shell)
    except Exception:
        try:
            tmp_shell = TopoDS_Shell()
            tmp_shell = occ_shell
            occ_shell = tmp_shell
        except Exception:
            return None

    try:
        solid_maker = BRepBuilderAPI_MakeSolid(occ_shell)
        if not solid_maker.IsDone():
            return None
        occ_solid = solid_maker.Solid()
        try:
            breplib.OrientClosedSolid(occ_solid)
        except Exception:
            pass
        try:
            analyzer = BRepCheck_Analyzer(occ_solid)
            if not analyzer.IsValid():
                return occ_solid
        except Exception:
            pass
        return occ_solid
    except Exception:
        return None
