#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PythonOCC backend Topology class.

This file is intended as a drop-in replacement for:

    pythonocc_backend/topology.py

It implements:
    - Topology.IsInstance
    - Topology.TypeAsString
    - Topology.Dictionary
    - Topology.SetDictionary
    - Topology.Vertices
    - Topology.Edges
    - Topology.Wires
    - Topology.Faces
    - Topology.Shells
    - Topology.ByOcctShape
    - Topology.Merge

The implementation is deliberately defensive. It works with the lightweight
Python backend wrapper classes already used in the smoke tests, while using
real PythonOCC/OCCT operations whenever an OCCT shape is available.
"""

from __future__ import annotations

# REVISION: 2026-08-15 Boolean parity fix 004

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .helpers import new_uuid as _new_uuid, distance3, vertex_key

from .attribute_manager import AttributeManager


# -----------------------------------------------------------------------------
# Optional PythonOCC imports
# -----------------------------------------------------------------------------

try:
    from OCC.Core.TopAbs import (
        TopAbs_VERTEX,
        TopAbs_EDGE,
        TopAbs_WIRE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_COMPSOLID,
        TopAbs_COMPOUND,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape, topods
    # pythonocc-core 7.9.0 removed the deprecated topods_Vertex/topods_Edge/...
    # free functions entirely (they raise ImportError, not just a
    # DeprecationWarning as in 7.7.1) -- use the topods module-proxy form and
    # keep these names as aliases so every existing call site below is
    # unaffected. Do not revert to the free-function imports: a single
    # missing name here previously poisoned this entire try/except block,
    # silently nulling out ~40 unrelated OCC symbols (BRepBuilderAPI_Transform,
    # BOPAlgo_CellsBuilder, GProp_GProps, etc.) and breaking Translate/Rotate/
    # Scale/boolean ops/etc. across the whole backend.
    topods_Vertex = topods.Vertex
    topods_Edge = topods.Edge
    topods_Wire = topods.Wire
    topods_Face = topods.Face
    topods_Shell = topods.Shell
    topods_Solid = topods.Solid
    topods_Compound = topods.Compound
    from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
    from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder, BOPAlgo_Splitter
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.gp import gp_Trsf, gp_Pnt, gp_Dir, gp_Ax1, gp_Vec, gp_GTrsf, gp_Mat, gp_XYZ
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform, BRepBuilderAPI_Copy
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepAlgoAPI import (
        BRepAlgoAPI_Fuse,
        BRepAlgoAPI_Cut,
        BRepAlgoAPI_Common,
        BRepAlgoAPI_Section,
    )
    from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN as _TopAbs_IN, TopAbs_ON as _TopAbs_ON
except Exception:  # pragma: no cover - allows import without PythonOCC
    TopAbs_VERTEX = TopAbs_EDGE = TopAbs_WIRE = TopAbs_FACE = None
    TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPSOLID = TopAbs_COMPOUND = None
    TopExp_Explorer = None
    TopoDS_Shape = None
    topods_Vertex = topods_Edge = topods_Wire = topods_Face = None
    topods_Shell = topods_Solid = topods_Compound = None
    TopTools_ListOfShape = None
    TopTools_ListIteratorOfListOfShape = None
    BOPAlgo_CellsBuilder = None
    BOPAlgo_Splitter = None
    BRepCheck_Analyzer = None
    ShapeFix_Shape = None
    gp_Trsf = gp_Pnt = gp_Dir = gp_Ax1 = gp_Vec = gp_GTrsf = gp_Mat = gp_XYZ = None
    BRepBuilderAPI_Transform = BRepBuilderAPI_GTransform = BRepBuilderAPI_Copy = None
    GProp_GProps = None
    brepgprop = None
    BRepAlgoAPI_Fuse = BRepAlgoAPI_Cut = BRepAlgoAPI_Common = BRepAlgoAPI_Section = None
    ShapeUpgrade_UnifySameDomain = None
    BRepExtrema_DistShapeShape = None
    BRepClass3d_SolidClassifier = None
    _TopAbs_IN = _TopAbs_ON = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _not_implemented(name: str):
    print(f"{name} - Not implemented.")
    return None


def _is_null_shape(shape: Any) -> bool:
    if shape is None:
        return True
    if hasattr(shape, "IsNull"):
        try:
            return bool(shape.IsNull())
        except Exception:
            return False
    return False


def _shape_from_topology(topology: Any) -> Any:
    if topology is None:
        return None

    if isinstance(topology, dict):
        return topology.get("shape", None)

    if hasattr(topology, "shape"):
        return getattr(topology, "shape")

    if hasattr(topology, "GetOcctShape"):
        try:
            return topology.GetOcctShape()
        except Exception:
            return None

    if TopoDS_Shape is not None and isinstance(topology, TopoDS_Shape):
        return topology

    return None


def _safe_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _topology_type_name(topology: Any) -> Optional[str]:
    if topology is None:
        return None

    # Graph is not an OCCT topology, but TopologicPy treats it as a recognised
    # object in IsInstance.
    if topology.__class__.__name__ == "Graph":
        return "Graph"

    name = topology.__class__.__name__
    known = {
        "Vertex",
        "Edge",
        "Wire",
        "Face",
        "Shell",
        "Cell",
        "CellComplex",
        "Cluster",
        "Aperture",
        "Context",
        "Graph",
    }
    if name in known:
        return name

    shape = _shape_from_topology(topology)
    if _is_null_shape(shape) or shape is None or not hasattr(shape, "ShapeType"):
        return None

    try:
        st = shape.ShapeType()
    except Exception:
        return None

    if st == TopAbs_VERTEX:
        return "Vertex"
    if st == TopAbs_EDGE:
        return "Edge"
    if st == TopAbs_WIRE:
        return "Wire"
    if st == TopAbs_FACE:
        return "Face"
    if st == TopAbs_SHELL:
        return "Shell"
    if st == TopAbs_SOLID:
        return "Cell"
    if st == TopAbs_COMPSOLID:
        return "CellComplex"
    if st == TopAbs_COMPOUND:
        return "Cluster"

    return None


# -----------------------------------------------------------------------------
# Numeric type IDs
# -----------------------------------------------------------------------------
#
# These mirror the values hardcoded in topologicpy.Topology.TypeID (see
# src/topologicpy/Topology.py) which in turn mirror topologic_core's bitmask
# type ids. Do not change these without also checking that file.

_TYPE_IDS = {
    "Vertex": 1,
    "Edge": 2,
    "Wire": 4,
    "Face": 8,
    "Shell": 16,
    "Cell": 32,
    "CellComplex": 64,
    "Cluster": 128,
    "Aperture": 256,
    "Context": 512,
    "Dictionary": 1024,
    "Graph": 2048,
    "Topology": 4096,
}

# Dimensional ordering (not the bit-flag type IDs above) used to decide
# whether a Vertices/Edges/Wires/Faces/Shells/Cells/CellComplexes call means
# "my own substructure" (target rank <= self rank) or "adjacency/host-scoped
# super-topology search" (target rank > self rank) -- see
# Topology._dispatch_subtopologies.
_TYPE_RANK = {
    "Vertex": 0,
    "Edge": 1,
    "Wire": 2,
    "Face": 3,
    "Shell": 4,
    "Cell": 5,
    "CellComplex": 6,
}


def _type_id_for(topology: Any) -> Optional[int]:
    """
    Resolve the numeric topologic_core-style type id for a topology/graph.

    Uses the same resolution logic as ``_topology_type_name`` (class-name first,
    OCCT ShapeType fallback), so CellComplex (TopAbs_COMPSOLID) is distinguished
    from Cell (TopAbs_SOLID) correctly.
    """
    name = _topology_type_name(topology)
    if name is None:
        return None
    return _TYPE_IDS.get(name)


def _is_topology_like(topology: Any) -> bool:
    t = _topology_type_name(topology)
    return t in {
        "Vertex",
        "Edge",
        "Wire",
        "Face",
        "Shell",
        "Cell",
        "CellComplex",
        "Cluster",
        "Aperture",
        "Context",
    }


def _make_toptools_list(shapes: Iterable[Any]) -> Any:
    if TopTools_ListOfShape is None:
        raise ImportError("PythonOCC TopTools_ListOfShape is not available.")

    result = TopTools_ListOfShape()
    for shape in shapes:
        if not _is_null_shape(shape):
            result.Append(shape)
    return result


def _toptools_list_to_pylist(toptools_list: Any) -> list:
    """
    TopTools_ListOfShape is not directly Python-iterable in this pythonocc
    build -- use the standard OCCT C++ iterator pattern instead.
    """
    if toptools_list is None or TopTools_ListIteratorOfListOfShape is None:
        return []
    result = []
    it = TopTools_ListIteratorOfListOfShape(toptools_list)
    while it.More():
        result.append(it.Value())
        it.Next()
    return result


def _iter_occ_subshapes(shape: Any, shape_type: Any) -> list:
    if _is_null_shape(shape) or TopExp_Explorer is None or shape_type is None:
        return []

    result = []
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result

def _iter_occ_subshapes_unique(shape: Any, shape_type: Any) -> list:
    """
    Returns the unique OCCT subshapes of the requested type.

    Uniqueness is determined using the Python hash of each TopoDS_Shape.
    This provides substantially better performance than repeatedly calling
    TopTools_MapOfShape from Python while preserving OCCT shape identity
    semantics for the topology traversal used by the backend.

    Parameters
    ----------
    shape : OCC.Core.TopoDS.TopoDS_Shape
        The input OCCT shape.
    shape_type : OCC.Core.TopAbs.TopAbs_ShapeEnum
        The requested OCCT subshape type.

    Returns
    -------
    list
        The unique OCCT subshapes.
    """
    if _is_null_shape(shape) or TopExp_Explorer is None or shape_type is None:
        return []

    result = []
    seen = set()

    try:
        explorer = TopExp_Explorer(
            shape,
            shape_type
        )

        while explorer.More():
            current = explorer.Current()

            key = hash(current)

            if key not in seen:
                seen.add(key)
                result.append(current)

            explorer.Next()

        return result

    except Exception:
        return []

def _downward_wrappers(topology: Any, shape_type: Any) -> list:
    """
    Returns backend topology wrappers for the unique OCCT subshapes of
    the requested type.
    """
    shape = _shape_from_topology(topology)

    if _is_null_shape(shape):
        return []

    result = []

    for subshape in _iter_occ_subshapes_unique(shape, shape_type):
        item = Topology.ByOcctShape(subshape)

        if item is not None:
            result.append(item)

    return result

def _deduplicate_by_identity(items: list) -> list:
    """
    Two wrapper objects extracted from the same underlying OCCT (sub-)shape
    (e.g. two Vertex wrappers built from the shared endpoint of adjacent
    edges) get distinct Python identity and distinct `_uuid`s, but should
    still be treated as the same topological entity. Prefer OCCT shape
    identity (HashCode, consistent with IsSame) over `_uuid` for that reason.
    """
    seen = set()
    result = []
    for item in items:
        shape = getattr(item, "shape", None)
        key = None
        if shape is None and hasattr(item, "IsSame"):
            shape = item
        if shape is not None and not _is_null_shape(shape):
            try:
                key = ("shape", hash(shape))
            except Exception:
                key = None
        if key is None and hasattr(item, "_uuid"):
            key = ("uuid", getattr(item, "_uuid"))
        if key is None:
            key = ("id", id(item))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _copy_common_metadata(source: Any, target: Any) -> Any:
    if source is None or target is None:
        return target

    for attr in ("dictionary", "contents", "contexts", "apertures"):
        if hasattr(source, attr) and hasattr(target, attr):
            try:
                setattr(target, attr, getattr(source, attr))
            except Exception:
                pass

    return target


def _merge_backend_dictionaries(a: Any, b: Any) -> Any:
    """
    Merge Python dictionaries and backend dictionary-like objects.

    Later values override earlier values. This function intentionally returns a
    plain Python dict if it cannot construct a backend Dictionary safely.
    """
    def to_python_dict(d):
        if d is None:
            return {}
        if isinstance(d, dict):
            return dict(d)

        # Backend Dictionary used in the current smoke tests stores data in
        # .data and wraps primitive values in attribute classes.
        if hasattr(d, "data") and isinstance(getattr(d, "data"), dict):
            raw = getattr(d, "data")
            out = {}
            for k, v in raw.items():
                if hasattr(v, "Value"):
                    try:
                        out[k] = v.Value()
                        continue
                    except Exception:
                        pass
                if hasattr(v, "value"):
                    out[k] = getattr(v, "value")
                else:
                    out[k] = v
            return out

        if hasattr(d, "Keys") and hasattr(d, "ValueAtKey"):
            try:
                return {k: d.ValueAtKey(k) for k in d.Keys()}
            except Exception:
                return {}

        if hasattr(d, "keys") and hasattr(d, "__getitem__"):
            try:
                return {k: d[k] for k in d.keys()}
            except Exception:
                return {}

        return {}

    result = to_python_dict(a)
    result.update(to_python_dict(b))
    return result


def _transfer_contents(source: Any, target: Any) -> None:
    if source is None or target is None:
        return

    for attr in ("contents", "contexts", "apertures"):
        if hasattr(source, attr) and hasattr(target, attr):
            try:
                current = _safe_list(getattr(target, attr))
                incoming = _safe_list(getattr(source, attr))
                setattr(target, attr, _deduplicate_by_identity(current + incoming))
            except Exception:
                pass


def _postprocess_boolean_result(shape: Any) -> Any:
    """
    Validates and, if necessary, repairs the result of an OCCT boolean operation.

    Parameters
    ----------
    shape : OCC.Core.TopoDS.TopoDS_Shape
        The input OCCT shape.

    Returns
    -------
    OCC.Core.TopoDS.TopoDS_Shape
        The original shape if valid, otherwise the repaired shape if repair
        succeeds.
    """
    if _is_null_shape(shape):
        return shape

    # ------------------------------------------------------------------
    # Fast path: if the boolean result is already valid, return it
    # without running ShapeFix.
    # ------------------------------------------------------------------
    if BRepCheck_Analyzer is not None:
        try:
            analyzer = BRepCheck_Analyzer(
                shape,
                True,   # Geometric controls
                True,   # Run in parallel
            )

            if analyzer.IsValid():
                return shape

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Repair only if validation failed or could not be performed.
    # ------------------------------------------------------------------
    if ShapeFix_Shape is None:
        return shape

    try:
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()

        fixed_shape = fixer.Shape()

        if not _is_null_shape(fixed_shape):
            return fixed_shape

    except Exception:
        pass

    return shape


def _unify_same_domain(shape: Any) -> Any:
    """
    ShapeUpgrade_UnifySameDomain for Merge/Union only: MakeContainers keeps operands'
    internal grid subdivisions, and merged material must collapse to its minimal
    representation (verified vs topologic_core). Solid-containing shapes only -- too
    aggressive for pure-2D merges where just-touching coplanar faces must stay separate.
    """
    if _is_null_shape(shape) or ShapeUpgrade_UnifySameDomain is None:
        return shape
    if not _iter_occ_subshapes(shape, TopAbs_SOLID):
        return shape
    try:
        unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        unifier.Build()
        unified = unifier.Shape()
        if not _is_null_shape(unified):
            return unified
    except Exception:
        pass
    return shape


def _solids_share_face(solids: list) -> bool:
    """
    True when the Solids connect via shared FACE boundaries (not only an edge/
    vertex), i.e. one block to promote to COMPSOLID. Adjacency graph on faces
    via TopoDS_Shape.IsSame (robust in this build), then a fully-connected
    check.
    """
    if not solids or len(solids) < 2:
        return False

    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    def _faces(solid):
        out = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            out.append(exp.Current())
            exp.Next()
        return out

    face_sets = [_faces(s) for s in solids]
    if any(len(fs) == 0 for fs in face_sets):
        # Can't reliably determine connectivity; be conservative and refuse.
        return False

    n = len(solids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            shared = False
            for fa in face_sets[i]:
                for fb in face_sets[j]:
                    if fa.IsSame(fb):
                        shared = True
                        break
                if shared:
                    break
            if shared:
                union(i, j)

    roots = {find(k) for k in range(n)}
    return len(roots) == 1


def _promote_to_compsolid_if_multi_solid(shape: Any, force: bool = False) -> Any:
    """
    MakeContainers yields a COMPOUND (not COMPSOLID) even for face-sharing solids, so
    rebuild 2+ connected Solids as a CompSolid so ByOcctShape wraps them as a CellComplex.
    force=True (Slice/Imprint/Divide) promotes unconditionally; False (Merge/Impose) only
    when the solids share a face.
    """
    if _is_null_shape(shape):
        return shape
    try:
        if shape.ShapeType() == TopAbs_COMPSOLID:
            return shape
    except Exception:
        return shape

    solids = _iter_occ_subshapes(shape, TopAbs_SOLID)
    if len(solids) < 2:
        return shape

    if not force and not _solids_share_face(solids):
        return shape

    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_CompSolid, topods
        builder = BRep_Builder()
        compsolid = TopoDS_CompSolid()
        builder.MakeCompSolid(compsolid)
        for solid in solids:
            builder.Add(compsolid, topods.Solid(solid))
        return compsolid
    except Exception:
        return shape


def _collect_boolean_operand_shapes(topology: Any) -> list:
    """
    Collect OCCT shapes to use as boolean operands.

    For wrapper objects that have a valid .shape, use the shape directly.
    For aggregate wrappers with no single shape, collect their component shapes.
    """
    if topology is None:
        return []

    shape = _shape_from_topology(topology)
    if not _is_null_shape(shape):
        return [shape]

    result = []

    # Shell wrapper
    if hasattr(topology, "faces"):
        for face in _safe_list(getattr(topology, "faces")):
            f_shape = _shape_from_topology(face)
            if not _is_null_shape(f_shape):
                result.append(f_shape)

    # Cell wrapper
    if hasattr(topology, "shells"):
        for shell in _safe_list(getattr(topology, "shells")):
            result.extend(_collect_boolean_operand_shapes(shell))

    # CellComplex / Cluster style wrappers
    for attr in ("cells", "topologies", "members"):
        if hasattr(topology, attr):
            for child in _safe_list(getattr(topology, attr)):
                result.extend(_collect_boolean_operand_shapes(child))

    return _deduplicate_by_identity(result)


def _wrap_shape_as_topology(shape: Any, dictionary=None, contents=None, contexts=None, apertures=None) -> Any:
    """
    Convert an OCCT shape into the appropriate local backend wrapper object.
    """
    if _is_null_shape(shape):
        return None

    try:
        shape_type = shape.ShapeType()
    except Exception:
        return None

    if shape_type == TopAbs_VERTEX:
        from .vertex import Vertex
        try:
            return Vertex.ByOcctShape(
                topods_Vertex(shape),
                dictionary=dictionary,
                contents=contents,
                contexts=contexts,
                apertures=apertures,
            )
        except Exception:
            return Vertex(shape=topods_Vertex(shape), dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])

    if shape_type == TopAbs_EDGE:
        from .edge import Edge
        try:
            return Edge.ByOcctShape(
                topods_Edge(shape),
                dictionary=dictionary,
                contents=contents,
                contexts=contexts,
                apertures=apertures,
            )
        except Exception:
            return Edge(shape=topods_Edge(shape), dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])

    if shape_type == TopAbs_WIRE:
        from .wire import Wire
        try:
            return Wire.ByOcctShape(
                topods_Wire(shape),
                dictionary=dictionary,
                contents=contents,
                contexts=contexts,
                apertures=apertures,
            )
        except Exception:
            # Fallback: build a Wire from its edges.
            edges = [Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, TopAbs_EDGE)]
            try:
                return Wire.ByEdges(edges)
            except Exception:
                return Wire(shape=topods_Wire(shape), edges=edges, dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])

    if shape_type == TopAbs_FACE:
        from .face import Face
        try:
            return Face.ByOcctShape(
                topods_Face(shape),
                dictionary=dictionary,
                contents=contents,
                contexts=contexts,
                apertures=apertures,
            )
        except Exception:
            try:
                from .wire import Wire
                wires = [Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, TopAbs_WIRE)]
                external = wires[0] if wires else None
                internals = wires[1:] if len(wires) > 1 else []
                return Face(shape=topods_Face(shape), external=external, internals=internals, dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])
            except Exception:
                return None

    if shape_type == TopAbs_SHELL:
        from .shell import Shell
        faces = [Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, TopAbs_FACE)]
        faces = [f for f in faces if f is not None]
        try:
            sh = Shell.ByFaces(faces)
            if sh is not None:
                sh.shape = topods_Shell(shape)
                sh.dictionary = dictionary or {}
                sh.contents = contents or []
                sh.contexts = contexts or []
                sh.apertures = apertures or []
                return sh
        except Exception:
            pass
        return Shell(shape=topods_Shell(shape), faces=faces, dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])

    if shape_type == TopAbs_SOLID:
        from .cell import Cell
        shells = [Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, TopAbs_SHELL)]
        shells = [s for s in shells if s is not None]
        try:
            if hasattr(Cell, "ByShell") and shells:
                c = Cell.ByShell(shells[0])
                if c is not None:
                    c.shape = topods_Solid(shape)
                    c.shells = shells
                    c.dictionary = dictionary or {}
                    c.contents = contents or []
                    c.contexts = contexts or []
                    c.apertures = apertures or []
                    return c
        except Exception:
            pass
        return Cell(shape=topods_Solid(shape), shells=shells, dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])

    if shape_type == TopAbs_COMPSOLID:
        # A COMPSOLID is a non-manifold complex of Solids sharing Faces:
        # exactly what CellComplex represents. Wrap it as such rather than as
        # a generic Cluster, so Cells()/Faces()/etc. traverse from the direct
        # Solid sub-shapes (each already reconstructed as a proper Cell with
        # its own Shells) rather than a flattened grab-bag of every sub-type.
        try:
            from .cell_complex import CellComplex
            cells = [Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, TopAbs_SOLID)]
            cells = [c for c in cells if c is not None]
            if len(cells) == 1:
                # A COMPSOLID wrapping exactly one Solid isn't a genuine
                # non-manifold complex -- e.g. BRepAlgoAPI_Fuse of two
                # COMPSOLID operands that fully merge into one Solid still
                # preserves a COMPSOLID container around that single Solid.
                # Unwrap to the plain Cell directly, matching
                # topologic_core's own behavior (verified: Union of two
                # touching/overlapping boxes is a plain Cell).
                only = cells[0]
                only.dictionary = dictionary or {}
                if contents:
                    only.contents = list(contents)
                if contexts:
                    only.contexts = list(contexts)
                if apertures:
                    only.apertures = list(apertures)
                return only
            cc = CellComplex(shape=shape, cells=cells, dictionary=dictionary or {},
                              contents=contents or [], contexts=contexts or [], apertures=apertures or [])
            return cc
        except Exception:
            return None

    if shape_type == TopAbs_COMPOUND:
        # Prefer Cluster as the safest aggregate wrapper for a heterogeneous
        # compound of mixed sub-shape types.
        try:
            from .cluster import Cluster
            from OCC.Core.TopoDS import TopoDS_Iterator
            # Walk DIRECT children only (TopoDS_Iterator), not every
            # sub-shape of every type anywhere in the tree
            # (_iter_occ_subshapes/TopExp_Explorer): a COMPOUND wrapping one
            # COMPSOLID wrapping one SOLID would otherwise be matched once
            # as a COMPSOLID and again as that same nested SOLID, double
            # counting it and defeating the single-child unwrap below.
            children = []
            it = TopoDS_Iterator(shape)
            while it.More():
                child = Topology.ByOcctShape(it.Value())
                if child is not None:
                    children.append(child)
                it.Next()
            if len(children) == 1:
                # OCCT boolean ops (BRepAlgoAPI_Fuse/Cut/Common/...) always
                # wrap their result in a COMPOUND container even when it is
                # a single piece -- e.g. two fully-overlapping/flush-touching
                # solids fuse into one Solid, but Shape() still hands back a
                # 1-child COMPOUND around it. Unwrap to that single child's
                # own type directly (matching topologic_core's own behavior:
                # a Union of two touching boxes is a plain Cell, not a
                # 1-member Cluster) instead of always forcing a Cluster.
                only = children[0]
                only.dictionary = dictionary or {}
                if contents:
                    only.contents = list(contents)
                if contexts:
                    only.contexts = list(contexts)
                if apertures:
                    only.apertures = list(apertures)
                return only
            if children and all(Topology.IsInstance(c, "Face") for c in children):
                # Multiple coplanar/adjacent Faces (e.g. Merge of two Faces
                # that only partially overlap) form a Shell, not a Cluster
                # -- verified against topologic_core: merging a
                # rectangle-with-a-hole with a second rectangle that only
                # partially covers the hole gives a Shell of Faces.
                from .shell import Shell
                sh = Shell.ByFaces(children)
                if sh is not None:
                    sh.dictionary = dictionary or {}
                    if contents:
                        sh.contents = list(contents)
                    if contexts:
                        sh.contexts = list(contexts)
                    if apertures:
                        sh.apertures = list(apertures)
                    return sh
            if children and len(children) >= 2 and all(Topology.IsInstance(c, "Edge") for c in children):
                # A connected collection of Edges is a Topologic Wire even when
                # it is non-manifold (branch vertices may have degree > 2).
                #
                # The PythonOCC Wire.ByEdges implementation now supports such
                # connected branching networks by using OCCT's batch MakeWire
                # path when a single head-to-tail ordering does not exist.
                # Therefore the old "degree <= 2" restriction is no longer
                # correct here.
                #
                # Only promote when every input Edge belongs to one connected
                # component. A genuinely disjoint edge set remains a Cluster.
                from .wire import Wire
                from .helpers import same_vertex

                n = len(children)
                parent = list(range(n))

                def _find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for i in range(n):
                    for j in range(i + 1, n):
                        ei, ej = children[i], children[j]

                        if (
                            same_vertex(ei.start, ej.start)
                            or same_vertex(ei.start, ej.end)
                            or same_vertex(ei.end, ej.start)
                            or same_vertex(ei.end, ej.end)
                        ):
                            ra = _find(i)
                            rb = _find(j)

                            if ra != rb:
                                parent[ra] = rb

                if len({_find(i) for i in range(n)}) == 1:
                    w = Wire.ByEdges(children)

                    # Never accept a partial reconstruction. This guards
                    # against any OCCT MakeWire result that silently omits
                    # one or more supplied Edges.
                    if (
                        w is not None
                        and len(getattr(w, "edges", None) or []) == n
                    ):
                        w.dictionary = dictionary or {}

                        if contents:
                            w.contents = list(contents)

                        if contexts:
                            w.contexts = list(contexts)

                        if apertures:
                            w.apertures = list(apertures)

                        return w
            if hasattr(Cluster, "ByTopologies"):
                cl = Cluster.ByTopologies(children)
                if cl is not None:
                    cl.shape = shape
                    cl.dictionary = dictionary or {}
                    return cl
            return Cluster(shape=shape, topologies=children, dictionary=dictionary or {}, contents=contents or [], contexts=contexts or [], apertures=apertures or [])
        except Exception:
            return None

    return None


def _compound_of_shapes(shapes: Iterable[Any]) -> Any:
    from OCC.Core.TopoDS import TopoDS_Compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        if not _is_null_shape(shape):
            builder.Add(compound, shape)
    return compound

def _make_occ_merge(topology: Any, other_topology: Any = None, transfer_dictionary: bool = False) -> Any:
    """
    PythonOCC Topology.Merge: BOPAlgo_CellsBuilder + AddAllToResult (keep ALL material)
    then MakeContainers(), so the operands' shared face survives as a real internal
    (non-manifold) boundary -- the result is a CellComplex, not one dissolved Cell.
    """
    if topology is None:
        return None

    base_shape = _shape_from_topology(topology)
    if _is_null_shape(base_shape):
        return None

    if other_topology is None:
        return Topology.ByOcctShape(
            base_shape,
            dictionary=Topology.Dictionary(topology),
            contents=getattr(topology, "contents", []),
            contexts=getattr(topology, "contexts", []),
            apertures=getattr(topology, "apertures", []),
        )

    shapes_a = _collect_boolean_operand_shapes(topology)
    shapes_b = _collect_boolean_operand_shapes(other_topology)

    if not shapes_a or not shapes_b:
        print("Topology.Merge - Error: Could not collect valid OCCT operands. Returning None.")
        return None

    # Merge/Union in Topologic preserves the *interface* between the two
    # operands as a real internal (non-manifold) boundary, so the result is a
    # CellComplex -- not a single dissolved Cell. A true BRepAlgoAPI_Fuse
    # welds overlapping/flush-touching solids into one Solid and erases that
    # interface, which is why the old Fuse path returned a Cell and failed
    # test_09Topology.py (Topology.Merge should be a CellComplex).
    #
    # BOPAlgo_CellsBuilder is the right tool here: with every operand added as
    # an argument and AddAllToResult() (NO geometric classification filter --
    # Merge keeps *all* material from both operands, unlike Divide/Slice which
    # keep only self's fragments), MakeContainers() partitions the fused space
    # into cells that keep the shared face between the operands as a genuine
    # internal boundary. _promote_to_compsolid_if_multi_solid then turns the
    # resulting COMPOUND of solids into a CompSolid so ByOcctShape wraps it as
    # a CellComplex.
    if BOPAlgo_CellsBuilder is None:
        print("Topology.Merge - Error: PythonOCC BOPAlgo_CellsBuilder is not available. Returning None.")
        return None

    try:
        builder = BOPAlgo_CellsBuilder()
        for shape in shapes_a:
            builder.AddArgument(shape)
        for shape in shapes_b:
            builder.AddArgument(shape)
        builder.Perform()
        if hasattr(builder, "HasErrors") and builder.HasErrors():
            print("Topology.Merge - Error: BOPAlgo_CellsBuilder failed. Returning None.")
            return None

        builder.AddAllToResult()
        result_shape = builder.Shape()
    except Exception:
        print("Topology.Merge - Error: BOPAlgo_CellsBuilder raised. Returning None.")
        return None

    if _is_null_shape(result_shape):
        print("Topology.Merge - Error: Boolean result is null. Returning None.")
        return None

    result_shape = _postprocess_boolean_result(result_shape)
    result_shape = _unify_same_domain(result_shape)
    result_shape = _promote_to_compsolid_if_multi_solid(result_shape)

    result_dictionary = {}
    if transfer_dictionary:
        result_dictionary = _merge_backend_dictionaries(
            Topology.Dictionary(topology),
            Topology.Dictionary(other_topology),
        )

    result = Topology.ByOcctShape(
        result_shape,
        dictionary=result_dictionary,
        contents=[],
        contexts=[],
        apertures=[],
    )

    if result is None:
        print("Topology.Merge - Error: Could not convert OCCT result to backend topology. Returning None.")
        return None

    _transfer_contents(topology, result)
    _transfer_contents(other_topology, result)

    return result


def _make_occ_union(
    topology: Any,
    other_topology: Any = None,
    transfer_dictionary: bool = False
) -> Any:
    """
    Returns the union of the input topologies using the PythonOCC backend.

    The OCCT Fuse operation is used as the primary Boolean operation.
    Backend-specific semantic normalization is then applied only where
    necessary to preserve Topologic topology semantics.

    In particular:

    - Overlapping or touching solids are fused normally.
    - Disjoint solids remain separate and are wrapped according to the
      resulting OCCT topology.
    - Overlapping coplanar Face/Face unions are unified into a single
      same-domain Face while preserving boundary edge segmentation.
    - Connected Wire/Wire unions are promoted to a Topologic Wire when
      all resulting edges can be retained without loss.

    Parameters
    ----------
    topology : Any
        The first input topology.
    other_topology : Any , optional
        The second input topology. Default is None.
    transfer_dictionary : bool , optional
        If True, the dictionaries of the input topologies are merged and
        assigned to the result. Default is False.

    Returns
    -------
    Any
        The resultant topology, or None if the operation fails.
    """
    if topology is None:
        return None

    base_shape = _shape_from_topology(
        topology
    )

    if _is_null_shape(base_shape):
        return None

    # ------------------------------------------------------------------
    # No second operand
    # ------------------------------------------------------------------

    if other_topology is None:
        return Topology.ByOcctShape(
            base_shape,
            dictionary=Topology.Dictionary(
                topology
            ),
            contents=getattr(
                topology,
                "contents",
                []
            ),
            contexts=getattr(
                topology,
                "contexts",
                []
            ),
            apertures=getattr(
                topology,
                "apertures",
                []
            ),
        )

    # ------------------------------------------------------------------
    # Collect OCCT operand shapes
    # ------------------------------------------------------------------

    shapes_a = _collect_boolean_operand_shapes(
        topology
    )

    shapes_b = _collect_boolean_operand_shapes(
        other_topology
    )

    if not shapes_a or not shapes_b:
        print(
            "Topology.Union - Error: Could not collect valid OCCT "
            "operands. Returning None."
        )
        return None

    if BRepAlgoAPI_Fuse is None:
        print(
            "Topology.Union - Error: PythonOCC BRepAlgoAPI_Fuse is "
            "not available. Returning None."
        )
        return None

    # ------------------------------------------------------------------
    # OCCT Fuse
    # ------------------------------------------------------------------

    try:
        if len(shapes_a) == 1:
            shape_a = shapes_a[0]
        else:
            shape_a = _compound_of_shapes(
                shapes_a
            )

        if len(shapes_b) == 1:
            shape_b = shapes_b[0]
        else:
            shape_b = _compound_of_shapes(
                shapes_b
            )

        fuse = BRepAlgoAPI_Fuse(
            shape_a,
            shape_b
        )

        fuse.Build()

        if not fuse.IsDone():
            print(
                "Topology.Union - Error: BRepAlgoAPI_Fuse failed. "
                "Returning None."
            )
            return None

        result_shape = fuse.Shape()

    except Exception:
        print(
            "Topology.Union - Error: BRepAlgoAPI_Fuse raised. "
            "Returning None."
        )
        return None

    if _is_null_shape(result_shape):
        print(
            "Topology.Union - Error: Boolean result is null. "
            "Returning None."
        )
        return None

    # ------------------------------------------------------------------
    # Face / Face semantic normalization
    #
    # OCCT Fuse preserves the Boolean partition when two coplanar Faces
    # overlap. For example, two overlapping rectangles may be returned as
    # three adjacent Faces contained in a Shell/Compound:
    #
    #     A-only | overlap | B-only
    #
    # Topologic Union semantics require these same-domain regions to form
    # one Face.
    #
    # Unify Faces only. Do NOT unify boundary Edges: their segmentation is
    # valid topology and must be preserved. In the Boolean regression case
    # this converts:
    #
    #     Shell: 3 Faces / 10 Edges / 8 Vertices
    #
    # into:
    #
    #     Face: 1 Face / 8 Edges / 8 Vertices
    #
    # This normalization is deliberately restricted to Face/Face operands.
    # Applying same-domain unification generally to solid results can erase
    # valid Boolean partitions and alter expected topology counts.
    # ------------------------------------------------------------------

    try:
        from .face import Face

        if (
            isinstance(topology, Face)
            and isinstance(other_topology, Face)
            and ShapeUpgrade_UnifySameDomain is not None
        ):
            result_faces = _iter_occ_subshapes_unique(
                result_shape,
                TopAbs_FACE
            )

            if len(result_faces) > 1:
                unifier = ShapeUpgrade_UnifySameDomain(
                    result_shape,
                    False,  # UnifyEdges
                    True,   # UnifyFaces
                    False   # ConcatBSplines
                )

                unifier.Build()

                unified_shape = unifier.Shape()

                if not _is_null_shape(
                    unified_shape
                ):
                    result_shape = unified_shape

    except Exception:
        pass

    # ------------------------------------------------------------------
    # Pure 1-D same-domain normalization
    #
    # Do NOT apply general same-domain unification to solid or face
    # results. BRepAlgoAPI_Fuse already gives the desired partitioning
    # there, apart from the explicitly handled Face/Face case above.
    #
    # Pure Edge/Wire operands are different. Fusing two overlapping
    # collinear Edges can leave several split segments instead of the
    # single logical Edge expected by Topologic. Therefore same-domain
    # unification is permitted for purely 1-D results.
    # ------------------------------------------------------------------

    if (
        ShapeUpgrade_UnifySameDomain is not None
        and not _iter_occ_subshapes(
            result_shape,
            TopAbs_SOLID
        )
        and not _iter_occ_subshapes(
            result_shape,
            TopAbs_FACE
        )
    ):
        try:
            unifier = ShapeUpgrade_UnifySameDomain(
                result_shape,
                True,
                True,
                True
            )

            unifier.Build()

            unified_shape = unifier.Shape()

            if not _is_null_shape(
                unified_shape
            ):
                result_shape = unified_shape

        except Exception:
            pass

    # ------------------------------------------------------------------
    # General Boolean post-processing
    # ------------------------------------------------------------------

    result_shape = _postprocess_boolean_result(
        result_shape
    )

    # ------------------------------------------------------------------
    # Result dictionary
    # ------------------------------------------------------------------

    result_dictionary = {}

    if transfer_dictionary:
        result_dictionary = _merge_backend_dictionaries(
            Topology.Dictionary(
                topology
            ),
            Topology.Dictionary(
                other_topology
            ),
        )

    # ------------------------------------------------------------------
    # Wire / Wire semantic normalization
    #
    # A connected branching one-dimensional Fuse result may be stored by
    # OCCT in a Compound even though it represents one valid Topologic
    # non-manifold Wire.
    #
    # Build the Wire directly from the unique resulting OCCT Edges. Accept
    # the promotion only if every Edge survives, so genuinely disconnected
    # results remain aggregates.
    # ------------------------------------------------------------------

    from .wire import Wire

    if (
        isinstance(topology, Wire)
        and isinstance(other_topology, Wire)
    ):
        edge_shapes = _iter_occ_subshapes_unique(
            result_shape,
            TopAbs_EDGE
        )

        edges = [
            Topology.ByOcctShape(
                edge_shape
            )
            for edge_shape in edge_shapes
        ]

        edges = [
            edge
            for edge in edges
            if edge is not None
        ]

        if (
            edges
            and len(edges) == len(edge_shapes)
        ):
            wire_result = Wire.ByEdges(
                edges
            )

            if (
                wire_result is not None
                and len(
                    wire_result.Edges() or []
                ) == len(edges)
            ):
                wire_result.dictionary = result_dictionary

                _transfer_contents(
                    topology,
                    wire_result
                )

                _transfer_contents(
                    other_topology,
                    wire_result
                )

                return wire_result

    # ------------------------------------------------------------------
    # Generic result wrapping
    # ------------------------------------------------------------------

    result = Topology.ByOcctShape(
        result_shape,
        dictionary=result_dictionary,
        contents=[],
        contexts=[],
        apertures=[],
    )

    if result is None:
        print(
            "Topology.Union - Error: Could not convert OCCT result to "
            "backend topology. Returning None."
        )
        return None

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    _transfer_contents(
        topology,
        result
    )

    _transfer_contents(
        other_topology,
        result
    )

    return result

# BREP / string serialization helpers
# -----------------------------------------------------------------------------
#
# PythonOCC does not expose a stable, version-independent in-memory string BREP
# API across releases (older builds expose free functions such as
# ``breptools_Write``/``breptools_Read``; newer builds expose them as static
# methods ``BRepTools.Write_s``/``BRepTools.Read_s``). We defensively support
# both, round-tripping through a short-lived temp file since that is the one
# entry point guaranteed to exist in every PythonOCC generation.

_BREP_STRING_FORMAT = "topologicpy-pythonocc-brep-v1"



def _ensure_compound_shape(topology: Any) -> Any:
    """
    Return a valid OCCT shape for a wrapper, building a compound for shapeless
    aggregates (a Cluster constructed via ByTopologies keeps shape=None).
    Without this, BREP serialization of a Cluster returns None.
    """
    shape = _shape_from_topology(topology)
    if not _is_null_shape(shape):
        return shape
    members = getattr(topology, "topologies", None)
    if not members:
        return None
    member_shapes = []
    for m in members:
        s_ = _ensure_compound_shape(m)
        if s_ is not None and not _is_null_shape(s_):
            member_shapes.append(s_)
    if not member_shapes:
        return None
    try:
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s_ in member_shapes:
            builder.Add(compound, s_)
        return compound
    except Exception:
        return None


def _shape_to_brep_text(shape: Any) -> Optional[str]:
    if _is_null_shape(shape):
        return None

    try:
        import OCC.Core.BRepTools as _breptools_module
    except Exception:
        return None

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".brep")
        os.close(fd)

        wrote = False
        wrote = False
        if hasattr(_breptools_module, "breptools") and hasattr(_breptools_module.breptools, "Write"):
            _breptools_module.breptools.Write(shape, tmp_path)
            wrote = True
        elif hasattr(_breptools_module, "breptools_Write"):
            _breptools_module.breptools_Write(shape, tmp_path)
            wrote = True
        if not wrote:
            return None

        with open(tmp_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except Exception:
        return None
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _shape_from_brep_text(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return None

    try:
        import OCC.Core.BRepTools as _breptools_module
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder
    except Exception:
        return None

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".brep")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)

        shape = TopoDS_Shape()
        builder = BRep_Builder()

        read_ok = None
        if hasattr(_breptools_module, "breptools") and hasattr(_breptools_module.breptools, "Read"):
            read_ok = _breptools_module.breptools.Read(shape, tmp_path, builder)
        elif hasattr(_breptools_module, "breptools_Read"):
            read_ok = _breptools_module.breptools_Read(shape, tmp_path, builder)

        if read_ok is False or _is_null_shape(shape):
            return None

        return shape
    except Exception:
        return None
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _dictionary_to_json_safe(dictionary: Any) -> Any:
    """Best-effort conversion of a backend dictionary into a JSON-serialisable dict."""
    plain = _merge_backend_dictionaries(dictionary, None)
    if not plain:
        return None
    try:
        json.dumps(plain)
    except Exception:
        return None
    return plain


# -----------------------------------------------------------------------------
# Public Topology class
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class Topology:
    # Base fields inherited by every wrapper subclass (Vertex, Edge, Wire,
    # Face, Shell, Cell, CellComplex, Cluster, Aperture, Context). These were
    # previously missing entirely, which meant every subclass constructor call
    # such as ``Vertex(shape=..., dictionary=..., contents=..., contexts=...,
    # apertures=...)`` raised "unexpected keyword argument" and no topology
    # object of any kind could be built. Shell and Cell define their own
    # __init__ (which dataclass leaves untouched) and forward to this one via
    # super().__init__(...).
    shape: Any = None
    dictionary: Any = None
    contents: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    apertures: list = field(default_factory=list)
    _uuid: str = field(default_factory=_new_uuid)

    def __hash__(self) -> int:
        return hash(self._uuid)

    def Type(self) -> Optional[int]:
        """
        Return the numeric topologic_core-style type id. A plain instance method
        so both Core.InstanceCall(topology,'Type') and Topology.Type(topology)
        calling conventions work (inherited by every subclass).
        """
        result = _type_id_for(self)
        if result is None:
            print("Topology.Type - Error: The input topology parameter is not a valid topology or graph. Returning None.")
        return result

    @staticmethod
    def IsInstance(topology: Any, typeName: str) -> bool:
        if typeName is None:
            return False

        requested = str(typeName).strip().lower()
        actual = _topology_type_name(topology)

        if requested == "topology":
            return _is_topology_like(topology)

        if actual is None:
            return False

        return actual.lower() == requested

    def TypeAsString(self) -> Optional[str]:
        """
        Instance method (not @staticmethod) so both calling conventions work:
        Core.Topology.TypeAsString(topology) and
        Core.InstanceCall(topology, 'GetTypeAsString').
        """
        result = _topology_type_name(self)
        if result is None:
            print("Topology.TypeAsString - Error: The input topology parameter is not a valid topology or graph. Returning None.")
        return result

    # Real topologic_core exposes this instance method under the name
    # GetTypeAsString; TopologyUtility/tests may also reference TypeAsString.
    GetTypeAsString = TypeAsString

    def GetDictionary(self):
        if self is None:
            return None

        if isinstance(self, dict):
            return self.get("dictionary", None) or {}

        shape = _shape_from_topology(self)

        if not _is_null_shape(shape):
            manager = AttributeManager.GetInstance()

            if manager.HasDictionary(shape):
                return manager.GetDictionary(shape)

        # Fallback for shapeless wrapper objects.
        return getattr(self, "dictionary", None) or {}


    Dictionary = GetDictionary


    def SetDictionary(self, dictionary: Any):
        if self is None:
            return None

        dictionary = dictionary if dictionary is not None else {}

        if isinstance(self, dict):
            self["dictionary"] = dictionary
            return self

        shape = _shape_from_topology(self)

        if not _is_null_shape(shape):
            AttributeManager.GetInstance().SetDictionary(
                shape,
                dictionary
            )

        # Keep the wrapper attribute as a compatibility mirror.
        try:
            self.dictionary = dictionary
        except Exception:
            pass

        return self

    @staticmethod
    def ByOcctShape(
        shape: Any,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None
    ):
        """
        Wraps an existing OCCT shape as the corresponding PythonOCC backend
        topology.

        Shape-keyed dictionaries and relationships are restored from
        AttributeManager so that metadata survives reconstruction of transient
        Python wrappers around the same OCCT topology.
        """
        if _is_null_shape(shape):
            return None

        try:
            shape_type = shape.ShapeType()
        except Exception:
            return None

        manager = AttributeManager.GetInstance()

        dictionary_known = dictionary is not None or manager.HasDictionary(shape)
        contents_known = contents is not None or manager.HasContents(shape)
        contexts_known = contexts is not None or manager.HasContexts(shape)
        apertures_known = apertures is not None or manager.HasApertures(shape)

        if dictionary is None:
            dictionary = manager.GetDictionary(shape) if manager.HasDictionary(shape) else {}
        else:
            manager.SetDictionary(shape, dictionary)
            dictionary = manager.GetDictionary(shape)

        if contents is None:
            contents = manager.GetContents(shape) if manager.HasContents(shape) else []
        else:
            manager.SetContents(shape, _safe_list(contents))
            contents = manager.GetContents(shape)

        if contexts is None:
            contexts = manager.GetContexts(shape) if manager.HasContexts(shape) else []
        else:
            manager.SetContexts(shape, _safe_list(contexts))
            contexts = manager.GetContexts(shape)

        if apertures is None:
            apertures = manager.GetApertures(shape) if manager.HasApertures(shape) else []
        else:
            manager.SetApertures(shape, _safe_list(apertures))
            apertures = manager.GetApertures(shape)

        def _finalize(result):
            if result is None:
                return None

            actual_shape = _shape_from_topology(result)

            if not _is_null_shape(actual_shape):
                if dictionary_known:
                    manager.SetDictionary(actual_shape, dictionary)
                if contents_known:
                    manager.SetContents(actual_shape, contents)
                if contexts_known:
                    manager.SetContexts(actual_shape, contexts)
                if apertures_known:
                    manager.SetApertures(actual_shape, apertures)

            try:
                result.dictionary = dictionary
            except Exception:
                pass
            try:
                result.contents = list(contents)
            except Exception:
                pass
            try:
                result.contexts = list(contexts)
            except Exception:
                pass
            try:
                result.apertures = list(apertures)
            except Exception:
                pass

            return result

        # ------------------------------------------------------------------
        # Vertex / Edge / Wire / Face
        # ------------------------------------------------------------------
        if shape_type in (
            TopAbs_VERTEX,
            TopAbs_EDGE,
            TopAbs_WIRE,
            TopAbs_FACE,
        ):
            return _finalize(
                _wrap_shape_as_topology(
                    shape,
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures,
                )
            )

        # ------------------------------------------------------------------
        # Shell -- shallow wrapper only.
        # ------------------------------------------------------------------
        if shape_type == TopAbs_SHELL:
            from .shell import Shell
            return _finalize(
                Shell(
                    shape=topods_Shell(shape),
                    faces=[],
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures,
                )
            )

        # ------------------------------------------------------------------
        # Solid -> Cell -- shallow wrapper only.
        # ------------------------------------------------------------------
        if shape_type == TopAbs_SOLID:
            from .cell import Cell
            return _finalize(
                Cell(
                    shape=topods_Solid(shape),
                    shells=[],
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures,
                )
            )

        # ------------------------------------------------------------------
        # CompSolid -> CellComplex
        # ------------------------------------------------------------------
        if shape_type == TopAbs_COMPSOLID:
            from .cell import Cell
            from .cell_complex import CellComplex

            solids = _iter_occ_subshapes_unique(
                shape,
                TopAbs_SOLID
            )

            # Preserve the existing backend behaviour of unwrapping a
            # one-solid CompSolid to a Cell.
            if len(solids) == 1:
                return _finalize(
                    Cell(
                        shape=topods_Solid(solids[0]),
                        shells=[],
                        dictionary=dictionary,
                        contents=contents,
                        contexts=contexts,
                        apertures=apertures,
                    )
                )

            return _finalize(
                CellComplex(
                    shape=shape,
                    cells=[],
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures,
                )
            )

        # ------------------------------------------------------------------
        # Compound
        # ------------------------------------------------------------------
        if shape_type == TopAbs_COMPOUND:
            return _finalize(
                _wrap_shape_as_topology(
                    shape,
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures,
                )
            )

        return None

    @staticmethod
    def BREPString(topology: Any, version: int = 0) -> Optional[str]:
        """
        Returns a raw OCCT BREP text representation of the topology's shape.

        This only round-trips OCCT geometry/topology; attached dictionaries are
        not included here (see Topology.String for a dictionary-preserving
        variant).
        """
        shape = _ensure_compound_shape(topology)
        return _shape_to_brep_text(shape)

    @staticmethod
    def String(topology: Any, version: int = 0) -> Optional[str]:
        """
        Returns a textual serialization of the topology.

        This is a small JSON envelope wrapping the raw BREP text plus (when
        possible) the topology's attached dictionary, so that
        Topology.ByString can round-trip both geometry and metadata. If the
        dictionary cannot be safely converted to JSON it is dropped rather
        than failing the whole call.
        """
        shape = _ensure_compound_shape(topology)
        brep_text = _shape_to_brep_text(shape)
        if brep_text is None:
            return None

        envelope = {
            "format": _BREP_STRING_FORMAT,
            "version": version,
            "typeName": _topology_type_name(topology),
            "brep": brep_text,
            "dictionary": _dictionary_to_json_safe(Topology.Dictionary(topology)),
        }
        try:
            return json.dumps(envelope)
        except Exception:
            return brep_text

    @staticmethod
    def ByString(string: Any):
        """
        Reconstructs a topology from a string produced by Topology.String or
        Topology.BREPString (or a raw OCCT BREP text string from another
        source).
        """
        if not isinstance(string, str) or not string:
            return None

        brep_text = string
        dictionary = None

        try:
            parsed = json.loads(string)
            if isinstance(parsed, dict) and parsed.get("format") == _BREP_STRING_FORMAT:
                brep_text = parsed.get("brep")
                dictionary = parsed.get("dictionary")
        except Exception:
            pass

        shape = _shape_from_brep_text(brep_text)
        if shape is None:
            return None

        return Topology.ByOcctShape(shape, dictionary=dictionary)

    def _dispatch_subtopologies(
        self,
        own_attr,
        shape_type,
        child_attrs,
        recurse_name,
        self_type_name=None,
        hostTopology=None,
        output=None
    ):
        """
        Returns subtopologies of the requested type.

        The method first uses explicitly stored subtopology wrappers when they
        exist and are non-empty. If no stored wrappers are available, it queries
        the native OCCT shape. If that also produces no result, it recursively
        searches child topology collections.

        This supports both Pythonic return-list calls and topologic_core-style
        output-list calls.

        Parameters
        ----------
        self : Topology
            The input topology.
        own_attr : str
            The attribute name containing cached subtopologies of the requested
            type.
        shape_type : OCC.Core.TopAbs.TopAbs_ShapeEnum
            The requested native OCCT shape type.
        child_attrs : tuple
            Attribute names containing lower-level topology collections through
            which the query may recurse.
        recurse_name : str
            The topology query method name used during recursive traversal.
        self_type_name : str , optional
            The Topologic type name corresponding to the requested topology.
        hostTopology : Topology , optional
            The host topology used for super-topology queries.
        output : list , optional
            An output list following the topologic_core calling convention.

        Returns
        -------
        list or int
            The requested subtopologies, or 0 when an output list is supplied.
        """

        if self is None:
            if output is not None:
                return 0
            return None

        # ------------------------------------------------------------------
        # Determine source and target dimensional ranks.
        # ------------------------------------------------------------------

        self_type = _topology_type_name(
            self
        )

        self_rank = _TYPE_RANK.get(
            self_type
        )

        target_rank = (
            _TYPE_RANK.get(
                self_type_name
            )
            if self_type_name
            else None
        )

        # ------------------------------------------------------------------
        # Super-topology / adjacency query.
        # ------------------------------------------------------------------

        if (
            hostTopology is not None
            and self_rank is not None
            and target_rank is not None
            and self_rank < target_rank
            and not hasattr(self, own_attr)
        ):

            result = Topology.SuperTopologies(
                self,
                hostTopology,
                self_type_name
            ) or []

            result = _deduplicate_by_identity(
                result
            )

            if output is not None:
                output.extend(
                    result
                )
                return 0

            return result

        # ------------------------------------------------------------------
        # The topology itself is already of the requested type.
        # ------------------------------------------------------------------

        if (
            self_type_name is not None
            and self_type == self_type_name
        ):

            result = [
                self
            ]

            if output is not None:
                output.extend(
                    result
                )
                return 0

            return result

        result = []

        # ------------------------------------------------------------------
        # 1. Use explicitly stored wrappers when NON-EMPTY.
        #
        # An empty list is not necessarily authoritative. Container wrappers
        # created from native OCCT shapes are intentionally shallow, e.g.:
        #
        #     Cell(shape=solid, shells=[])
        #     Shell(shape=shell, faces=[])
        #
        # Their descendants must therefore be discovered from the OCCT shape.
        # ------------------------------------------------------------------

        if hasattr(
            self,
            own_attr
        ):

            try:
                stored = _safe_list(
                    getattr(
                        self,
                        own_attr
                    )
                )
            except Exception:
                stored = []

            if stored:
                result.extend(
                    stored
                )

        # ------------------------------------------------------------------
        # 2. If no cached wrappers were available, inspect the native OCCT
        #    topology.
        # ------------------------------------------------------------------

        if not result:

            shape = _shape_from_topology(
                self
            )

            if (
                not _is_null_shape(shape)
                and shape_type is not None
            ):

                try:
                    subshapes = _iter_occ_subshapes(
                        shape,
                        shape_type
                    )
                except Exception:
                    subshapes = []

                for subshape in subshapes:

                    try:
                        item = Topology.ByOcctShape(
                            subshape
                        )
                    except Exception:
                        item = None

                    if item is not None:
                        result.append(
                            item
                        )

        # ------------------------------------------------------------------
        # 3. Aggregate / wrapper fallback.
        #
        # For shapeless aggregate wrappers such as the lightweight Cluster,
        # recurse through their direct topology members.
        # ------------------------------------------------------------------

        if not result:

            for attr in child_attrs:

                if not hasattr(
                    self,
                    attr
                ):
                    continue

                try:
                    children = _safe_list(
                        getattr(
                            self,
                            attr
                        )
                    )
                except Exception:
                    children = []

                for child in children:

                    if child is None:
                        continue

                    try:
                        child_result = getattr(
                            Topology,
                            recurse_name
                        )(
                            child
                        ) or []

                    except Exception:
                        child_result = []

                    result.extend(
                        child_result
                    )

        # ------------------------------------------------------------------
        # Deduplicate using native identity where possible.
        # ------------------------------------------------------------------

        result = _deduplicate_by_identity(
            result
        )

        if output is not None:
            output.extend(
                result
            )
            return 0

        return result

    def Vertices(self, hostTopology=None, output=None):
        if hasattr(self, "start") and hasattr(self, "end") and not hasattr(self, "vertices"):
            result = [v for v in [getattr(self, "start"), getattr(self, "end")] if v is not None]
            result = _deduplicate_by_identity(result)
            if output is not None:
                output.extend(result)
                return 0
            return result
        return Topology._dispatch_subtopologies(
            self, "vertices", TopAbs_VERTEX, ("edges", "faces", "shells", "cells", "topologies", "members"),
            "Vertices", self_type_name="Vertex", hostTopology=hostTopology, output=output,
        )

    def Edges(self, hostTopology=None, output=None):
        return Topology._dispatch_subtopologies(
            self, "edges", TopAbs_EDGE, ("faces", "shells", "cells", "topologies", "members"),
            "Edges", self_type_name="Edge", hostTopology=hostTopology, output=output,
        )

    def Wires(self, hostTopology=None, output=None):
        if hasattr(self, "external") and getattr(self, "external") is not None:
            result = [getattr(self, "external")]
            result.extend(_safe_list(getattr(self, "internals", [])))
            result = _deduplicate_by_identity(result)
            if output is not None:
                output.extend(result)
                return 0
            return result
        return Topology._dispatch_subtopologies(
            self, "wires", TopAbs_WIRE, ("faces", "shells", "cells", "topologies", "members"),
            "Wires", self_type_name="Wire", hostTopology=hostTopology, output=output,
        )

    def Faces(self, hostTopology=None, output=None):
        return Topology._dispatch_subtopologies(
            self, "faces", TopAbs_FACE, ("shells", "cells", "topologies", "members"),
            "Faces", self_type_name="Face", hostTopology=hostTopology, output=output,
        )

    def Shells(self, hostTopology=None, output=None):
        return Topology._dispatch_subtopologies(
            self, "shells", TopAbs_SHELL, ("cells", "topologies", "members"),
            "Shells", self_type_name="Shell", hostTopology=hostTopology, output=output,
        )

    def Cells(self, hostTopology=None, output=None):
        return Topology._dispatch_subtopologies(
            self, "cells", TopAbs_SOLID, ("topologies", "members"),
            "Cells", self_type_name="Cell", hostTopology=hostTopology, output=output,
        )

    def CellComplexes(self, hostTopology=None, output=None):
        return Topology._dispatch_subtopologies(
            self, "cellComplexes", TopAbs_COMPSOLID, ("topologies", "members"),
            "CellComplexes", self_type_name="CellComplex", hostTopology=hostTopology, output=output,
        )

    def Clusters(self, hostTopology=None, output=None):
        result = [self] if _topology_type_name(self) == "Cluster" else []
        if output is not None:
            output.extend(result)
            return 0
        return result

    def Apertures(self, output=None):
        """
        Returns the apertures attached to this topology.

        The signature intentionally matches topologic_core's output-list calling
        convention: topology.Apertures(output_list).
        """
        if self is None:
            result = []
        else:
            shape = _shape_from_topology(self)
            local = _safe_list(getattr(self, "apertures", []))

            if not _is_null_shape(shape):
                manager = AttributeManager.GetInstance()
                if manager.HasApertures(shape):
                    result = manager.GetApertures(shape)
                else:
                    result = _deduplicate_by_identity(local)
                    if result:
                        manager.SetApertures(shape, result)
            else:
                result = _deduplicate_by_identity(local)

            try:
                self.apertures = list(result)
            except Exception:
                pass

        if output is not None:
            output.extend(result)
            return 0

        return result

    def Contents(self, output=None):
        """Returns the contents attached to this topology."""
        if self is None:
            result = []
        else:
            shape = _shape_from_topology(self)
            local = _safe_list(getattr(self, "contents", []))

            if not _is_null_shape(shape):
                manager = AttributeManager.GetInstance()
                if manager.HasContents(shape):
                    result = manager.GetContents(shape)
                else:
                    result = _deduplicate_by_identity(local)
                    if result:
                        manager.SetContents(shape, result)
            else:
                result = _deduplicate_by_identity(local)

            try:
                self.contents = list(result)
            except Exception:
                pass

        if output is not None:
            output.extend(result)
            return 0

        return result

    def Contexts(self, output=None):
        """Returns the contexts attached to this topology."""
        if self is None:
            result = []
        else:
            shape = _shape_from_topology(self)
            local = _safe_list(getattr(self, "contexts", []))

            if not _is_null_shape(shape):
                manager = AttributeManager.GetInstance()
                if manager.HasContexts(shape):
                    result = manager.GetContexts(shape)
                else:
                    result = _deduplicate_by_identity(local)
                    if result:
                        manager.SetContexts(shape, result)
            else:
                result = _deduplicate_by_identity(local)

            try:
                self.contexts = list(result)
            except Exception:
                pass

        if output is not None:
            output.extend(result)
            return 0

        return result

    def AddContent(self, content: Any):
        """Adds one content topology and persists the relationship by OCCT shape identity."""
        if self is None or content is None:
            return self

        shape = _shape_from_topology(self)

        if not _is_null_shape(shape):
            manager = AttributeManager.GetInstance()
            if manager.HasContents(shape):
                current = manager.GetContents(shape)
            else:
                current = _safe_list(getattr(self, "contents", []))

            current = _deduplicate_by_identity(current + [content])
            manager.SetContents(shape, current)
            self.contents = manager.GetContents(shape)
        else:
            self.contents = _deduplicate_by_identity(
                _safe_list(getattr(self, "contents", [])) + [content]
            )

        return self

    def AddContents(self, contents: Any, typeID: Any = None):
        """Adds multiple contents. typeID is accepted for topologic_core compatibility."""
        for content in _safe_list(contents):
            self.AddContent(content)

        return self

    def AddContext(self, context: Any):
        """Adds one context and persists the relationship by OCCT shape identity."""
        if self is None or context is None:
            return self

        shape = _shape_from_topology(self)

        if not _is_null_shape(shape):
            manager = AttributeManager.GetInstance()
            if manager.HasContexts(shape):
                current = manager.GetContexts(shape)
            else:
                current = _safe_list(getattr(self, "contexts", []))

            current = _deduplicate_by_identity(current + [context])
            manager.SetContexts(shape, current)
            self.contexts = manager.GetContexts(shape)
        else:
            self.contexts = _deduplicate_by_identity(
                _safe_list(getattr(self, "contexts", [])) + [context]
            )

        return self

    def RemoveContents(self, contents: Any):
        """Removes contents using topological identity and persists the result."""
        if self is None:
            return self

        to_remove = _safe_list(contents)
        shape = _shape_from_topology(self)

        if not _is_null_shape(shape):
            manager = AttributeManager.GetInstance()
            if manager.HasContents(shape):
                current = manager.GetContents(shape)
            else:
                current = _safe_list(getattr(self, "contents", []))
        else:
            current = _safe_list(getattr(self, "contents", []))

        kept = []
        for item in current:
            remove = False
            for candidate in to_remove:
                if Topology.IsSame(item, candidate):
                    remove = True
                    break

            if not remove:
                kept.append(item)

        kept = _deduplicate_by_identity(kept)
        self.contents = kept

        if not _is_null_shape(shape):
            manager.SetContents(shape, kept)
            self.contents = manager.GetContents(shape)

        return self

    def RemoveCoplanarFaces(
        self,
        epsilon: float = 0.01,
        tolerance: float = 0.0001
    ):
        """
        Removes coplanar faces using OCCT's ShapeUpgrade_UnifySameDomain.

        Parameters
        ----------
        epsilon : float, optional
            The linear tolerance used by OCCT when determining whether
            neighbouring faces lie on the same geometric domain.
            Default is 0.01.
        tolerance : float, optional
            The sewing tolerance used when the input is a shapeless aggregate
            of independently constructed faces. Default is 0.0001.

        Returns
        -------
        Topology
            The topology with adjacent coplanar faces unified.
        """

        type_name = _topology_type_name(self)

        if type_name in ("Vertex", "Edge", "Wire", "Face"):
            return self

        if ShapeUpgrade_UnifySameDomain is None:
            return self

        shape = _shape_from_topology(self)
        temporary_shape = False

        # ------------------------------------------------------------------
        # Shapeless face soup
        #
        # Cluster.ByTopologies intentionally stores shape=None. This is the
        # common case for imported OBJ meshes. The individual faces therefore
        # need to be sewn once so coincident triangle edges become genuine
        # shared OCCT edges before same-domain unification.
        # ------------------------------------------------------------------

        if _is_null_shape(shape):

            members = _safe_list(
                getattr(self, "topologies", None)
            )

            face_soup = (
                len(members) > 0
                and all(
                    _topology_type_name(member) == "Face"
                    for member in members
                )
            )

            if face_soup:
                try:
                    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing

                    sewing = BRepBuilderAPI_Sewing(
                        max(float(tolerance), 1e-12)
                    )

                    count = 0

                    for member in members:
                        member_shape = _shape_from_topology(
                            member
                        )

                        if _is_null_shape(member_shape):
                            continue

                        sewing.Add(
                            member_shape
                        )

                        count += 1

                    if count == 0:
                        return self

                    sewing.Perform()

                    shape = sewing.SewedShape()

                    if _is_null_shape(shape):
                        return self

                    temporary_shape = True

                except Exception:
                    return self

            else:
                # Preserve arbitrary heterogeneous Clusters by constructing
                # an OCCT Compound without discarding any members.
                shape = _ensure_compound_shape(
                    self
                )

                if _is_null_shape(shape):
                    return self

        # ------------------------------------------------------------------
        # Native OCCT same-domain unification.
        #
        # Constructor arguments:
        #
        #   shape
        #   UnifyEdges=False
        #   UnifyFaces=True
        #   ConcatBSplines=False
        #
        # We specifically want face unification. Removing collinear edges or
        # joining curves is a different operation and should not be performed
        # implicitly by RemoveCoplanarFaces.
        # ------------------------------------------------------------------

        try:
            unifier = ShapeUpgrade_UnifySameDomain(
                shape,
                False,
                True,
                False
            )

            if hasattr(
                unifier,
                "SetLinearTolerance"
            ):
                unifier.SetLinearTolerance(
                    max(float(epsilon), 1e-12)
                )

            if hasattr(
                unifier,
                "AllowInternalEdges"
            ):
                unifier.AllowInternalEdges(
                    False
                )

            # The sewn face soup is a temporary shape. OCCT is free to modify
            # it in-place, avoiding unnecessary defensive copying.
            if (
                temporary_shape
                and hasattr(
                    unifier,
                    "SetSafeInputMode"
                )
            ):
                unifier.SetSafeInputMode(
                    False
                )

            unifier.Build()

            unified_shape = unifier.Shape()

            if _is_null_shape(
                unified_shape
            ):
                return self

        except Exception:
            return self

        # ------------------------------------------------------------------
        # Wrap the native result.
        # ------------------------------------------------------------------

        dictionary = Topology.GetDictionary(
            self
        )

        contents = list(
            getattr(
                self,
                "contents",
                []
            ) or []
        )

        contexts = list(
            getattr(
                self,
                "contexts",
                []
            ) or []
        )

        apertures = list(
            getattr(
                self,
                "apertures",
                []
            ) or []
        )

        result = Topology.ByOcctShape(
            unified_shape,
            dictionary=dictionary,
            contents=contents,
            contexts=contexts,
            apertures=apertures
        )

        if result is None:
            return self

        # ------------------------------------------------------------------
        # Simplify a one-face Shell to a Face.
        #
        # ShapeUpgrade_UnifySameDomain preserves the Shell container even when
        # every constituent triangle has collapsed into one OCCT Face. Topologic
        # semantics are cleaner if that result is returned directly as a Face.
        # ------------------------------------------------------------------

        if _topology_type_name(
            result
        ) == "Shell":

            faces = _iter_occ_subshapes_unique(
                unified_shape,
                TopAbs_FACE
            )

            if len(faces) == 1:

                face = Topology.ByOcctShape(
                    faces[0],
                    dictionary=dictionary,
                    contents=contents,
                    contexts=contexts,
                    apertures=apertures
                )

                if face is not None:
                    return face

        return result

    def Distance(
        self,
        otherTopology: Any,
        tolerance: float = 0.0001
    ):
        """
        Returns the shortest Euclidean distance between this topology and the
        input topology.

        For topologies represented by one native OCCT shape, the distance is
        computed directly using BRepExtrema_DistShapeShape.

        For aggregate wrappers without one native OCCT shape, such as a
        heterogeneous Cluster, the constituent native shapes are collected and
        the minimum pairwise distance is returned.

        Parameters
        ----------
        otherTopology : Topology
            The other input topology.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        float
            The shortest distance between the two topologies, or None if the
            distance cannot be computed.
        """

        if self is None or otherTopology is None:
            return None

        if BRepExtrema_DistShapeShape is None:
            return None

        # ------------------------------------------------------------------
        # Retrieve native operand shapes.
        #
        # Ordinary topology:
        #     one native OCCT shape.
        #
        # Aggregate wrapper such as Cluster:
        #     one or more constituent native OCCT shapes.
        # ------------------------------------------------------------------

        def _operand_shapes(topology):

            shape = _shape_from_topology(
                topology
            )

            if not _is_null_shape(
                shape
            ):
                return [
                    shape
                ]

            try:
                shapes = _collect_boolean_operand_shapes(
                    topology
                )
            except Exception:
                shapes = []

            if not isinstance(
                shapes,
                list
            ):
                return []

            return [
                shape
                for shape in shapes
                if not _is_null_shape(
                    shape
                )
            ]

        shapes_a = _operand_shapes(
            self
        )

        shapes_b = _operand_shapes(
            otherTopology
        )

        if (
            len(shapes_a) == 0
            or len(shapes_b) == 0
        ):
            return None

        # ------------------------------------------------------------------
        # Compute exact minimum pairwise distance.
        # ------------------------------------------------------------------

        best_distance = None

        for shape_a in shapes_a:

            for shape_b in shapes_b:

                try:
                    dist = BRepExtrema_DistShapeShape(
                        shape_a,
                        shape_b
                    )

                    if not dist.IsDone():
                        return None

                    value = float(
                        dist.Value()
                    )

                except Exception:
                    return None

                # ----------------------------------------------------------
                # Validate the native result.
                # ----------------------------------------------------------

                if not math.isfinite(
                    value
                ):
                    return None

                if value < -abs(
                    float(tolerance)
                ):
                    return None

                # Small numerical negative or positive values represent zero.
                if abs(value) <= tolerance:
                    return 0.0

                if (
                    best_distance is None
                    or value < best_distance
                ):
                    best_distance = value

        if best_distance is None:
            return None

        return best_distance
        
    @staticmethod
    def IsSame(topologyA: Any, topologyB: Any) -> bool:
        """
        Returns True if topologyA and topologyB refer to the same underlying
        topological entity (identity), not merely geometric coincidence.
        Mirrors OCCT TopoDS_Shape.IsSame semantics when shapes are available.
        """
        if topologyA is None or topologyB is None:
            return False
        if topologyA is topologyB:
            return True

        shape_a = _shape_from_topology(topologyA)
        shape_b = _shape_from_topology(topologyB)
        if not _is_null_shape(shape_a) and not _is_null_shape(shape_b):
            try:
                return bool(shape_a.IsSame(shape_b))
            except Exception:
                pass

        uuid_a = getattr(topologyA, "_uuid", None)
        uuid_b = getattr(topologyB, "_uuid", None)
        if uuid_a is not None and uuid_b is not None:
            return uuid_a == uuid_b

        return False

    def Merge(self, otherTopology: Any = None, transferDictionary: bool = False):
        return _make_occ_merge(
            self,
            otherTopology,
            transfer_dictionary=transferDictionary,
        )

    # Union dissolves the operand interface (-> Cell), unlike Merge which
    # preserves it (-> CellComplex). See _make_occ_union / _make_occ_merge.
    def Union(self, otherTopology: Any, transferDictionary: bool = False):
        result = _make_occ_union(
            self,
            otherTopology,
            transfer_dictionary=transferDictionary,
        )

        if result is None:
            return None

        # Topologic permits connected non-manifold Wires. OCCT Fuse can return
        # such a connected branching 1-D result as a Compound, which the generic
        # wrapper exposes as a Cluster even though the geometry is one valid
        # Topologic Wire. Normalize only the Wire/Wire case, and only when every
        # result Edge can be retained by Wire.ByEdges.
        try:
            from .wire import Wire

            if (
                isinstance(self, Wire)
                and isinstance(otherTopology, Wire)
                and not isinstance(result, Wire)
            ):
                result_edges = result.Edges() or []

                if result_edges:
                    wire = Wire.ByEdges(result_edges)

                    if (
                        wire is not None
                        and len(wire.Edges() or []) == len(result_edges)
                    ):
                        try:
                            wire.dictionary = Topology.GetDictionary(result)
                        except Exception:
                            pass

                        for attr in (
                            "contents",
                            "contexts",
                            "apertures",
                        ):
                            try:
                                setattr(
                                    wire,
                                    attr,
                                    list(getattr(result, attr, []) or [])
                                )
                            except Exception:
                                pass

                        return wire

        except Exception:
            pass

        return result

    def _binary_boolean(
        self,
        otherTopology: Any,
        occt_op_class,
        transferDictionary: bool = False
    ):
        """
        Executes a binary OCCT Boolean operation.

        Parameters
        ----------
        otherTopology : Topology
            The second input topology.
        occt_op_class : type
            The OCCT Boolean operation class.
        transferDictionary : bool , optional
            If True, dictionaries from the input topologies are transferred to
            the result. Default is False.

        Returns
        -------
        Topology
            The resulting topology, or None if the Boolean result is empty or
            the operation fails.
        """
        if occt_op_class is None:
            return None

        shape_a = _shape_from_topology(self)
        shape_b = _shape_from_topology(otherTopology)

        if _is_null_shape(shape_a) or _is_null_shape(shape_b):
            return None

        try:
            op = occt_op_class(shape_a, shape_b)
            op.Build()

            if not op.IsDone():
                return None

            result_shape = op.Shape()

        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None

        # ------------------------------------------------------------------
        # OCCT may return a non-null empty Compound. A Boolean result with no
        # vertices at all is genuinely empty for both Cut and Common.
        # ------------------------------------------------------------------

        if occt_op_class in (BRepAlgoAPI_Cut, BRepAlgoAPI_Common):
            if not _iter_occ_subshapes(result_shape, TopAbs_VERTEX):
                return None

        # ------------------------------------------------------------------
        # Difference cleanup
        #
        # For Cut, Face-or-higher-dimensional operands can leave only
        # lower-dimensional fragments at coincident/shared boundaries.
        # These are not meaningful remnants of the minuend and should be
        # treated as an empty Difference result.
        #
        # IMPORTANT:
        # Do NOT apply this rule to Common/Intersect. A lower-dimensional
        # intersection is legitimate:
        #
        #     Cell ∩ Cell -> Face / Edge / Vertex
        #     Face ∩ Face -> Edge / Vertex
        #
        # and is required by spatial predicates such as Touches.
        # ------------------------------------------------------------------

        if occt_op_class is BRepAlgoAPI_Cut:

            operand_dimension = min(
                Topology._max_shape_dimension(shape_a),
                Topology._max_shape_dimension(shape_b),
            )

            if operand_dimension >= 2:

                solids = _iter_occ_subshapes(
                    result_shape,
                    TopAbs_SOLID
                )

                if not solids:

                    faces = _iter_occ_subshapes(
                        result_shape,
                        TopAbs_FACE
                    )

                    if not faces:
                        return None

                    if Topology._total_face_area(result_shape) <= 1e-9:
                        return None

        # ------------------------------------------------------------------
        # Validate/fix only if required.
        # ------------------------------------------------------------------

        result_shape = _postprocess_boolean_result(
            result_shape
        )

        if _is_null_shape(result_shape):
            return None

        # ------------------------------------------------------------------
        # Dictionaries
        # ------------------------------------------------------------------

        result_dictionary = {}

        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self),
                Topology.GetDictionary(otherTopology)
            )

        return Topology.ByOcctShape(
            result_shape,
            dictionary=result_dictionary
        )

    def Difference(self, otherTopology: Any, transferDictionary: bool = False):
        return self._binary_boolean(otherTopology, BRepAlgoAPI_Cut, transferDictionary)

    def Intersect(
        self,
        otherTopology,
        transferDictionary: bool = False
    ):
        """
        Returns the intersection of this topology and the input topology.

        The primary operation uses BRepAlgoAPI_Common. For selected same-type
        intersections, the result is wrapped in a Cluster to match TopologicCore
        result semantics.

        For Shell/Shell intersections, the raw OCCT Common result is first
        normalized to a Shell before being wrapped in a Cluster. This avoids
        duplicate coincident edge representations.

        If BRepAlgoAPI_Common returns no topology, BRepAlgoAPI_Section is used as
        a fallback to preserve legitimate lower-dimensional contact such as
        face-, edge-, and vertex-only intersections between higher-dimensional
        topologies.

        Parameters
        ----------
        otherTopology : Topology
            The second input topology.
        transferDictionary : bool , optional
            If True, dictionaries from the input topologies are transferred to
            the result. Default is False.

        Returns
        -------
        Topology
            The resulting topology, or None if no intersection exists.
        """

        if not isinstance(otherTopology, Topology):
            return None

        from .cluster import Cluster
        from .face import Face
        from .shell import Shell
        from .cell import Cell
        from .cell_complex import CellComplex

        # ------------------------------------------------------------------
        # 1. Standard Boolean Common
        # ------------------------------------------------------------------

        result = self._binary_boolean(
            otherTopology,
            BRepAlgoAPI_Common,
            transferDictionary
        )

        if result is not None:

            # --------------------------------------------------------------
            # Face / Face
            #
            # The normalized geometry is already correct. TopologicCore
            # exposes this Boolean intersection as a Cluster.
            # --------------------------------------------------------------

            if (
                isinstance(self, Face)
                and isinstance(otherTopology, Face)
            ):
                cluster = Cluster.ByTopologies(
                    [result]
                )

                if cluster is not None:
                    if transferDictionary:
                        cluster.dictionary = _merge_backend_dictionaries(
                            Topology.GetDictionary(self),
                            Topology.GetDictionary(otherTopology)
                        )

                    return cluster

                return result

            # --------------------------------------------------------------
            # Shell / Shell
            #
            # The normal Boolean path can produce a Cluster whose constituent
            # Faces contain duplicate coincident Edge wrappers.
            #
            # Rebuild the raw Common result and normalize that Compound first.
            # This gives the correct Shell topology before wrapping it in a
            # Cluster.
            # --------------------------------------------------------------

            if (
                isinstance(self, Shell)
                and isinstance(otherTopology, Shell)
            ):
                try:
                    shape_a = _shape_from_topology(
                        self
                    )

                    shape_b = _shape_from_topology(
                        otherTopology
                    )

                    if (
                        not _is_null_shape(shape_a)
                        and not _is_null_shape(shape_b)
                    ):
                        common = BRepAlgoAPI_Common(
                            shape_a,
                            shape_b
                        )

                        common.Build()

                        if common.IsDone():
                            common_shape = common.Shape()

                            if not _is_null_shape(common_shape):
                                # The raw Common result is normally a Compound
                                # containing one or more genuine OCCT Shell
                                # subshapes. Wrap those Shells directly.
                                #
                                # Do not normalize the whole Compound through
                                # _wrap_shape_as_topology here: that path can
                                # rebuild a Shell from independent Face wrappers,
                                # causing shared boundary Edges to be represented
                                # multiple times (24 instead of the correct 12
                                # in the regression case).
                                shell_shapes = _iter_occ_subshapes_unique(
                                    common_shape,
                                    TopAbs_SHELL
                                )

                                normalized_shells = []

                                for shell_shape in shell_shapes:
                                    normalized_shell = Topology.ByOcctShape(
                                        shell_shape
                                    )

                                    if normalized_shell is not None:
                                        normalized_shells.append(
                                            normalized_shell
                                        )

                                if normalized_shells:
                                    # Do not use Cluster.ByTopologies here. In
                                    # this specific Boolean result it can create
                                    # another Compound container whose traversal
                                    # re-exposes coincident Shell boundary Edges.
                                    #
                                    # Instead, keep the genuine OCCT Shell shape
                                    # as the Cluster's underlying shape while
                                    # preserving Cluster as the public semantic
                                    # result type required by Topologic.
                                    if len(normalized_shells) == 1:
                                        normalized_shell = normalized_shells[0]

                                        cluster_dictionary = {}

                                        if transferDictionary:
                                            cluster_dictionary = _merge_backend_dictionaries(
                                                Topology.GetDictionary(self),
                                                Topology.GetDictionary(otherTopology)
                                            )

                                        cluster = Cluster(
                                            shape=getattr(normalized_shell, "shape", None),
                                            topologies=[normalized_shell],
                                            dictionary=cluster_dictionary,
                                            contents=[],
                                            contexts=[],
                                            apertures=[]
                                        )

                                        return cluster

                                    cluster = Cluster.ByTopologies(
                                        normalized_shells
                                    )

                                    if cluster is not None:
                                        if transferDictionary:
                                            cluster.dictionary = _merge_backend_dictionaries(
                                                Topology.GetDictionary(self),
                                                Topology.GetDictionary(otherTopology)
                                            )

                                        return cluster

                except Exception:
                    pass

                return result

            # --------------------------------------------------------------
            # Cell / Cell
            #
            # Preserve the correctly normalized Cell but expose it as a
            # Cluster to match TopologicCore Boolean semantics.
            # --------------------------------------------------------------

            if (
                isinstance(self, Cell)
                and isinstance(otherTopology, Cell)
            ):
                cluster = Cluster.ByTopologies(
                    [result]
                )

                if cluster is not None:
                    if transferDictionary:
                        cluster.dictionary = _merge_backend_dictionaries(
                            Topology.GetDictionary(self),
                            Topology.GetDictionary(otherTopology)
                        )

                    return cluster

                return result

            # --------------------------------------------------------------
            # CellComplex / CellComplex
            #
            # IMPORTANT:
            #
            # Do not reconstruct this from the raw BRepAlgoAPI_Common shape.
            # The raw Common result exposes fewer solids than the existing
            # TopologicPy Boolean normalization.
            #
            # Preserve the already-normalized CellComplex result and only
            # wrap it in a Cluster.
            # --------------------------------------------------------------

            if (
                isinstance(self, CellComplex)
                and isinstance(otherTopology, CellComplex)
            ):
                cells = []

                try:
                    result.Cells(
                        None,
                        cells
                    )
                except Exception:
                    cells = []

                if cells:
                    cluster = Cluster.ByTopologies(
                        cells
                    )

                    if cluster is not None:
                        if transferDictionary:
                            cluster.dictionary = _merge_backend_dictionaries(
                                Topology.GetDictionary(self),
                                Topology.GetDictionary(otherTopology)
                            )

                        return cluster

                return result

            # --------------------------------------------------------------
            # All other intersections retain the normal Boolean result.
            # --------------------------------------------------------------

            return result

        # ------------------------------------------------------------------
        # 2. Lower-dimensional contact fallback
        #
        # BRepAlgoAPI_Common returns an empty Compound for cases such as:
        #
        #     Cell / Cell face contact
        #     Cell / Cell edge contact
        #     Cell / Cell vertex contact
        #
        # BRepAlgoAPI_Section preserves those intersections.
        # ------------------------------------------------------------------

        if BRepAlgoAPI_Section is None:
            return None

        shape_a = _shape_from_topology(
            self
        )

        shape_b = _shape_from_topology(
            otherTopology
        )

        if (
            _is_null_shape(shape_a)
            or _is_null_shape(shape_b)
        ):
            return None

        try:
            section = BRepAlgoAPI_Section(
                shape_a,
                shape_b,
                False
            )

            section.Build()

            if not section.IsDone():
                return None

            section_shape = section.Shape()

        except Exception:
            return None

        if _is_null_shape(section_shape):
            return None

        # ------------------------------------------------------------------
        # Section can return a non-null but empty Compound.
        # A genuine intersection must contain at least one Vertex.
        # ------------------------------------------------------------------

        vertices = _iter_occ_subshapes_unique(
            section_shape,
            TopAbs_VERTEX
        )

        if not vertices:
            return None

        # ------------------------------------------------------------------
        # Normalize the Section result.
        # ------------------------------------------------------------------

        section_result = Topology.ByOcctShape(
            section_shape
        )

        if section_result is None:
            return None

        # ------------------------------------------------------------------
        # Dictionary transfer.
        # ------------------------------------------------------------------

        if transferDictionary:
            dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self),
                Topology.GetDictionary(otherTopology)
            )

            try:
                section_result = Topology.SetDictionary(
                    section_result,
                    dictionary
                )
            except Exception:
                section_result.dictionary = dictionary

        return section_result
    def XOR(self, otherTopology: Any, transferDictionary: bool = False):
        a_minus_b = self._binary_boolean(otherTopology, BRepAlgoAPI_Cut, False)
        b_minus_a = Topology._binary_boolean(otherTopology, self, BRepAlgoAPI_Cut, False)
        if a_minus_b is None or b_minus_a is None:
            return None
        # symdif = the two disjoint caps of material. Wrapping them as a
        # Cluster (not via Merge, which would promote to a CellComplex) keeps
        # the result type faithful to topologic_core's symdif->Cluster contract.
        from topologicpy.Cluster import Cluster
        return Cluster.ByTopologies([a_minus_b, b_minus_a])

    @staticmethod
    def _max_shape_dimension(shape: Any) -> int:
        """
        Highest topological dimension present in ``shape``: 0=Vertex,
        1=Edge/Wire, 2=Face/Shell, 3=Solid/CompSolid, -1=null/unknown.
        For an aggregate (Compound/Cluster) this is the highest dimension
        found among its subshapes, since a heterogeneous cluster has no
        single ShapeType of its own.
        """
        if _is_null_shape(shape):
            return -1
        try:
            shape_type = shape.ShapeType()
        except Exception:
            return -1
        if shape_type in (TopAbs_SOLID, TopAbs_COMPSOLID):
            return 3
        if shape_type in (TopAbs_SHELL, TopAbs_FACE):
            return 2
        if shape_type in (TopAbs_WIRE, TopAbs_EDGE):
            return 1
        if shape_type == TopAbs_VERTEX:
            return 0
        if _iter_occ_subshapes(shape, TopAbs_SOLID):
            return 3
        if _iter_occ_subshapes(shape, TopAbs_FACE):
            return 2
        if _iter_occ_subshapes(shape, TopAbs_EDGE):
            return 1
        if _iter_occ_subshapes(shape, TopAbs_VERTEX):
            return 0
        return -1

    @staticmethod
    def _total_face_area(shape: Any) -> float:
        """Sum of the surface area of every FACE subshape of ``shape``."""
        if _is_null_shape(shape) or brepgprop is None or GProp_GProps is None:
            return 0.0
        try:
            total = 0.0
            for face_shape in _iter_occ_subshapes(shape, TopAbs_FACE):
                props = GProp_GProps()
                brepgprop.SurfaceProperties(face_shape, props)
                total += props.Mass()
            return total
        except Exception:
            return 0.0

    @staticmethod
    def _piece_belongs_to_any(piece_shape, reference_shapes) -> bool:
        """
        Tell which fuse fragment came from self's material: Solid/Shell references classify by
        centre-of-mass (BRepClass3d_SolidClassifier); Face references use the 2D surface/UV
        FClass2d test, since SolidClassifier is undefined for a bare face.
        """
        if GProp_GProps is None or brepgprop is None:
            return False
        try:
            props = GProp_GProps()
            if piece_shape.ShapeType() == TopAbs_FACE:
                brepgprop.SurfaceProperties(piece_shape, props)
            else:
                brepgprop.VolumeProperties(piece_shape, props)
            center = props.CentreOfMass()
        except Exception:
            return False

        for reference in reference_shapes:
            try:
                if reference.ShapeType() == TopAbs_FACE:
                    if Topology._point_in_face_shape(reference, center, 1e-6):
                        return True
                    continue
                if BRepClass3d_SolidClassifier is None:
                    continue
                classifier = BRepClass3d_SolidClassifier(reference)
                classifier.Perform(center, 1e-6)
                state = classifier.State()
                if state in (_TopAbs_IN, _TopAbs_ON):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _point_in_face_shape(occ_face, pnt, tolerance: float = 0.0001) -> bool:
        """Raw-shape point-in-face test (see _piece_belongs_to_any)."""
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            from OCC.Core.gp import gp_Pnt2d
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

            face = topods.Face(occ_face)
            surface = BRep_Tool.Surface(face)
            if surface is None:
                return False

            projector = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if projector.NbPoints() < 1:
                return False
            if projector.LowerDistance() > tolerance:
                return False
            u_raw, v_raw = projector.LowerDistanceParameters()

            classifier = BRepTopAdaptor_FClass2d(face, tolerance)
            state = classifier.Perform(gp_Pnt2d(u_raw, v_raw))
            return state in (TopAbs_IN, TopAbs_ON)
        except Exception:
            return False

    def _partition_by(self, otherTopology: Any, transferDictionary: bool = False, promote: bool = True):
        """
        BOPAlgo_CellsBuilder partition shared by Divide/Slice/Impose/Imprint. Use
        AddAllToResult() -- AddToResult(original) and Modified()/IsDeleted() both fail to
        track Solid-level splits -- then keep only fragments whose centre of mass classifies
        on/in self's original shapes. promote=True (Slice/Imprint/Divide) forces a CellComplex;
        False (Impose) preserves disjoint operands as a Cluster.
        """
        if BOPAlgo_CellsBuilder is None:
            return None
        shapes_a = _collect_boolean_operand_shapes(self)
        shapes_b = _collect_boolean_operand_shapes(otherTopology)
        if not shapes_a or not shapes_b:
            return None
        try:
            builder = BOPAlgo_CellsBuilder()
            for shape in shapes_a:
                builder.AddArgument(shape)
            for shape in shapes_b:
                builder.AddArgument(shape)
            builder.Perform()
            if hasattr(builder, "HasErrors") and builder.HasErrors():
                return None

            builder.AddAllToResult()
            all_result_shape = builder.Shape()
        except Exception:
            return None

        if _is_null_shape(all_result_shape):
            return None

        target_type = TopAbs_SOLID
        if all(not _is_null_shape(s) and s.ShapeType() == TopAbs_FACE for s in shapes_a):
            target_type = TopAbs_FACE

        pieces = _iter_occ_subshapes(all_result_shape, target_type)
        kept = [p for p in pieces if Topology._piece_belongs_to_any(p, shapes_a)]
        if not kept:
            return None

        try:
            empty_avoid = TopTools_ListOfShape()
            for piece in kept:
                to_take = TopTools_ListOfShape()
                to_take.Append(piece)
                builder.AddToResult(to_take, empty_avoid)
            builder.MakeContainers()
            result_shape = builder.Shape()
        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)
        if promote:
            result_shape = _promote_to_compsolid_if_multi_solid(result_shape, force=True)

        result_dictionary = {}
        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )
        return Topology.ByOcctShape(result_shape, dictionary=result_dictionary)

    def Divide(self, otherTopology: Any, transferDictionary: bool = False):
        return self._partition_by(otherTopology, transferDictionary)

    def _split_by_tool(self, otherTopology: Any, transferDictionary: bool = False):
        """
        Split self by otherTopology keeping only self's fragments (Slice/Imprint semantics)
        via BOPAlgo_Splitter(Arguments=[self], Tools=[other]). Verified exact vs core
        subtopology counts; the manual CellsBuilder+centroid path over-included fragments.
        """
        if BOPAlgo_Splitter is None:
            return None
        # Use _collect_boolean_operand_shapes, not a bare _shape_from_topology
        # lookup: aggregate wrappers (Cluster of cutting faces, CellComplex
        # with no single unified .shape) have no top-level .shape at all, so
        # a naive lookup would report them as null and abort. This matches
        # CellComplex.Prism's internal Topology.Slice(cell, Cluster.ByTopologies(
        # cutting faces)) call, which fed a Cluster tool with .shape=None.
        shapes_a = _collect_boolean_operand_shapes(self)
        shapes_b = _collect_boolean_operand_shapes(otherTopology)
        if not shapes_a or not shapes_b:
            return None
        try:
            splitter = BOPAlgo_Splitter()
            for shape in shapes_a:
                splitter.AddArgument(shape)
            for shape in shapes_b:
                splitter.AddTool(shape)
            splitter.SetFuzzyValue(1e-4)
            splitter.Perform()
            if hasattr(splitter, "HasErrors") and splitter.HasErrors():
                return None
            result_shape = splitter.Shape()
        except Exception:
            return None
        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)
        result_shape = _promote_to_compsolid_if_multi_solid(result_shape, force=True)

        result_dictionary = {}
        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )
        return Topology.ByOcctShape(result_shape, dictionary=result_dictionary)

    def Slice(self, otherTopology: Any, transferDictionary: bool = False):
        return self._split_by_tool(otherTopology, transferDictionary)

    @staticmethod
    def _collect_impose_operand_shapes(topology: Any) -> list:
        """
        Always decompose container types (Wire/Shell/CellComplex/Cluster) into their immediate
        sub-shapes, unlike _collect_boolean_operand_shapes: Impose needs one AddToResult (and
        one material index) per original operand shape, so a merged COMPSOLID would collapse
        separate tool-material groups and lose a cell/face.
        """
        if topology is None:
            return []
        from .wire import Wire as _Wire
        from .shell import Shell as _Shell
        from .cell_complex import CellComplex as _CellComplex
        from .cluster import Cluster as _Cluster

        if isinstance(topology, _Wire):
            children = topology.Edges() or []

        elif isinstance(topology, _Shell):
            children = topology.Faces() or []

        elif isinstance(topology, _CellComplex):
            children = topology.Cells() or []

        elif isinstance(topology, _Cluster):
            children = getattr(topology, "topologies", None) or []

        else:
            children = None

        if children:
            result = []
            for child in children:
                result.extend(Topology._collect_impose_operand_shapes(child))
            return _deduplicate_by_identity(result) if result else []

        shape = _shape_from_topology(topology)
        if _is_null_shape(shape):
            return []
        return [shape]

    def Impose(self, otherTopology: Any, transferDictionary: bool = False):
        """
        Port of topologic_core Topology::Impose: CellsBuilder over both operands; one
        AddToResult per self-operand shape keeping its exclusive material, then per tool
        shape keeping ALL its fragments tagged with a distinct material index (so
        MakeContainers remerges a tool's own fragments but not different tools). Verified
        count-parity against real topologic_core across all topology types.
        """
        if BOPAlgo_CellsBuilder is None:
            return None
        shapes_a = Topology._collect_impose_operand_shapes(self)
        shapes_b = Topology._collect_impose_operand_shapes(otherTopology)
        if not shapes_a or not shapes_b:
            return None
        try:
            builder = BOPAlgo_CellsBuilder()
            args = TopTools_ListOfShape()
            for shape in shapes_a + shapes_b:
                args.Append(shape)
            builder.SetArguments(args)
            builder.Perform()
            if builder.HasErrors():
                return None

            for a in shapes_a:
                to_take = TopTools_ListOfShape()
                to_take.Append(a)
                to_avoid = TopTools_ListOfShape()
                for b in shapes_b:
                    to_avoid.Append(b)
                builder.AddToResult(to_take, to_avoid)

            for i, b in enumerate(shapes_b, start=1):
                to_take = TopTools_ListOfShape()
                to_take.Append(b)
                to_avoid = TopTools_ListOfShape()
                builder.AddToResult(to_take, to_avoid, i, True)

            builder.MakeContainers()
            result_shape = builder.Shape()
        except Exception:
            return None
        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)

        result_dictionary = {}
        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )
        return Topology.ByOcctShape(result_shape, dictionary=result_dictionary)

    def Imprint(self, otherTopology: Any, transferDictionary: bool = False):
        return self._split_by_tool(otherTopology, transferDictionary)

    # -------------------------------------------------------------------
    # Transform / Translate / Rotate / Scale
    # -------------------------------------------------------------------

    @staticmethod
    def _apply_transform_to_members(topology: Any, apply_one) -> Any:
        """
        Applies a transformation recursively to the direct members of an
        aggregate topology and rebuilds the same aggregate type.

        This is used for heterogeneous Clusters, for which transforming the
        complete OCCT Compound as one shape can alter the intermediate topology
        hierarchy, and as a fallback for aggregate wrappers without a usable
        native OCCT shape.

        Parameters
        ----------
        topology : Topology
            The input aggregate topology.
        apply_one : callable
            A callable that accepts one topology and returns its transformed
            topology.

        Returns
        -------
        Topology
            The transformed aggregate topology, or None if the transformation
            or reconstruction fails.
        """
        type_name = _topology_type_name(
            topology
        )

        # ------------------------------------------------------------------
        # Cluster
        #
        # Always preserve the actual DIRECT constituent hierarchy.
        #
        # Cluster.ByTopologies intentionally stores a native OCCT Compound, so
        # topology.topologies may be empty. Retrieve the direct children through
        # Cluster.Topologies() first.
        # ------------------------------------------------------------------

        if type_name == "Cluster":

            from .cluster import Cluster

            members = []

            try:
                members = topology.Topologies()
            except Exception:
                members = []

            if not isinstance(
                members,
                list
            ):
                members = []

            if len(members) == 0:
                try:
                    members = list(
                        getattr(
                            topology,
                            "topologies",
                            []
                        )
                        or []
                    )
                except Exception:
                    members = []

            members = [
                member
                for member in members
                if isinstance(
                    member,
                    Topology
                )
            ]

            if len(members) == 0:
                return None

            transformed = []

            for member in members:

                try:
                    transformed_member = apply_one(
                        member
                    )
                except Exception:
                    transformed_member = None

                # Do not silently drop a failed constituent.
                if not isinstance(
                    transformed_member,
                    Topology
                ):
                    return None

                transformed.append(
                    transformed_member
                )

            if len(transformed) != len(members):
                return None

            try:
                result = Cluster.ByTopologies(
                    transformed
                )
            except Exception:
                result = None

            if not isinstance(
                result,
                Cluster
            ):
                return None

            # --------------------------------------------------------------
            # Preserve aggregate metadata.
            # --------------------------------------------------------------

            try:
                result.dictionary = Topology.GetDictionary(
                    topology
                )
            except Exception:
                pass

            try:
                result.contents = list(
                    getattr(
                        topology,
                        "contents",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            try:
                result.contexts = list(
                    getattr(
                        topology,
                        "contexts",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            try:
                result.apertures = list(
                    getattr(
                        topology,
                        "apertures",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            return result

        # ------------------------------------------------------------------
        # CellComplex fallback
        #
        # Normally a shaped CellComplex is transformed natively as one OCCT
        # CompSolid. This branch is primarily for wrappers without a usable
        # shape.
        # ------------------------------------------------------------------

        if type_name == "CellComplex":

            from .cell_complex import CellComplex

            cells = []

            try:
                cells = list(
                    getattr(
                        topology,
                        "cells",
                        []
                    )
                    or []
                )
            except Exception:
                cells = []

            if len(cells) == 0:
                try:
                    cells = topology.Cells()
                except Exception:
                    cells = []

            cells = [
                cell
                for cell in cells
                if isinstance(
                    cell,
                    Topology
                )
            ]

            if len(cells) == 0:
                return None

            transformed = []

            for cell in cells:

                try:
                    transformed_cell = apply_one(
                        cell
                    )
                except Exception:
                    transformed_cell = None

                if not isinstance(
                    transformed_cell,
                    Topology
                ):
                    return None

                transformed.append(
                    transformed_cell
                )

            if len(transformed) != len(cells):
                return None

            try:
                result = CellComplex.ByCells(
                    transformed
                )
            except Exception:
                result = None

            if result is None:
                return None

            try:
                result.dictionary = Topology.GetDictionary(
                    topology
                )
            except Exception:
                pass

            try:
                result.contents = list(
                    getattr(
                        topology,
                        "contents",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            try:
                result.contexts = list(
                    getattr(
                        topology,
                        "contexts",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            try:
                result.apertures = list(
                    getattr(
                        topology,
                        "apertures",
                        []
                    )
                    or []
                )
            except Exception:
                pass

            return result

        return None


    @staticmethod
    def _rewrap_preserving_wrapper(result: Any, original: Any) -> Any:
        """
        Preserves a CellComplex wrapper across transforms even when the underlying
        transformed OCCT result contains only one Solid.
        """
        if result is None:
            return None

        try:
            from .cell import Cell
            from .cell_complex import CellComplex

            if not isinstance(original, CellComplex):
                return result

            if isinstance(result, CellComplex):
                return result

            if isinstance(result, Cell):
                return CellComplex(
                    shape=getattr(result, "shape", None),
                    cells=[],
                    dictionary=Topology.GetDictionary(original),
                    contents=list(getattr(original, "contents", []) or []),
                    contexts=list(getattr(original, "contexts", []) or []),
                    apertures=list(getattr(original, "apertures", []) or []),
                )

        except Exception:
            pass

        return result

    def _apply_gtrsf(
        self,
        gtrsf,
        dictionary_passthrough: bool = True
    ):
        """
        Applies a gp_GTrsf general affine transformation to the topology.

        Heterogeneous Clusters are transformed constituent-by-constituent so
        that their direct topology hierarchy is preserved. Other topologies are
        transformed natively as a single OCCT shape whenever possible.

        Parameters
        ----------
        gtrsf : OCC.Core.gp.gp_GTrsf
            The general affine transformation.
        dictionary_passthrough : bool , optional
            If set to True, the topology dictionary is preserved.
            Default is True.

        Returns
        -------
        Topology
            The transformed topology, or None if the transformation fails.
        """

        # ------------------------------------------------------------------
        # Heterogeneous Cluster
        #
        # Do not transform the complete mixed-dimensional Compound in one
        # BRepBuilderAPI_GTransform operation. Doing so can alter intermediate
        # containers such as the Shell inside a Cell.
        #
        # Transform the direct constituent topologies independently and rebuild
        # the Cluster instead.
        # ------------------------------------------------------------------

        if _topology_type_name(
            self
        ) == "Cluster":

            return Topology._apply_transform_to_members(
                self,
                lambda member: Topology._apply_gtrsf(
                    member,
                    gtrsf,
                    dictionary_passthrough
                )
            )

        # ------------------------------------------------------------------
        # Native OCCT affine transformation
        # ------------------------------------------------------------------

        shape = _shape_from_topology(
            self
        )

        if (
            not _is_null_shape(shape)
            and BRepBuilderAPI_GTransform is not None
        ):

            try:
                maker = BRepBuilderAPI_GTransform(
                    shape,
                    gtrsf,
                    True
                )

                if maker.IsDone():

                    new_shape = maker.Shape()

                    if not _is_null_shape(
                        new_shape
                    ):

                        result = Topology.ByOcctShape(
                            new_shape
                        )

                        if result is not None:

                            if dictionary_passthrough:
                                try:
                                    result.dictionary = Topology.GetDictionary(
                                        self
                                    )
                                except Exception:
                                    pass

                            result = Topology._rewrap_preserving_wrapper(
                                result,
                                self
                            )

                            return result

            except Exception:
                pass

        # ------------------------------------------------------------------
        # Aggregate fallback
        # ------------------------------------------------------------------

        return Topology._apply_transform_to_members(
            self,
            lambda member: Topology._apply_gtrsf(
                member,
                gtrsf,
                dictionary_passthrough
            )
        )

    def Translate(self, x: float, y: float, z: float):
        if self is None:
            return None
        try:
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(float(x), float(y), float(z)))
        except Exception:
            return None
        return Topology._apply_rigid(self, trsf)

    def Rotate(self, origin: Any, x: float, y: float, z: float, angle: float):
        if self is None:
            return None
        try:
            ox, oy, oz = origin.x, origin.y, origin.z
            axis = gp_Ax1(gp_Pnt(float(ox), float(oy), float(oz)), gp_Dir(float(x), float(y), float(z)))
            trsf = gp_Trsf()
            trsf.SetRotation(axis, math.radians(float(angle)))
        except Exception:
            return None
        return Topology._apply_rigid(self, trsf)

    def Scale(self, origin: Any, x: float, y: float, z: float):
        if self is None:
            return None
        try:
            ox, oy, oz = origin.x, origin.y, origin.z
            gtrsf = gp_GTrsf()
            mat = gp_Mat(float(x), 0.0, 0.0, 0.0, float(y), 0.0, 0.0, 0.0, float(z))
            gtrsf.SetVectorialPart(mat)
            gtrsf.SetTranslationPart(gp_XYZ(
                ox - float(x) * ox, oy - float(y) * oy, oz - float(z) * oz
            ))
        except Exception:
            return None
        return Topology._apply_gtrsf(self, gtrsf)

    def Transform(self, *args):
        """
        Accepts either a single 4x4 (row-major) matrix, a flat 16-value list,
        or 12 scalar args (tx, ty, tz, r11..r33) as used by EnergyModel.py.
        """
        if self is None:
            return None
        try:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                flat_or_nested = args[0]
                if len(flat_or_nested) == 4 and isinstance(flat_or_nested[0], (list, tuple)):
                    m = [v for row in flat_or_nested for v in row]
                else:
                    m = list(flat_or_nested)
                tx, ty, tz = m[3], m[7], m[11]
                a00, a01, a02 = m[0], m[1], m[2]
                a10, a11, a12 = m[4], m[5], m[6]
                a20, a21, a22 = m[8], m[9], m[10]
            elif len(args) == 12:
                tx, ty, tz, a00, a01, a02, a10, a11, a12, a20, a21, a22 = args
            else:
                return None

            gtrsf = gp_GTrsf()
            mat = gp_Mat(float(a00), float(a01), float(a02),
                         float(a10), float(a11), float(a12),
                         float(a20), float(a21), float(a22))
            gtrsf.SetVectorialPart(mat)
            gtrsf.SetTranslationPart(gp_XYZ(float(tx), float(ty), float(tz)))
        except Exception:
            return None
        return Topology._apply_gtrsf(self, gtrsf)

    @staticmethod
    def _apply_rigid(
        topology: Any,
        trsf
    ) -> Any:
        """
        Applies a gp_Trsf rigid transformation and rebuilds the topology.

        Heterogeneous Clusters are transformed constituent-by-constituent so
        that their direct topology hierarchy is preserved. Other topologies are
        transformed natively as a single OCCT shape whenever possible.

        Parameters
        ----------
        topology : Topology
            The input topology.
        trsf : OCC.Core.gp.gp_Trsf
            The rigid transformation.

        Returns
        -------
        Topology
            The transformed topology, or None if the transformation fails.
        """

        # ------------------------------------------------------------------
        # Heterogeneous Cluster
        # ------------------------------------------------------------------

        if _topology_type_name(
            topology
        ) == "Cluster":

            return Topology._apply_transform_to_members(
                topology,
                lambda member: Topology._apply_rigid(
                    member,
                    trsf
                )
            )

        # ------------------------------------------------------------------
        # Native OCCT rigid transformation
        # ------------------------------------------------------------------

        shape = _shape_from_topology(
            topology
        )

        if (
            not _is_null_shape(shape)
            and BRepBuilderAPI_Transform is not None
        ):

            try:
                maker = BRepBuilderAPI_Transform(
                    shape,
                    trsf,
                    True
                )

                if maker.IsDone():

                    new_shape = maker.Shape()

                    if not _is_null_shape(
                        new_shape
                    ):

                        result = Topology.ByOcctShape(
                            new_shape
                        )

                        if result is not None:

                            try:
                                result.dictionary = Topology.GetDictionary(
                                    topology
                                )
                            except Exception:
                                pass

                            result = Topology._rewrap_preserving_wrapper(
                                result,
                                topology
                            )

                            return result

            except Exception:
                pass

        # ------------------------------------------------------------------
        # Aggregate fallback
        # ------------------------------------------------------------------

        return Topology._apply_transform_to_members(
            topology,
            lambda member: Topology._apply_rigid(
                member,
                trsf
            )
        )

    # -------------------------------------------------------------------
    # Analysis / copy / mass properties
    # -------------------------------------------------------------------

    def GetOcctShape(self):
        return _shape_from_topology(self)

    def Analyze(self):
        type_name = _topology_type_name(self) or "Unknown"
        counts = {}
        for label, getter in (
            ("Vertices", "Vertices"), ("Edges", "Edges"), ("Wires", "Wires"),
            ("Faces", "Faces"), ("Shells", "Shells"), ("Cells", "Cells"),
        ):
            try:
                counts[label] = len(getattr(Topology, getter)(self) or [])
            except Exception:
                counts[label] = 0
        return f"{type_name}: " + ", ".join(f"{k}={v}" for k, v in counts.items())

    def Cleanup(self):
        shape = _shape_from_topology(self)
        if _is_null_shape(shape) or ShapeUpgrade_UnifySameDomain is None:
            return self
        try:
            unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
            unifier.Build()
            new_shape = unifier.Shape()
            if _is_null_shape(new_shape):
                return self
            result = Topology.ByOcctShape(new_shape)
            if result is None:
                return self
            result.dictionary = Topology.GetDictionary(self)
            return result
        except Exception:
            return self

    def Copy(self):
        """
        Creates an independent copy of the topology.

        Native OCCT topologies are copied using BRepBuilderAPI_Copy. Aggregate
        wrappers without a native OCCT shape, such as a lightweight Cluster,
        are copied recursively constituent-by-constituent.

        The topology's parent dictionary is deep-copied. A native copy failure
        is returned as None rather than being concealed by wrapping the original
        OCCT shape.

        Returns
        -------
        Topology
            An independent copy of the topology, or None if copying fails.
        """

        if self is None:
            return None

        type_name = _topology_type_name(
            self
        )

        shape = _shape_from_topology(
            self
        )

        # ------------------------------------------------------------------
        # Aggregate wrappers without one native OCCT shape.
        # ------------------------------------------------------------------

        if _is_null_shape(shape):

            # --------------------------------------------------------------
            # Cluster
            # --------------------------------------------------------------

            if type_name == "Cluster":

                from .cluster import Cluster

                members = []

                try:
                    members = self.Topologies()
                except Exception:
                    members = []

                if not isinstance(
                    members,
                    list
                ):
                    members = []

                if len(members) == 0:
                    try:
                        members = list(
                            getattr(
                                self,
                                "topologies",
                                []
                            )
                            or []
                        )
                    except Exception:
                        members = []

                if len(members) == 0:
                    return None

                copied_members = []

                for member in members:

                    if not isinstance(
                        member,
                        Topology
                    ):
                        return None

                    try:
                        copied_member = member.Copy()
                    except Exception:
                        copied_member = None

                    if not isinstance(
                        copied_member,
                        Topology
                    ):
                        return None

                    copied_members.append(
                        copied_member
                    )

                try:
                    result = Cluster.ByTopologies(
                        copied_members
                    )
                except Exception:
                    result = None

                if not isinstance(
                    result,
                    Cluster
                ):
                    return None

                try:
                    result.dictionary = copy.deepcopy(
                        Topology.GetDictionary(
                            self
                        )
                    )
                except Exception:
                    pass

                return result

            # --------------------------------------------------------------
            # Shapeless CellComplex fallback.
            # --------------------------------------------------------------

            if type_name == "CellComplex":

                from .cell_complex import CellComplex

                try:
                    cells = list(
                        getattr(
                            self,
                            "cells",
                            []
                        )
                        or []
                    )
                except Exception:
                    cells = []

                if len(cells) == 0:
                    try:
                        cells = self.Cells()
                    except Exception:
                        cells = []

                if not isinstance(
                    cells,
                    list
                ) or len(cells) == 0:
                    return None

                copied_cells = []

                for cell in cells:

                    if not isinstance(
                        cell,
                        Topology
                    ):
                        return None

                    try:
                        copied_cell = cell.Copy()
                    except Exception:
                        copied_cell = None

                    if not isinstance(
                        copied_cell,
                        Topology
                    ):
                        return None

                    copied_cells.append(
                        copied_cell
                    )

                try:
                    result = CellComplex.ByCells(
                        copied_cells
                    )
                except Exception:
                    result = None

                if not isinstance(
                    result,
                    CellComplex
                ):
                    return None

                try:
                    result.dictionary = copy.deepcopy(
                        Topology.GetDictionary(
                            self
                        )
                    )
                except Exception:
                    pass

                return result

            # A native topology with no native shape should not silently
            # pretend that it has been copied.
            return None

        # ------------------------------------------------------------------
        # Native OCCT copy.
        # ------------------------------------------------------------------

        if BRepBuilderAPI_Copy is None:
            return None

        try:
            copier = BRepBuilderAPI_Copy(
                shape
            )

            copied_shape = copier.Shape()

        except Exception:
            return None

        if _is_null_shape(
            copied_shape
        ):
            return None

        # ------------------------------------------------------------------
        # Wrap copied native topology.
        # ------------------------------------------------------------------

        try:
            result = Topology.ByOcctShape(
                copied_shape
            )
        except Exception:
            result = None

        if result is None:
            return None

        # Preserve explicit wrapper semantics where OCCT normalization may
        # otherwise unwrap a single-member container.
        try:
            result = Topology._rewrap_preserving_wrapper(
                result,
                self
            )
        except Exception:
            pass

        if not isinstance(
            result,
            Topology
        ):
            return None

        # ------------------------------------------------------------------
        # Copy must preserve its Topologic type.
        # ------------------------------------------------------------------

        if _topology_type_name(
            result
        ) != type_name:
            return None

        # ------------------------------------------------------------------
        # Preserve the parent dictionary independently.
        # ------------------------------------------------------------------

        try:
            result.dictionary = copy.deepcopy(
                Topology.GetDictionary(
                    self
                )
            )
        except Exception:
            pass

        return result

    def DeepCopy(self):
        """
        Creates an independent deep copy of the topology.

        The native OCCT topology is duplicated using BRepBuilderAPI_Copy.
        Dictionaries attached to the topology and all of its native
        subtopologies are transferred to their exact copied counterparts using
        the OCCT source-to-copy mapping provided by ModifiedShape.

        Aggregate wrappers without a single native OCCT shape, such as the
        lightweight Cluster, are copied recursively constituent-by-constituent.

        Returns
        -------
        Topology
            An independent deep copy of the topology, or None if copying fails.
        """

        if self is None:
            return None

        type_name = _topology_type_name(
            self
        )

        shape = _shape_from_topology(
            self
        )

        # ------------------------------------------------------------------
        # Shapeless aggregate wrappers
        # ------------------------------------------------------------------

        if _is_null_shape(shape):

            # --------------------------------------------------------------
            # Cluster
            # --------------------------------------------------------------

            if type_name == "Cluster":

                from .cluster import Cluster

                try:
                    members = self.Topologies()
                except Exception:
                    members = []

                if not isinstance(
                    members,
                    list
                ):
                    members = []

                if len(members) == 0:

                    try:
                        members = list(
                            getattr(
                                self,
                                "topologies",
                                []
                            )
                            or []
                        )
                    except Exception:
                        members = []

                if len(members) == 0:
                    return None

                copied_members = []

                for member in members:

                    if not isinstance(
                        member,
                        Topology
                    ):
                        return None

                    try:
                        copied_member = member.DeepCopy()
                    except Exception:
                        copied_member = None

                    if not isinstance(
                        copied_member,
                        Topology
                    ):
                        return None

                    copied_members.append(
                        copied_member
                    )

                try:
                    result = Cluster.ByTopologies(
                        copied_members
                    )
                except Exception:
                    result = None

                if not isinstance(
                    result,
                    Cluster
                ):
                    return None

                try:
                    result.dictionary = copy.deepcopy(
                        Topology.GetDictionary(
                            self
                        )
                    )
                except Exception:
                    pass

                return result

            # --------------------------------------------------------------
            # Shapeless CellComplex
            # --------------------------------------------------------------

            if type_name == "CellComplex":

                from .cell_complex import CellComplex

                try:
                    cells = list(
                        getattr(
                            self,
                            "cells",
                            []
                        )
                        or []
                    )
                except Exception:
                    cells = []

                if len(cells) == 0:

                    try:
                        cells = self.Cells()
                    except Exception:
                        cells = []

                if not isinstance(
                        cells,
                        list
                ) or len(cells) == 0:
                    return None

                copied_cells = []

                for cell in cells:

                    if not isinstance(
                        cell,
                        Topology
                    ):
                        return None

                    try:
                        copied_cell = cell.DeepCopy()
                    except Exception:
                        copied_cell = None

                    if not isinstance(
                        copied_cell,
                        Topology
                    ):
                        return None

                    copied_cells.append(
                        copied_cell
                    )

                try:
                    result = CellComplex.ByCells(
                        copied_cells
                    )
                except Exception:
                    result = None

                if not isinstance(
                        result,
                        CellComplex
                ):
                    return None

                try:
                    result.dictionary = copy.deepcopy(
                        Topology.GetDictionary(
                            self
                        )
                    )
                except Exception:
                    pass

                return result

            return None

        # ------------------------------------------------------------------
        # Native OCCT deep copy
        #
        # copyGeom=True:
        #     duplicate underlying geometry rather than sharing it.
        #
        # copyMesh=True:
        #     duplicate any existing triangulation as well.
        # ------------------------------------------------------------------

        if BRepBuilderAPI_Copy is None:
            return None

        try:
            copier = BRepBuilderAPI_Copy(
                shape,
                True,
                True
            )

            copied_shape = copier.Shape()

        except Exception:
            return None

        if _is_null_shape(
            copied_shape
        ):
            return None

        # ------------------------------------------------------------------
        # Transfer dictionaries for the root shape and every subshape.
        #
        # AttributeManager stores dictionaries against native OCCT shape
        # identity. ModifiedShape gives the exact copied counterpart of each
        # original subshape.
        # ------------------------------------------------------------------

        manager = AttributeManager.GetInstance()

        def _transfer_shape_dictionary(
            source_shape,
            target_shape
        ):

            if (
                _is_null_shape(source_shape)
                or _is_null_shape(target_shape)
            ):
                return

            try:

                if not manager.HasDictionary(
                    source_shape
                ):
                    return

                source_dictionary = manager.GetDictionary(
                    source_shape
                )

                manager.SetDictionary(
                    target_shape,
                    copy.deepcopy(
                        source_dictionary
                    )
                )

            except Exception:
                pass

        # Root dictionary.
        _transfer_shape_dictionary(
            shape,
            copied_shape
        )

        # ------------------------------------------------------------------
        # Native subtopology dictionaries.
        # ------------------------------------------------------------------

        shape_types = (
            TopAbs_VERTEX,
            TopAbs_EDGE,
            TopAbs_WIRE,
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_COMPSOLID,
        )

        for shape_type in shape_types:

            try:
                source_subshapes = _iter_occ_subshapes_unique(
                    shape,
                    shape_type
                )
            except Exception:
                source_subshapes = []

            for source_subshape in source_subshapes:

                # No dictionary means there is nothing to transfer.
                try:
                    if not manager.HasDictionary(
                        source_subshape
                    ):
                        continue
                except Exception:
                    continue

                # ----------------------------------------------------------
                # Ask OCCT for the exact corresponding copied subshape.
                # ----------------------------------------------------------

                try:
                    copied_subshape = copier.ModifiedShape(
                        source_subshape
                    )
                except Exception:
                    copied_subshape = None

                if _is_null_shape(
                    copied_subshape
                ):
                    continue

                _transfer_shape_dictionary(
                    source_subshape,
                    copied_subshape
                )

        # ------------------------------------------------------------------
        # Wrap copied native topology.
        # ------------------------------------------------------------------

        try:
            result = Topology.ByOcctShape(
                copied_shape
            )
        except Exception:
            result = None

        if result is None:
            return None

        # Preserve wrapper semantics such as CellComplex where required.
        try:
            result = Topology._rewrap_preserving_wrapper(
                result,
                self
            )
        except Exception:
            pass

        if not isinstance(
            result,
            Topology
        ):
            return None

        # ------------------------------------------------------------------
        # Deep copy must preserve the original Topologic type.
        # ------------------------------------------------------------------

        if _topology_type_name(
            result
        ) != type_name:
            return None

        # ------------------------------------------------------------------
        # Keep the wrapper-level parent dictionary mirror synchronized.
        # ------------------------------------------------------------------

        try:
            result.dictionary = copy.deepcopy(
                Topology.GetDictionary(
                    self
                )
            )
        except Exception:
            pass

        return result

    def CenterOfMass(self):
        from .vertex import Vertex
        from .wire import Wire
        from .face import Face, FaceUtility
        shape = _shape_from_topology(self)

        type_name = _topology_type_name(self)

        # A Vertex has no mass: its center of mass IS its own coordinates.
        # GProp `LinearProperties` on a bare TopoDS_Vertex yields the origin.
        if type_name == "Vertex" and hasattr(self, "x"):
            return Vertex.ByCoordinates(float(self.x), float(self.y), float(self.z))

        # Cluster: average member centroids (matches topologic_core; VolumeProperties
        # over the compound is wrong for mixed-dimension clusters).
        members = getattr(self, "topologies", None)
        if type_name == "Cluster" and members:
            pts = []
            for member in members:
                c = Topology.CenterOfMass(member)
                if c is not None and hasattr(c, "x"):
                    pts.append((float(c.x), float(c.y), float(c.z)))
            if pts:
                n = len(pts)
                return Vertex.ByCoordinates(
                    sum(p[0] for p in pts) / n,
                    sum(p[1] for p in pts) / n,
                    sum(p[2] for p in pts) / n,
                )

        # For faces with internal boundaries (holes), calculate centroid correctly
        # by subtracting hole contributions from the outer face
        internals = getattr(self, 'internals', []) or []
        if internals and hasattr(self, 'external') and self.external is not None:
            # Calculate outer face centroid and area
            outer_face = Face.ByWire(self.external)
            if outer_face is not None:
                outer_com = Topology.CenterOfMass(outer_face)
                outer_area = FaceUtility.Area(outer_face)
                
                if outer_com is not None and outer_area and outer_area > 0:
                    # Subtract hole contributions
                    cx = outer_com.x * outer_area
                    cy = outer_com.y * outer_area
                    cz = outer_com.z * outer_area
                    total_area = outer_area
                    
                    for wire in internals:
                        if isinstance(wire, Wire):
                            hole_face = Face.ByWire(wire)
                            if hole_face is not None:
                                hole_com = Topology.CenterOfMass(hole_face)
                                hole_area = FaceUtility.Area(hole_face)
                                if hole_com is not None and hole_area and hole_area > 0:
                                    cx -= hole_com.x * hole_area
                                    cy -= hole_com.y * hole_area
                                    cz -= hole_com.z * hole_area
                                    total_area -= hole_area
                    
                    if total_area > 0:
                        return Vertex.ByCoordinates(cx / total_area, cy / total_area, cz / total_area)
        
        if not _is_null_shape(shape) and GProp_GProps is not None and brepgprop is not None:
            try:
                props = GProp_GProps()
                type_name = _topology_type_name(self)
                if type_name in ("Cell", "CellComplex", "Cluster"):
                    brepgprop.VolumeProperties(shape, props)
                elif type_name in ("Face", "Shell"):
                    brepgprop.SurfaceProperties(shape, props)
                else:
                    brepgprop.LinearProperties(shape, props)
                center = props.CentreOfMass()
                return Vertex.ByCoordinates(center.X(), center.Y(), center.Z())
            except Exception:
                pass

        vertices = Topology.Vertices(self) or []
        if not vertices:
            return None
        n = len(vertices)
        cx = sum(v.x for v in vertices) / n
        cy = sum(v.y for v in vertices) / n
        cz = sum(v.z for v in vertices) / n
        return Vertex.ByCoordinates(cx, cy, cz)

    Centroid = CenterOfMass

    # -------------------------------------------------------------------
    # Structural queries: SubTopologies / SuperTopologies / SharedTopologies /
    # SelectSubtopology / SelfMerge
    # -------------------------------------------------------------------

    _SUBTOPOLOGY_GETTERS = {
        "vertex": "Vertices", "edge": "Edges", "wire": "Wires", "face": "Faces",
        "shell": "Shells", "cell": "Cells", "cellcomplex": "CellComplexes",
        "cluster": "Clusters", "aperture": "Apertures",
    }

    def SubTopologies(self, typeName: str = None, hostTopology: Any = None, output=None):
        getter_name = Topology._SUBTOPOLOGY_GETTERS.get(str(typeName).strip().lower()) if typeName else None
        if getter_name is None:
            result = []
        else:
            result = getattr(Topology, getter_name)(self) or []
        if output is not None:
            output.extend(result)
            return 0
        return result

    def SuperTopologies(self, hostTopology: Any, typeName: str, output=None):
        getter_name = Topology._SUBTOPOLOGY_GETTERS.get(str(typeName).strip().lower()) if typeName else None
        result = []
        if getter_name is not None and hostTopology is not None:
            candidates = getattr(Topology, getter_name)(hostTopology) or []
            self_vertices = {vertex_key(v) for v in (Topology.Vertices(self) or []) if hasattr(v, "x")}
            for candidate in candidates:
                if Topology.IsSame(candidate, self):
                    continue
                candidate_vertices = {vertex_key(v) for v in (Topology.Vertices(candidate) or []) if hasattr(v, "x")}
                if self_vertices and self_vertices.issubset(candidate_vertices):
                    result.append(candidate)
        result = _deduplicate_by_identity(result)
        if output is not None:
            output.extend(result)
            return 0
        return result

    def SharedTopologies(self, otherTopology: Any, typeID: Any = None, output=None):
        type_name = None
        if isinstance(typeID, str):
            type_name = typeID.strip().lower()
        elif isinstance(typeID, int):
            for name, tid in _TYPE_IDS.items():
                if tid == typeID:
                    type_name = name.lower()
                    break

        getter_name = Topology._SUBTOPOLOGY_GETTERS.get(type_name) if type_name else "Vertices"
        my_items = getattr(Topology, getter_name)(self) or []
        other_items = getattr(Topology, getter_name)(otherTopology) or []

        other_keys = set()
        for item in other_items:
            if hasattr(item, "x") and hasattr(item, "y") and hasattr(item, "z"):
                other_keys.add(vertex_key(item))
            else:
                other_keys.add(getattr(item, "_uuid", id(item)))

        result = []
        for item in my_items:
            if hasattr(item, "x") and hasattr(item, "y") and hasattr(item, "z"):
                key = vertex_key(item)
            else:
                key = getattr(item, "_uuid", id(item))
            if key in other_keys:
                result.append(item)
                continue
            # Geometric fallback: identities (uuids) differ when the same
            # face was rebuilt by BOPAlgo_MakerVolume in adjacent cells,
            # but the underlying OCCT shapes are the same topology.
            item_shape = getattr(item, "shape", None)
            if item_shape is not None:
                for other in other_items:
                    other_shape = getattr(other, "shape", None)
                    if other_shape is not None and Topology.IsSame(item, other):
                        result.append(item)
                        break

        result = _deduplicate_by_identity(result)
        if output is not None:
            output.extend(result)
            return 0
        return result

    def SelectSubtopology(self, selector: Any, typeID: int = 8):
        type_name = None
        for name, tid in _TYPE_IDS.items():
            if tid == typeID:
                type_name = name
                break
        getter_name = Topology._SUBTOPOLOGY_GETTERS.get((type_name or "face").lower())
        candidates = getattr(Topology, getter_name)(self) or [] if getter_name else []
        if not candidates:
            return None

        selector_shape = _shape_from_topology(selector)
        best = None
        best_distance = None
        for candidate in candidates:
            candidate_shape = _shape_from_topology(candidate)
            distance = None
            if (
                not _is_null_shape(selector_shape)
                and not _is_null_shape(candidate_shape)
                and BRepExtrema_DistShapeShape is not None
            ):
                try:
                    dist_calc = BRepExtrema_DistShapeShape(selector_shape, candidate_shape)
                    if dist_calc.IsDone():
                        distance = dist_calc.Value()
                except Exception:
                    distance = None
            if distance is None and hasattr(selector, "x"):
                center = Topology.CenterOfMass(candidate)
                if center is not None:
                    distance = distance3(selector, center)
            if distance is not None and (best_distance is None or distance < best_distance):
                best_distance = distance
                best = candidate
        return best

    def SelfMerge(self, tolerance: float = 0.0001):
        """
        Merges a Cluster's members into the simplest topology that represents
        them: connected Edges become a Wire (or a Cluster of Wires if there
        are disjoint chains), connected Faces become a Shell, connected Cells
        become a CellComplex. Falls back to OCCT same-domain unification for
        topologies that already carry a real shape, and returns self as-is
        when nothing applies.
        """
        from .edge import Edge as _Edge
        from .face import Face as _Face
        from .cell import Cell as _Cell
        from .vertex import Vertex as _Vertex

        members = getattr(self, "topologies", None)
        if members:
            members = [m for m in members if m is not None]
            if members and all(isinstance(m, _Edge) for m in members):
                merged = Topology._merge_edges_into_wires(members, tolerance)
                if merged is not None:
                    return merged
            elif members and any(isinstance(m, _Edge) for m in members) and all(
                isinstance(m, (_Edge, _Vertex)) for m in members
            ):
                # A mix of Edges and standalone Vertices -- typical of
                # Topology.Intersect's per-sub-element decomposition, which
                # independently detects both a genuine edge-vs-edge overlap
                # AND a perpendicular-edge crossing at what is geometrically
                # the same point (one of the overlap edge's own endpoints).
                # Drop any standalone Vertex that coincides with an edge
                # endpoint -- it adds no information and only inflates the
                # vertex count -- keeping only genuinely free vertices.
                from .helpers import same_vertex
                from .cluster import Cluster as _Cluster
                edges = [m for m in members if isinstance(m, _Edge)]
                loose_vertices = [m for m in members if isinstance(m, _Vertex)]
                endpoints = [ep for e in edges for ep in (e.start, e.end) if ep is not None]
                free_vertices = [
                    v for v in loose_vertices
                    if not any(same_vertex(v, ep, tolerance) for ep in endpoints)
                ]
                merged_edges = Topology._merge_edges_into_wires(edges, tolerance)
                if merged_edges is None:
                    pieces = list(edges)
                elif isinstance(merged_edges, _Cluster):
                    pieces = list(getattr(merged_edges, "topologies", None) or [merged_edges])
                else:
                    pieces = [merged_edges]
                pieces = [p for p in pieces if p is not None] + free_vertices
                if len(pieces) == 1:
                    return pieces[0]
                if len(pieces) > 1:
                    merged_cluster = _Cluster.ByTopologies(pieces)
                    if merged_cluster is not None:
                        return merged_cluster
            elif members and all(isinstance(m, _Face) for m in members):
                from .shell import Shell
                shell = Shell.ByFaces(members, tolerance=tolerance, silent=True)
                if shell is not None:
                    return shell
            elif members and all(isinstance(m, _Cell) for m in members):
                merged = Topology._self_merge_cells(members, tolerance)
                if merged is not None:
                    return merged

        shape = _shape_from_topology(self)
        if _is_null_shape(shape):
            return self
        if ShapeUpgrade_UnifySameDomain is not None:
            try:
                unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
                unifier.Build()
                new_shape = unifier.Shape()
                if not _is_null_shape(new_shape):
                    result = Topology.ByOcctShape(new_shape)
                    if result is not None:
                        result.dictionary = Topology.GetDictionary(self)
                        return result
            except Exception:
                pass
        return self

    @staticmethod
    def _self_merge_cells(members, tolerance=0.0001):
        """
        Port of topologic_core Topology::SelfMerge for all-cell members: CellsBuilder -> unique
        faces -> BOPAlgo_MakerVolume rebuild -> a SECOND CellsBuilder pass re-including the
        original cells. The re-inclusion is what discovers finer subdivisions and reproduces
        core's exact cell counts for overlapping subdivided CellComplexes.
        """
        try:
            from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume
        except Exception:
            return None

        shapes = [m.shape for m in members if not _is_null_shape(getattr(m, "shape", None))]
        if not shapes:
            return None

        try:
            args1 = TopTools_ListOfShape()
            for s in shapes:
                args1.Append(s)
            cb1 = BOPAlgo_CellsBuilder()
            cb1.SetArguments(args1)
            cb1.Perform()
            if cb1.HasErrors() or cb1.HasWarnings():
                return None
            cb1.AddAllToResult()

            discarded = [s for s in shapes if cb1.IsDeleted(s)]

            seen_hashes = set()
            occt_faces = []
            explorer = TopExp_Explorer(cb1.Shape(), TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                h = hash(face)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    occt_faces.append(face)
                explorer.Next()

            args2 = TopTools_ListOfShape()
            for f in occt_faces:
                args2.Append(f)
            volume_maker = BOPAlgo_MakerVolume()
            volume_maker.SetArguments(args2)
            volume_maker.SetRunParallel(False)
            volume_maker.SetIntersect(True)
            volume_maker.SetFuzzyValue(0.0)
            volume_maker.Perform()
            volume_maker_ok = not (volume_maker.HasErrors() or volume_maker.HasWarnings())
            if volume_maker_ok:
                discarded.extend(f for f in occt_faces if volume_maker.IsDeleted(f))

            other_shapes = [s for s in shapes if s.ShapeType() != TopAbs_FACE]

            final_args = []
            if volume_maker_ok:
                final_args.append(volume_maker.Shape())
            final_args.extend(discarded)
            final_args.extend(other_shapes)

            if len(final_args) == 1:
                return Topology.ByOcctShape(volume_maker.Shape())

            args3 = TopTools_ListOfShape()
            for s in final_args:
                args3.Append(s)
            cb2 = BOPAlgo_CellsBuilder()
            cb2.SetArguments(args3)
            cb2.Perform()
            if cb2.HasErrors():
                return None
            cb2.AddAllToResult()
            cb2.MakeContainers()
            result_shape = cb2.Shape()
        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)
        result_shape = _promote_to_compsolid_if_multi_solid(result_shape, force=True)
        return Topology.ByOcctShape(result_shape)

    @staticmethod
    def _merge_edges_into_wires(edges, tolerance=0.0001):
        """
        Connects a list of Edge wrapper objects into one or more Wires using
        BRepBuilderAPI_MakeWire, which auto-connects edges sharing endpoints
        regardless of input order. Returns a single Wire, a Cluster of Wires
        (if the edges form disjoint chains), or None on total failure.
        """
        from .edge import Edge as _Edge
        from .wire import Wire

        edges = [e for e in edges if isinstance(e, _Edge) and not _is_null_shape(_shape_from_topology(e))]
        if not edges:
            return None

        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
        except Exception:
            return None

        remaining = list(edges)
        wires = []
        while remaining:
            maker = BRepBuilderAPI_MakeWire()
            first = remaining.pop(0)
            maker.Add(first.shape)
            progress = True
            while progress and remaining:
                progress = False
                for edge in list(remaining):
                    try:
                        maker.Add(edge.shape)
                    except Exception:
                        continue
                    if maker.IsDone():
                        remaining.remove(edge)
                        progress = True
            if not maker.IsDone():
                # Single dangling edge or a maker that never became a valid wire.
                w = Wire.ByOcctShape(first.shape) if hasattr(Wire, "ByOcctShape") else None
                if w is None:
                    w = Wire(shape=first.shape, edges=[first])
                wires.append(w)
                continue
            occ_wire = maker.Wire()
            w = Wire.ByOcctShape(occ_wire) if hasattr(Wire, "ByOcctShape") else None
            if w is None:
                w = Wire(shape=occ_wire, edges=edges)
            wires.append(w)

        if not wires:
            return None
        if len(wires) == 1:
            return wires[0]
        from .cluster import Cluster
        return Cluster.ByTopologies(wires)


# -----------------------------------------------------------------------------
# TopologyUtility namespace
# -----------------------------------------------------------------------------
#
# The developer guide (section 3 / Appendix A) lists TopologyUtility as a
# required namespace, noting it "can alias Topology if Core exposes it that
# way." backend.py and __init__.py already import TopologyUtility from this
# module; without this alias the whole pythonocc_backend package fails to
# import (ImportError: cannot import name 'TopologyUtility').
TopologyUtility = Topology
