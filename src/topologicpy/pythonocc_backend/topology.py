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

from typing import Any, Iterable, Optional


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
    from OCC.Core.TopoDS import (
        TopoDS_Shape,
        topods_Vertex,
        topods_Edge,
        topods_Wire,
        topods_Face,
        topods_Shell,
        topods_Solid,
        topods_Compound,
    )
    from OCC.Core.TopTools import TopTools_ListOfShape
    from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.ShapeFix import ShapeFix_Shape
except Exception:  # pragma: no cover - allows import without PythonOCC
    TopAbs_VERTEX = TopAbs_EDGE = TopAbs_WIRE = TopAbs_FACE = None
    TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPSOLID = TopAbs_COMPOUND = None
    TopExp_Explorer = None
    TopoDS_Shape = None
    topods_Vertex = topods_Edge = topods_Wire = topods_Face = None
    topods_Shell = topods_Solid = topods_Compound = None
    TopTools_ListOfShape = None
    BOPAlgo_CellsBuilder = None
    BRepCheck_Analyzer = None
    ShapeFix_Shape = None


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


def _iter_occ_subshapes(shape: Any, shape_type: Any) -> list:
    if _is_null_shape(shape) or TopExp_Explorer is None or shape_type is None:
        return []

    result = []
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def _deduplicate_by_identity(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        key = id(item)
        if hasattr(item, "_uuid"):
            key = getattr(item, "_uuid")
        elif hasattr(item, "HashCode"):
            try:
                key = item.HashCode(2147483647)
            except Exception:
                key = id(item)
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
    if _is_null_shape(shape):
        return shape

    if ShapeFix_Shape is None:
        return shape

    try:
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if not _is_null_shape(fixed):
            return fixed
    except Exception:
        pass

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

    if shape_type in (TopAbs_COMPSOLID, TopAbs_COMPOUND):
        # Prefer Cluster as the safest aggregate wrapper.
        try:
            from .cluster import Cluster
            children = []
            for st in (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX):
                children.extend([Topology.ByOcctShape(s) for s in _iter_occ_subshapes(shape, st)])
            children = [c for c in children if c is not None]
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


def _make_occ_merge(topology: Any, other_topology: Any = None, transfer_dictionary: bool = False) -> Any:
    """
    PythonOCC implementation of Topology.Merge.

    This mirrors the Topologic C++ logic:
        1. Build non-regular boolean cells from both operands.
        2. Add all cells belonging to operand A.
        3. Add all cells belonging to operand B.
        4. Make containers.
        5. Convert result back into a backend topology object.
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

    if BOPAlgo_CellsBuilder is None:
        print("Topology.Merge - Error: PythonOCC BOPAlgo_CellsBuilder is not available. Returning None.")
        return None

    shapes_a = _collect_boolean_operand_shapes(topology)
    shapes_b = _collect_boolean_operand_shapes(other_topology)

    if not shapes_a or not shapes_b:
        print("Topology.Merge - Error: Could not collect valid OCCT operands. Returning None.")
        return None

    try:
        args_a = _make_toptools_list(shapes_a)
        args_b = _make_toptools_list(shapes_b)

        builder = BOPAlgo_CellsBuilder()

        for shape in shapes_a:
            builder.AddArgument(shape)
        for shape in shapes_b:
            builder.AddArgument(shape)

        builder.Perform()

        if hasattr(builder, "HasErrors") and builder.HasErrors():
            print("Topology.Merge - Error: BOPAlgo_CellsBuilder failed. Returning None.")
            return None

        empty_avoid = TopTools_ListOfShape()

        for shape in shapes_a:
            to_take = TopTools_ListOfShape()
            to_take.Append(shape)
            builder.AddToResult(to_take, empty_avoid)

        for shape in shapes_b:
            to_take = TopTools_ListOfShape()
            to_take.Append(shape)
            builder.AddToResult(to_take, empty_avoid)

        builder.MakeContainers()
        result_shape = builder.Shape()

        if _is_null_shape(result_shape):
            print("Topology.Merge - Error: Boolean result is null. Returning None.")
            return None

        result_shape = _postprocess_boolean_result(result_shape)

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

    except Exception as exc:
        print(f"Topology.Merge - Error: {exc}. Returning None.")
        return None


# -----------------------------------------------------------------------------
# Public Topology class
# -----------------------------------------------------------------------------

class Topology:
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

    @staticmethod
    def TypeAsString(topology: Any) -> Optional[str]:
        result = _topology_type_name(topology)
        if result is None:
            print("Topology.TypeAsString - Error: The input topology parameter is not a valid topology or graph. Returning None.")
        return result

    @staticmethod
    def Dictionary(topology: Any):
        if topology is None:
            return None
        if isinstance(topology, dict):
            return topology.get("dictionary", {})
        return getattr(topology, "dictionary", {})

    @staticmethod
    def SetDictionary(topology: Any, dictionary: Any):
        if topology is None:
            return None
        try:
            setattr(topology, "dictionary", dictionary if dictionary is not None else {})
            return topology
        except Exception:
            return None

    @staticmethod
    def ByOcctShape(shape: Any, dictionary=None, contents=None, contexts=None, apertures=None):
        return _wrap_shape_as_topology(
            shape,
            dictionary=dictionary,
            contents=contents,
            contexts=contexts,
            apertures=apertures,
        )

    @staticmethod
    def Vertices(topology: Any) -> Optional[list]:
        if topology is None:
            return None

        if hasattr(topology, "vertices"):
            return _safe_list(getattr(topology, "vertices"))

        if _topology_type_name(topology) == "Vertex":
            return [topology]

        if hasattr(topology, "start") and hasattr(topology, "end"):
            return [getattr(topology, "start"), getattr(topology, "end")]

        result = []
        shape = _shape_from_topology(topology)
        for subshape in _iter_occ_subshapes(shape, TopAbs_VERTEX):
            v = Topology.ByOcctShape(subshape)
            if v is not None:
                result.append(v)

        # Wrapper aggregate fallback
        if not result:
            for attr in ("edges", "faces", "shells", "cells", "topologies", "members"):
                if hasattr(topology, attr):
                    for child in _safe_list(getattr(topology, attr)):
                        child_vertices = Topology.Vertices(child) or []
                        result.extend(child_vertices)

        return _deduplicate_by_identity(result)

    @staticmethod
    def Edges(topology: Any) -> Optional[list]:
        if topology is None:
            print("Topology.Edges - Error: The input is not a valid topology. Returning None")
            return None

        if hasattr(topology, "edges"):
            return _safe_list(getattr(topology, "edges"))

        if _topology_type_name(topology) == "Edge":
            return [topology]

        result = []
        shape = _shape_from_topology(topology)
        for subshape in _iter_occ_subshapes(shape, TopAbs_EDGE):
            e = Topology.ByOcctShape(subshape)
            if e is not None:
                result.append(e)

        if not result:
            for attr in ("faces", "shells", "cells", "topologies", "members"):
                if hasattr(topology, attr):
                    for child in _safe_list(getattr(topology, attr)):
                        child_edges = Topology.Edges(child) or []
                        result.extend(child_edges)

        return _deduplicate_by_identity(result)

    @staticmethod
    def Wires(topology: Any) -> Optional[list]:
        if topology is None:
            print("Topology.Wires - Error: The input is not a valid topology. Returning None")
            return None

        if _topology_type_name(topology) == "Wire":
            return [topology]

        if hasattr(topology, "external") and getattr(topology, "external") is not None:
            wires = [getattr(topology, "external")]
            wires.extend(_safe_list(getattr(topology, "internals", [])))
            return wires

        result = []
        shape = _shape_from_topology(topology)
        for subshape in _iter_occ_subshapes(shape, TopAbs_WIRE):
            w = Topology.ByOcctShape(subshape)
            if w is not None:
                result.append(w)

        if not result:
            for attr in ("faces", "shells", "cells", "topologies", "members"):
                if hasattr(topology, attr):
                    for child in _safe_list(getattr(topology, attr)):
                        child_wires = Topology.Wires(child) or []
                        result.extend(child_wires)

        return _deduplicate_by_identity(result)

    @staticmethod
    def Faces(topology: Any) -> Optional[list]:
        if topology is None:
            print("Topology.Faces - Error: The input is not a valid topology. Returning None")
            return None

        if _topology_type_name(topology) == "Face":
            print("Topology.Faces - Warning: The input is a Face. Returning the same face embedded in a list.")
            return [topology]

        if hasattr(topology, "faces"):
            return _safe_list(getattr(topology, "faces"))

        result = []
        shape = _shape_from_topology(topology)
        for subshape in _iter_occ_subshapes(shape, TopAbs_FACE):
            f = Topology.ByOcctShape(subshape)
            if f is not None:
                result.append(f)

        if not result:
            for attr in ("shells", "cells", "topologies", "members"):
                if hasattr(topology, attr):
                    for child in _safe_list(getattr(topology, attr)):
                        child_faces = Topology.Faces(child) or []
                        result.extend(child_faces)

        return _deduplicate_by_identity(result)

    @staticmethod
    def Shells(topology: Any) -> Optional[list]:
        if topology is None:
            print("Topology.Shells - Error: The input is not a valid topology. Returning None")
            return None

        if _topology_type_name(topology) == "Shell":
            return [topology]

        if hasattr(topology, "shells"):
            return _safe_list(getattr(topology, "shells"))

        result = []
        shape = _shape_from_topology(topology)
        for subshape in _iter_occ_subshapes(shape, TopAbs_SHELL):
            sh = Topology.ByOcctShape(subshape)
            if sh is not None:
                result.append(sh)

        if not result:
            for attr in ("cells", "topologies", "members"):
                if hasattr(topology, attr):
                    for child in _safe_list(getattr(topology, attr)):
                        child_shells = Topology.Shells(child) or []
                        result.extend(child_shells)

        return _deduplicate_by_identity(result)

    @staticmethod
    def Merge(topology: Any, otherTopology: Any = None, transferDictionary: bool = False):
        return _make_occ_merge(
            topology,
            otherTopology,
            transfer_dictionary=transferDictionary,
        )

    # Compatibility aliases sometimes used by TopologicPy style code.
    @staticmethod
    def Union(topology: Any, otherTopology: Any, transferDictionary: bool = False):
        return Topology.Merge(topology, otherTopology, transferDictionary=transferDictionary)

    @staticmethod
    def Difference(topology: Any, otherTopology: Any, transferDictionary: bool = False):
        return _not_implemented("Topology.Difference")

    @staticmethod
    def Intersect(topology: Any, otherTopology: Any, transferDictionary: bool = False):
        return _not_implemented("Topology.Intersect")
