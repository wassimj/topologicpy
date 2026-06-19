# -*- coding: utf-8 -*-
"""
PythonOCCBackend.py

A lightweight experimental pythonocc-based replacement backend for topologic_core.

This backend is intentionally pragmatic. It implements the subset of the
topologic_core API shape that TopologicPy commonly calls through Core.py:
static constructors, utility namespaces, topology type checks, dictionary
attributes, list-output instance calls, and simple graph operations.

It is not a full topologic_core replacement. Boolean operations, robust
non-manifold cell-complex construction, exact OCCT topological healing, and
advanced geometric predicates are placeholders or simplified implementations.

The key design goal is compatibility with TopologicPy's Core abstraction:
    Core.Vertex.ByCoordinates(...)
    Core.Edge.ByStartVertexEndVertex(...)
    Core.InstanceCall(obj, "Edges", None, edges)
    Core.Topology.IsSame(a, b)

This file deliberately tolerates None dictionaries on topology objects. That
prevents Dictionary.Keys(None) failures during Cluster.ByTopologies,
Topology.SelfMerge, Wire.ByEdges, Face.ByWire, Shell.ByFaces, and Cell.ByShell.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# -----------------------------------------------------------------------------
# Optional pythonocc imports
# -----------------------------------------------------------------------------

try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
    )
    from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge, TopoDS_Wire, TopoDS_Face
    PYTHONOCC_AVAILABLE = True
except Exception:
    gp_Pnt = None
    BRepBuilderAPI_MakeVertex = None
    BRepBuilderAPI_MakeEdge = None
    BRepBuilderAPI_MakeWire = None
    BRepBuilderAPI_MakeFace = None
    TopoDS_Vertex = type("TopoDS_Vertex", (), {})
    TopoDS_Edge = type("TopoDS_Edge", (), {})
    TopoDS_Wire = type("TopoDS_Wire", (), {})
    TopoDS_Face = type("TopoDS_Face", (), {})
    PYTHONOCC_AVAILABLE = False


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _distance(a: "Vertex", b: "Vertex") -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _same_vertex(a: Any, b: Any, tolerance: float = 1e-6) -> bool:
    return isinstance(a, Vertex) and isinstance(b, Vertex) and _distance(a, b) <= tolerance


def _dedupe(items: Iterable[Any]) -> list:
    result = []
    seen = set()
    for item in items:
        key = getattr(item, "_uuid", id(item))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _valid_topologies(items: Iterable[Any]) -> list:
    return [x for x in items if isinstance(x, TopologyBase)]


def _as_dictionary(d: Any) -> dict:
    """
    Return a plain Python dict for TopologicPy compatibility.

    TopologicPy's own Dictionary.Keys/ValueAtKey functions reliably accept
    python dict objects. They may not recognise this backend's Dictionary
    class as a valid topologic dictionary, so topology.Dictionary() should
    expose a plain dict rather than a backend Dictionary instance.
    """
    if d is None:
        return {}
    if isinstance(d, Dictionary):
        return dict(d.data)
    if isinstance(d, dict):
        return dict(d)
    return {}


def _shape_class(shape: Any) -> Any:
    if shape is None:
        return None
    return shape.__class__


def _make_occ_vertex(x: float, y: float, z: float) -> Any:
    if not PYTHONOCC_AVAILABLE:
        return TopoDS_Vertex()
    return BRepBuilderAPI_MakeVertex(gp_Pnt(float(x), float(y), float(z))).Vertex()


def _make_occ_edge(v1: "Vertex", v2: "Vertex") -> Any:
    if not PYTHONOCC_AVAILABLE:
        return TopoDS_Edge()
    return BRepBuilderAPI_MakeEdge(
        gp_Pnt(v1.x, v1.y, v1.z),
        gp_Pnt(v2.x, v2.y, v2.z),
    ).Edge()


def _make_occ_wire(edges: list["Edge"]) -> Any:
    if not PYTHONOCC_AVAILABLE:
        return TopoDS_Wire()
    maker = BRepBuilderAPI_MakeWire()
    for e in edges:
        if getattr(e, "shape", None) is not None:
            maker.Add(e.shape)
    if not maker.IsDone():
        return None
    return maker.Wire()


def _make_occ_face(wire: "Wire") -> Any:
    if not PYTHONOCC_AVAILABLE:
        return TopoDS_Face()
    if getattr(wire, "shape", None) is None:
        return None
    maker = BRepBuilderAPI_MakeFace(wire.shape)
    if not maker.IsDone():
        return None
    return maker.Face()


def _polygon_area_xy(vertices: list["Vertex"]) -> float:
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        area += a.x * b.y - b.x * a.y
    return abs(area) * 0.5


def _ordered_edges(edges: list["Edge"], tolerance: float = 1e-6) -> Optional[list["Edge"]]:
    """Return edges ordered head-to-tail, reversing edges if needed."""
    if not edges:
        return []

    unused = list(edges)
    ordered = [unused.pop(0)]

    while unused:
        last = ordered[-1].EndVertex()
        found_index = None
        reverse = False

        for i, e in enumerate(unused):
            if _same_vertex(last, e.StartVertex(), tolerance):
                found_index = i
                reverse = False
                break
            if _same_vertex(last, e.EndVertex(), tolerance):
                found_index = i
                reverse = True
                break

        if found_index is None:
            return None

        e = unused.pop(found_index)
        if reverse:
            old_e = e
            e = Edge.ByStartVertexEndVertex(old_e.EndVertex(), old_e.StartVertex())
            e.dictionary = _as_dictionary(getattr(old_e, "dictionary", None))
        ordered.append(e)

    return ordered


def _vertices_from_edges(edges: list["Edge"], tolerance: float = 1e-6) -> list["Vertex"]:
    if not edges:
        return []

    ordered = _ordered_edges(edges, tolerance=tolerance) or edges
    vertices = [ordered[0].StartVertex()]
    for e in ordered:
        vertices.append(e.EndVertex())

    if len(vertices) > 1 and _same_vertex(vertices[0], vertices[-1], tolerance):
        vertices = vertices[:-1]
    return _dedupe(vertices)


# -----------------------------------------------------------------------------
# Attribute classes
# -----------------------------------------------------------------------------

class IntAttribute:
    def __init__(self, value: Any = 0):
        self.value = int(value)

    def IntValue(self) -> int:
        return int(self.value)


class DoubleAttribute:
    def __init__(self, value: Any = 0.0):
        self.value = float(value)

    def DoubleValue(self) -> float:
        return float(self.value)


class StringAttribute:
    def __init__(self, value: Any = ""):
        self.value = "" if value is None else str(value)

    def StringValue(self) -> str:
        return str(self.value)


class ListAttribute:
    def __init__(self, values: Optional[list[Any]] = None):
        self.values = list(values or [])

    def ListValue(self) -> list[Any]:
        return list(self.values)


# -----------------------------------------------------------------------------
# Dictionary
# -----------------------------------------------------------------------------

class Dictionary:
    def __init__(self, data: Optional[dict[str, Any]] = None):
        self.data = dict(data or {})

    @staticmethod
    def ByKeysValues(keys: list[str], values: list[Any]) -> "Dictionary":
        if keys is None or values is None:
            return Dictionary({})
        return Dictionary({str(k): values[i] for i, k in enumerate(keys) if i < len(values)})

    @staticmethod
    def Keys(dictionary: Any) -> list[str]:
        if dictionary is None:
            return []
        if isinstance(dictionary, Dictionary):
            return list(dictionary.data.keys())
        if isinstance(dictionary, dict):
            return list(dictionary.keys())
        return []

    @staticmethod
    def Values(dictionary: Any) -> list[Any]:
        if dictionary is None:
            return []
        if isinstance(dictionary, Dictionary):
            return list(dictionary.data.values())
        if isinstance(dictionary, dict):
            return list(dictionary.values())
        return []

    @staticmethod
    def ValueAtKey(dictionary: Any, key: str, defaultValue: Any = None) -> Any:
        if dictionary is None or key is None:
            return defaultValue
        if isinstance(dictionary, Dictionary):
            return dictionary.data.get(key, defaultValue)
        if isinstance(dictionary, dict):
            return dictionary.get(key, defaultValue)
        return defaultValue

    @staticmethod
    def SetValueAtKey(dictionary: Any, key: str, value: Any) -> "Dictionary":
        d = _as_dictionary(dictionary)
        if key is not None:
            d.data[str(key)] = value
        return d

    def Keys(self) -> list[str]:  # instance fallback
        return list(self.data.keys())

    def Values(self) -> list[Any]:  # instance fallback
        return list(self.data.values())

    def ValueAtKey(self, key: str, defaultValue: Any = None) -> Any:
        return self.data.get(key, defaultValue)

    def SetValueAtKey(self, key: str, value: Any) -> "Dictionary":
        if key is not None:
            self.data[str(key)] = value
        return self


# -----------------------------------------------------------------------------
# Base topology
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class TopologyBase:
    shape: Any = None
    dictionary: Any = None
    contents: list[Any] = field(default_factory=list)
    contexts: list[Any] = field(default_factory=list)
    apertures: list[Any] = field(default_factory=list)
    _uuid: str = field(default_factory=_uuid)

    def __hash__(self) -> int:
        return hash(self._uuid)

    def GetTypeAsString(self) -> str:
        return self.__class__.__name__

    def Type(self) -> int:
        return {
            "Vertex": 1,
            "Edge": 2,
            "Wire": 4,
            "Face": 8,
            "Shell": 16,
            "Cell": 32,
            "CellComplex": 64,
            "Cluster": 128,
            "Graph": 256,
        }.get(self.GetTypeAsString(), 0)

    def Dictionary(self) -> dict:
        return _as_dictionary(self.dictionary)

    def SetDictionary(self, dictionary: Any) -> "TopologyBase":
        self.dictionary = _as_dictionary(dictionary)
        return self

    def Contents(self, *args) -> Any:
        if args and isinstance(args[-1], list):
            args[-1].extend(self.contents)
            return 0
        return list(self.contents)

    def Contexts(self, *args) -> Any:
        if args and isinstance(args[-1], list):
            args[-1].extend(self.contexts)
            return 0
        return list(self.contexts)

    def Apertures(self, *args) -> Any:
        if args and isinstance(args[-1], list):
            args[-1].extend(self.apertures)
            return 0
        return list(self.apertures)

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        vertices = Topology.Vertices(self)
        if out is not None:
            out.extend(vertices)
            return 0
        return vertices

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = Topology.Edges(self)
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Wires(self, *args) -> Any:
        out = _find_output_list(args)
        wires = Topology.Wires(self)
        if out is not None:
            out.extend(wires)
            return 0
        return wires

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        faces = Topology.Faces(self)
        if out is not None:
            out.extend(faces)
            return 0
        return faces

    def Shells(self, *args) -> Any:
        out = _find_output_list(args)
        shells = Topology.Shells(self)
        if out is not None:
            out.extend(shells)
            return 0
        return shells

    def Cells(self, *args) -> Any:
        out = _find_output_list(args)
        cells = Topology.Cells(self)
        if out is not None:
            out.extend(cells)
            return 0
        return cells

    def DeepCopy(self) -> "TopologyBase":
        return copy.deepcopy(self)

    def Merge(self, other: Any, *args) -> Any:
        return Topology.Merge(self, other, *args)


def _find_output_list(args: tuple[Any, ...]) -> Optional[list]:
    for arg in reversed(args):
        if isinstance(arg, list):
            return arg
    return None


# -----------------------------------------------------------------------------
# Core topology classes
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class Vertex(TopologyBase):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def ByCoordinates(x: float = 0, y: float = 0, z: float = 0) -> "Vertex":
        x, y, z = float(x), float(y), float(z)
        return Vertex(shape=_make_occ_vertex(x, y, z), x=x, y=y, z=z)

    def X(self) -> float:
        return float(self.x)

    def Y(self) -> float:
        return float(self.y)

    def Z(self) -> float:
        return float(self.z)

    def Coordinates(self) -> list[float]:
        return [self.X(), self.Y(), self.Z()]

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.append(self)
            return 0
        return [self]


@dataclass(eq=False)
class Edge(TopologyBase):
    start: Optional[Vertex] = None
    end: Optional[Vertex] = None

    @staticmethod
    def ByStartVertexEndVertex(vertexA: Vertex, vertexB: Vertex) -> Optional["Edge"]:
        if not isinstance(vertexA, Vertex) or not isinstance(vertexB, Vertex):
            return None
        if _same_vertex(vertexA, vertexB, tolerance=1e-12):
            return None
        return Edge(shape=_make_occ_edge(vertexA, vertexB), start=vertexA, end=vertexB)

    @staticmethod
    def ByVertices(vertices: list[Vertex]) -> Optional["Edge"]:
        if not vertices or len(vertices) < 2:
            return None
        return Edge.ByStartVertexEndVertex(vertices[0], vertices[-1])

    def StartVertex(self) -> Optional[Vertex]:
        return self.start

    def EndVertex(self) -> Optional[Vertex]:
        return self.end

    def Length(self) -> Optional[float]:
        if not isinstance(self.start, Vertex) or not isinstance(self.end, Vertex):
            return None
        return _distance(self.start, self.end)

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = [v for v in [self.start, self.end] if isinstance(v, Vertex)]
        if out is not None:
            out.extend(verts)
            return 0
        return verts

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.append(self)
            return 0
        return [self]


@dataclass(eq=False)
class Wire(TopologyBase):
    edges: list[Edge] = field(default_factory=list)

    @staticmethod
    def ByEdges(edges: list[Edge], *args, **kwargs) -> Optional["Wire"]:
        tolerance = kwargs.get("tolerance", 1e-6)
        if not edges:
            return None
        valid_edges = [e for e in edges if isinstance(e, Edge)]
        if len(valid_edges) == 0:
            return None

        ordered = _ordered_edges(valid_edges, tolerance=tolerance)
        if ordered is None:
            ordered = valid_edges

        shape = _make_occ_wire(ordered)
        wire = Wire(shape=shape, edges=ordered)
        wire.dictionary = {}
        return wire

    @staticmethod
    def ByVertices(vertices: list[Vertex], close: bool = False, *args, **kwargs) -> Optional["Wire"]:
        if not vertices or len(vertices) < 2:
            return None

        tolerance = kwargs.get("tolerance", 1e-6)
        verts = [v for v in vertices if isinstance(v, Vertex)]
        if len(verts) < 2:
            return None

        edges = []
        count = len(verts)
        edge_count = count if close else count - 1
        for i in range(edge_count):
            a = verts[i]
            b = verts[(i + 1) % count]
            if _same_vertex(a, b, tolerance=tolerance):
                continue
            e = Edge.ByStartVertexEndVertex(a, b)
            if e is not None:
                edges.append(e)

        if len(edges) == 0:
            return None
        return Wire.ByEdges(edges, tolerance=tolerance)

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.edges)
            return 0
        return list(self.edges)

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = _vertices_from_edges(self.edges)
        if out is not None:
            out.extend(verts)
            return 0
        return verts

    def IsClosed(self) -> bool:
        if not self.edges:
            return False
        ordered = _ordered_edges(self.edges) or self.edges
        return _same_vertex(ordered[0].StartVertex(), ordered[-1].EndVertex())


@dataclass(eq=False)
class Face(TopologyBase):
    external: Optional[Wire] = None
    internals: list[Wire] = field(default_factory=list)

    @staticmethod
    def ByExternalBoundary(wire: Wire, *args, **kwargs) -> Optional["Face"]:
        return Face.ByWire(wire, *args, **kwargs)

    @staticmethod
    def ByWire(wire: Wire, *args, **kwargs) -> Optional["Face"]:
        if not isinstance(wire, Wire):
            return None
        if not wire.IsClosed():
            return None
        face = Face(shape=_make_occ_face(wire), external=wire, internals=[])
        face.dictionary = {}
        return face

    @staticmethod
    def ByWires(externalBoundary: Wire, internalBoundaries: Optional[list[Wire]] = None, *args, **kwargs) -> Optional["Face"]:
        face = Face.ByWire(externalBoundary, *args, **kwargs)
        if face is None:
            return None
        face.internals = [w for w in (internalBoundaries or []) if isinstance(w, Wire)]
        return face

    @staticmethod
    def ByVertices(vertices: list[Vertex], *args, **kwargs) -> Optional["Face"]:
        w = Wire.ByVertices(vertices, close=True, *args, **kwargs)
        return Face.ByWire(w, *args, **kwargs)

    def ExternalBoundary(self) -> Optional[Wire]:
        return self.external

    def InternalBoundaries(self, *args) -> Any:
        out = _find_output_list(args)
        internals = list(self.internals)
        if out is not None:
            out.extend(internals)
            return 0
        return internals

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = []
        if isinstance(self.external, Wire):
            edges.extend(self.external.Edges())
        for w in self.internals:
            edges.extend(w.Edges())
        edges = _dedupe(edges)
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Wires(self, *args) -> Any:
        out = _find_output_list(args)
        wires = []
        if isinstance(self.external, Wire):
            wires.append(self.external)
        wires.extend(self.internals)
        if out is not None:
            out.extend(wires)
            return 0
        return wires

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.append(self)
            return 0
        return [self]

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = []
        for e in self.Edges():
            verts.extend(e.Vertices())
        verts = _dedupe(verts)
        if out is not None:
            out.extend(verts)
            return 0
        return verts

    def Area(self) -> float:
        verts = self.external.Vertices() if isinstance(self.external, Wire) else []
        area = _polygon_area_xy(verts)
        for w in self.internals:
            area -= _polygon_area_xy(w.Vertices())
        return max(0.0, area)


@dataclass(eq=False)
class Shell(TopologyBase):
    faces: list[Face] = field(default_factory=list)

    @staticmethod
    def ByFaces(faces: list[Face], *args, **kwargs) -> Optional["Shell"]:
        valid_faces = [f for f in (faces or []) if isinstance(f, Face)]
        if not valid_faces:
            return None
        shell = Shell(shape=None, faces=valid_faces)
        shell.dictionary = {}
        return shell

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.faces)
            return 0
        return list(self.faces)

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = _dedupe(e for f in self.faces for e in f.Edges())
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = _dedupe(v for e in self.Edges() for v in e.Vertices())
        if out is not None:
            out.extend(verts)
            return 0
        return verts


@dataclass(eq=False)
class Cell(TopologyBase):
    faces: list[Face] = field(default_factory=list)

    @staticmethod
    def ByFaces(faces: list[Face], *args, **kwargs) -> Optional["Cell"]:
        valid_faces = [f for f in (faces or []) if isinstance(f, Face)]
        if not valid_faces:
            return None
        cell = Cell(shape=None, faces=valid_faces)
        cell.dictionary = {}
        return cell

    @staticmethod
    def ByShell(shell: Shell, *args, **kwargs) -> Optional["Cell"]:
        if not isinstance(shell, Shell):
            return None
        return Cell.ByFaces(shell.Faces(), *args, **kwargs)

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.faces)
            return 0
        return list(self.faces)

    def Shells(self, *args) -> Any:
        out = _find_output_list(args)
        shell = Shell.ByFaces(self.faces)
        shells = [shell] if shell is not None else []
        if out is not None:
            out.extend(shells)
            return 0
        return shells

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = _dedupe(e for f in self.faces for e in f.Edges())
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = _dedupe(v for e in self.Edges() for v in e.Vertices())
        if out is not None:
            out.extend(verts)
            return 0
        return verts


@dataclass(eq=False)
class CellComplex(TopologyBase):
    cells: list[Cell] = field(default_factory=list)

    @staticmethod
    def ByCells(cells: list[Cell], *args, **kwargs) -> Optional["CellComplex"]:
        valid = [c for c in (cells or []) if isinstance(c, Cell)]
        if not valid:
            return None
        cc = CellComplex(shape=None, cells=valid)
        cc.dictionary = {}
        return cc

    @staticmethod
    def ByFaces(faces: list[Face], *args, **kwargs) -> Optional["CellComplex"]:
        cell = Cell.ByFaces(faces or [])
        if cell is None:
            return None
        return CellComplex.ByCells([cell])

    def Cells(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.cells)
            return 0
        return list(self.cells)

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        faces = _dedupe(f for c in self.cells for f in c.Faces())
        if out is not None:
            out.extend(faces)
            return 0
        return faces

    def Shells(self, *args) -> Any:
        out = _find_output_list(args)
        shells = _dedupe(s for c in self.cells for s in c.Shells())
        if out is not None:
            out.extend(shells)
            return 0
        return shells

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = _dedupe(e for f in self.Faces() for e in f.Edges())
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = _dedupe(v for e in self.Edges() for v in e.Vertices())
        if out is not None:
            out.extend(verts)
            return 0
        return verts


@dataclass(eq=False)
class Cluster(TopologyBase):
    topologies: list[TopologyBase] = field(default_factory=list)

    @staticmethod
    def ByTopologies(topologyList: list[Any], *args, **kwargs) -> Optional["Cluster"]:
        valid = _valid_topologies(topologyList or [])
        if not valid:
            return None

        # Deliberately do not inspect/merge child dictionaries. Some TopologicPy
        # paths call Cluster.ByTopologies with children whose dictionary is None.
        # Treat that as an empty dictionary instead of calling Dictionary.Keys(None).
        cluster = Cluster(shape=None, topologies=valid)
        cluster.dictionary = {}
        return cluster

    def Topologies(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.topologies)
            return 0
        return list(self.topologies)

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        verts = _dedupe(v for t in self.topologies for v in Topology.Vertices(t))
        if out is not None:
            out.extend(verts)
            return 0
        return verts

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        edges = _dedupe(e for t in self.topologies for e in Topology.Edges(t))
        if out is not None:
            out.extend(edges)
            return 0
        return edges

    def Wires(self, *args) -> Any:
        out = _find_output_list(args)
        wires = _dedupe(w for t in self.topologies for w in Topology.Wires(t))
        if out is not None:
            out.extend(wires)
            return 0
        return wires

    def Faces(self, *args) -> Any:
        out = _find_output_list(args)
        faces = _dedupe(f for t in self.topologies for f in Topology.Faces(t))
        if out is not None:
            out.extend(faces)
            return 0
        return faces


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class Graph:
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    dictionary: Any = None
    _uuid: str = field(default_factory=_uuid)

    def __hash__(self) -> int:
        return hash(self._uuid)

    @staticmethod
    def ByVerticesEdges(vertices: list[Vertex], edges: list[Edge], *args, **kwargs) -> "Graph":
        return Graph(vertices=_dedupe([v for v in (vertices or []) if isinstance(v, Vertex)]),
                     edges=_dedupe([e for e in (edges or []) if isinstance(e, Edge)]),
                     dictionary={})

    def GetTypeAsString(self) -> str:
        return "Graph"

    def Vertices(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.vertices)
            return 0
        return list(self.vertices)

    def Edges(self, *args) -> Any:
        out = _find_output_list(args)
        if out is not None:
            out.extend(self.edges)
            return 0
        return list(self.edges)

    def Dictionary(self) -> dict:
        return _as_dictionary(self.dictionary)

    def SetDictionary(self, dictionary: Any) -> "Graph":
        self.dictionary = _as_dictionary(dictionary)
        return self


# -----------------------------------------------------------------------------
# Utility namespaces
# -----------------------------------------------------------------------------

class Topology:
    @staticmethod
    def IsSame(topologyA: Any, topologyB: Any) -> bool:
        if topologyA is topologyB:
            return True
        if topologyA is None or topologyB is None:
            return False
        if getattr(topologyA, "_uuid", None) == getattr(topologyB, "_uuid", None):
            return True
        if isinstance(topologyA, Vertex) and isinstance(topologyB, Vertex):
            return _same_vertex(topologyA, topologyB)
        if isinstance(topologyA, Edge) and isinstance(topologyB, Edge):
            return (
                _same_vertex(topologyA.StartVertex(), topologyB.StartVertex())
                and _same_vertex(topologyA.EndVertex(), topologyB.EndVertex())
            ) or (
                _same_vertex(topologyA.StartVertex(), topologyB.EndVertex())
                and _same_vertex(topologyA.EndVertex(), topologyB.StartVertex())
            )
        return False

    @staticmethod
    def Type(topology: Any) -> int:
        if hasattr(topology, "Type"):
            return topology.Type()
        if isinstance(topology, Graph):
            return 256
        return 0

    @staticmethod
    def TypeAsString(topology: Any) -> Optional[str]:
        if topology is None:
            return None
        if isinstance(topology, Graph):
            return "Graph"
        if isinstance(topology, TopologyBase):
            return topology.GetTypeAsString()
        return None

    @staticmethod
    def Dictionary(topology: Any) -> Optional[Dictionary]:
        if topology is None or not hasattr(topology, "Dictionary"):
            return None
        return topology.Dictionary()

    @staticmethod
    def SetDictionary(topology: Any, dictionary: Any) -> Any:
        if topology is None or not hasattr(topology, "SetDictionary"):
            return None
        return topology.SetDictionary(dictionary)

    @staticmethod
    def Vertices(topology: Any) -> list[Vertex]:
        if topology is None:
            return []
        if isinstance(topology, Vertex):
            return [topology]
        if isinstance(topology, Edge):
            return topology.Vertices()
        if isinstance(topology, Wire):
            return topology.Vertices()
        if isinstance(topology, Face):
            return topology.Vertices()
        if isinstance(topology, Shell):
            return topology.Vertices()
        if isinstance(topology, Cell):
            return topology.Vertices()
        if isinstance(topology, CellComplex):
            return topology.Vertices()
        if isinstance(topology, Cluster):
            return topology.Vertices()
        if isinstance(topology, Graph):
            return topology.Vertices()
        return []

    @staticmethod
    def Edges(topology: Any) -> list[Edge]:
        if topology is None:
            return []
        if isinstance(topology, Edge):
            return [topology]
        if isinstance(topology, Wire):
            return topology.Edges()
        if isinstance(topology, Face):
            return topology.Edges()
        if isinstance(topology, Shell):
            return topology.Edges()
        if isinstance(topology, Cell):
            return topology.Edges()
        if isinstance(topology, CellComplex):
            return topology.Edges()
        if isinstance(topology, Cluster):
            return topology.Edges()
        if isinstance(topology, Graph):
            return topology.Edges()
        return []

    @staticmethod
    def Wires(topology: Any) -> list[Wire]:
        if topology is None:
            return []
        if isinstance(topology, Wire):
            return [topology]
        if isinstance(topology, Face):
            return topology.Wires()
        if isinstance(topology, Cluster):
            return topology.Wires()
        return []

    @staticmethod
    def Faces(topology: Any) -> list[Face]:
        if topology is None:
            return []
        if isinstance(topology, Face):
            return [topology]
        if isinstance(topology, Shell):
            return topology.Faces()
        if isinstance(topology, Cell):
            return topology.Faces()
        if isinstance(topology, CellComplex):
            return topology.Faces()
        if isinstance(topology, Cluster):
            return topology.Faces()
        return []

    @staticmethod
    def Shells(topology: Any) -> list[Shell]:
        if topology is None:
            return []
        if isinstance(topology, Shell):
            return [topology]
        if isinstance(topology, Cell):
            return topology.Shells()
        if isinstance(topology, CellComplex):
            return topology.Shells()
        return []

    @staticmethod
    def Cells(topology: Any) -> list[Cell]:
        if topology is None:
            return []
        if isinstance(topology, Cell):
            return [topology]
        if isinstance(topology, CellComplex):
            return topology.Cells()
        return []

    @staticmethod
    def SelfMerge(topology: Any, *args, **kwargs) -> Any:
        tolerance = kwargs.get("tolerance", 1e-6)

        if isinstance(topology, Cluster):
            edges = [t for t in topology.topologies if isinstance(t, Edge)]
            faces = [t for t in topology.topologies if isinstance(t, Face)]
            cells = [t for t in topology.topologies if isinstance(t, Cell)]

            if edges and len(edges) == len(topology.topologies):
                return Wire.ByEdges(edges, tolerance=tolerance)
            if faces and len(faces) == len(topology.topologies):
                return Shell.ByFaces(faces)
            if cells and len(cells) == len(topology.topologies):
                return CellComplex.ByCells(cells)

            return topology

        return topology

    @staticmethod
    def Merge(topologyA: Any, topologyB: Any, *args) -> Any:
        if topologyA is None:
            return topologyB
        if topologyB is None:
            return topologyA

        if isinstance(topologyA, Cluster):
            a = topologyA.topologies
        else:
            a = [topologyA]

        if isinstance(topologyB, Cluster):
            b = topologyB.topologies
        else:
            b = [topologyB]

        return Cluster.ByTopologies(a + b)

    @staticmethod
    def DeepCopy(topology: Any) -> Any:
        return copy.deepcopy(topology)


class VertexUtility:
    @staticmethod
    def Distance(vertexA: Vertex, vertexB: Any) -> Optional[float]:
        if not isinstance(vertexA, Vertex):
            return None
        if isinstance(vertexB, Vertex):
            return _distance(vertexA, vertexB)
        verts = Topology.Vertices(vertexB)
        if not verts:
            return None
        return min(_distance(vertexA, v) for v in verts)


class EdgeUtility:
    @staticmethod
    def Length(edge: Edge) -> Optional[float]:
        if not isinstance(edge, Edge):
            return None
        return edge.Length()


class WireUtility:
    @staticmethod
    def IsClosed(wire: Wire) -> bool:
        return isinstance(wire, Wire) and wire.IsClosed()


class FaceUtility:
    @staticmethod
    def Area(face: Face) -> Optional[float]:
        if not isinstance(face, Face):
            return None
        return face.Area()


class ShellUtility:
    pass


class GraphUtility:
    @staticmethod
    def AdjacentVertices(graph: Graph, vertex: Vertex, *args, **kwargs) -> list[Vertex]:
        if not isinstance(graph, Graph) or not isinstance(vertex, Vertex):
            return []
        result = []
        for e in graph.edges:
            if _same_vertex(e.StartVertex(), vertex):
                result.append(e.EndVertex())
            elif _same_vertex(e.EndVertex(), vertex):
                result.append(e.StartVertex())
        return _dedupe(result)


# TopologicPy may call Core.Graph.AdjacentVertices or Core.GraphUtility.AdjacentVertices.
def _graph_adjacent_vertices(graph: Graph, vertex: Vertex, *args, **kwargs) -> list[Vertex]:
    return GraphUtility.AdjacentVertices(graph, vertex, *args, **kwargs)


Graph.AdjacentVertices = staticmethod(_graph_adjacent_vertices)


# Placeholder namespaces often imported by TopologicPy wrappers.
class Aperture(TopologyBase):
    pass


class Context(TopologyBase):
    pass


# -----------------------------------------------------------------------------
# Backend object
# -----------------------------------------------------------------------------

class PythonOCCBackend:
    """
    Backend namespace container consumed by topologicpy.Core.

    Usage depends on your Core.py implementation, but usually resembles:

        from topologicpy.Core import Core
        from topologicpy.PythonOCCBackend import PythonOCCBackend
        Core.SetBackend(PythonOCCBackend())

    or, if Core expects a class rather than an instance:

        Core.SetBackend(PythonOCCBackend)
    """

    Aperture = Aperture
    Cell = Cell
    CellComplex = CellComplex
    Cluster = Cluster
    Context = Context
    Dictionary = Dictionary
    DoubleAttribute = DoubleAttribute
    Edge = Edge
    EdgeUtility = EdgeUtility
    Face = Face
    FaceUtility = FaceUtility
    Graph = Graph
    GraphUtility = GraphUtility
    IntAttribute = IntAttribute
    ListAttribute = ListAttribute
    Shell = Shell
    ShellUtility = ShellUtility
    StringAttribute = StringAttribute
    Topology = Topology
    TopologyUtility = Topology
    Vertex = Vertex
    VertexUtility = VertexUtility
    Wire = Wire
    WireUtility = WireUtility

    def __init__(self):
        self.backend_name = "PythonOCCBackend"

    def __repr__(self) -> str:
        return "PythonOCCBackend"

    @staticmethod
    def Name() -> str:
        return "PythonOCCBackend"

    @staticmethod
    def Namespaces() -> list[str]:
        return [
            "Aperture", "Cell", "CellComplex", "Cluster", "Context",
            "Dictionary", "DoubleAttribute", "Edge", "EdgeUtility",
            "Face", "FaceUtility", "Graph", "GraphUtility",
            "IntAttribute", "ListAttribute", "Shell", "ShellUtility",
            "StringAttribute", "Topology", "TopologyUtility",
            "Vertex", "VertexUtility", "Wire", "WireUtility",
        ]


__all__ = [
    "PythonOCCBackend",
    "Aperture", "Cell", "CellComplex", "Cluster", "Context",
    "Dictionary", "DoubleAttribute", "Edge", "EdgeUtility",
    "Face", "FaceUtility", "Graph", "GraphUtility",
    "IntAttribute", "ListAttribute", "Shell", "ShellUtility",
    "StringAttribute", "Topology", "TopologyUtility",
    "Vertex", "VertexUtility", "Wire", "WireUtility",
]

TopologyUtility = Topology
