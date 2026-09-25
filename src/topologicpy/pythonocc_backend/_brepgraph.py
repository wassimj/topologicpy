#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Private OCCT 8 BRepGraph acceleration helpers for the PythonOCC backend.

This module is deliberately private and backend-specific.  It does not expose
BRepGraph through the public TopologicPy API; it provides an indexed incidence
layer around existing ``TopoDS_Shape`` objects.

The module is safe to import with pythonocc-core < 8.0.  In that case, or when
``TOPOLOGICPY_DISABLE_BREPGRAPH`` is truthy, every entry point returns ``None``
and callers should use their existing legacy traversal path.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

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
    from OCC.Core.BRepGraph import (
        brepgraph,
        BRepGraph_NodeId,
        BRepGraph_ChildExplorer,
    )
    _BREPGRAPH_IMPORTED = True
except Exception:  # pragma: no cover - pythonocc-core < 8.0 / no PythonOCC
    TopAbs_VERTEX = TopAbs_EDGE = TopAbs_WIRE = TopAbs_FACE = None
    TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPSOLID = TopAbs_COMPOUND = None
    brepgraph = None
    BRepGraph_NodeId = None
    BRepGraph_ChildExplorer = None
    _BREPGRAPH_IMPORTED = False


_CACHE_ATTR = "_topologicpy_brepgraph_index"
_CACHE_SIGNATURE_ATTR = "_topologicpy_brepgraph_signature"


def _truthy_env(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_available() -> bool:
    """Return whether the OCCT 8 BRepGraph path may be used."""
    return bool(_BREPGRAPH_IMPORTED and not _truthy_env("TOPOLOGICPY_DISABLE_BREPGRAPH"))


def _is_null_shape(shape: Any) -> bool:
    if shape is None:
        return True
    try:
        return bool(shape.IsNull())
    except Exception:
        return False


def _shape_signature(shape: Any):
    """Cheap identity signature used only to invalidate the per-wrapper cache."""
    if _is_null_shape(shape):
        return None
    try:
        shape_hash = hash(shape)
    except Exception:
        shape_hash = id(shape)
    try:
        shape_type = int(shape.ShapeType())
    except Exception:
        shape_type = None
    return id(shape), shape_hash, shape_type


def _kind_constant(name: str):
    """Resolve a BRepGraph_NodeId kind across SWIG enum spelling variants."""
    if BRepGraph_NodeId is None:
        return None
    for attr in (f"Kind_{name}", name):
        try:
            return getattr(BRepGraph_NodeId, attr)
        except Exception:
            pass
    try:
        return getattr(BRepGraph_NodeId.Kind, name)
    except Exception:
        return None


def _kind_for_shape_type(shape_type: Any):
    if not is_available() or shape_type is None:
        return None
    mapping = {
        TopAbs_VERTEX: _kind_constant("Vertex"),
        TopAbs_EDGE: _kind_constant("Edge"),
        TopAbs_WIRE: _kind_constant("Wire"),
        TopAbs_FACE: _kind_constant("Face"),
        TopAbs_SHELL: _kind_constant("Shell"),
        TopAbs_SOLID: _kind_constant("Solid"),
        TopAbs_COMPSOLID: _kind_constant("CompSolid"),
        TopAbs_COMPOUND: _kind_constant("Compound"),
    }
    return mapping.get(shape_type)


def _kind_for_shape(shape: Any):
    try:
        return _kind_for_shape_type(shape.ShapeType())
    except Exception:
        return None


def _kind_value(kind: Any):
    """Return a stable comparable value for a BRepGraph node-kind enum."""
    if kind is None:
        return None
    try:
        return int(kind)
    except Exception:
        pass
    try:
        return int(kind.value)
    except Exception:
        pass
    try:
        return str(kind)
    except Exception:
        return None


def _same_kind(a: Any, b: Any) -> bool:
    a_value = _kind_value(a)
    b_value = _kind_value(b)
    return a_value is not None and a_value == b_value


def _current_child_node(explorer: Any):
    """Return the current emitted child as a real ``BRepGraph_NodeId``.

    pythonocc-core 8.0.1 exposes ``BRepGraph_ChildExplorer.Current()`` as
    ``Any`` because ``BRepGraphInc::NodeInstance`` is not wrapped as a Python
    proxy class.  Do not use ``Current()`` here.

    OCCT's implementation stores the explicit root in stack frame 0 while
    ``NodeAt(0)`` addresses stack frame 1 (the first step below the root).
    During an emitted descendant, ``Depth() == current_frame + 1``; therefore
    the current node is ``NodeAt(Depth() - 2)``.
    """
    if explorer is None:
        return None
    try:
        depth = int(explorer.Depth())
    except Exception:
        return None

    # The two-argument explorer never emits the root itself, so a real current
    # descendant must have at least root + one child on the stack.
    if depth < 2:
        return None

    try:
        return explorer.NodeAt(depth - 2)
    except Exception:
        return None

def _node_key(node: Any):
    if node is None:
        return None
    try:
        kind = _kind_value(node.NodeKind)
        index = int(node.Index)
        if kind is None:
            return None
        return kind, index
    except Exception:
        return None


def _node_valid(node: Any, graph: Any = None) -> bool:
    """Return True when *node* is an active node in *graph*.

    ``NodeId.IsOwned(graph)`` is intentionally not used: it asks whether a
    node has a structural owner/parent, so a perfectly valid root is unowned.
    OCCT 8 provides the graph-level predicate we actually need through
    ``Topo().Gen().IsActive(node)``.
    """
    if node is None:
        return False

    try:
        if not bool(node.IsValid()):
            return False
    except Exception:
        return False

    if graph is None:
        return True

    try:
        return bool(graph.Topo().Gen().IsActive(node))
    except Exception:
        pass

    # Compatibility fallback for a wrapper lacking IsActive().
    try:
        if not bool(graph.Topo().Gen().IsValid(node)):
            return False
    except Exception:
        pass

    try:
        if bool(node.IsRemoved(graph)):
            return False
    except Exception:
        pass

    return True


def _same_shape(a: Any, b: Any) -> bool:
    if _is_null_shape(a) or _is_null_shape(b):
        return False
    try:
        return bool(a.IsSame(b))
    except Exception:
        return False


def _unique_shapes(shapes: Iterable[Any]) -> list:
    result = []
    buckets = {}
    for shape in shapes or []:
        if _is_null_shape(shape):
            continue
        try:
            key = hash(shape)
        except Exception:
            key = None
        if key is not None:
            bucket = buckets.setdefault(key, [])
            if any(_same_shape(shape, other) for other in bucket):
                continue
            bucket.append(shape)
            result.append(shape)
            continue
        if any(_same_shape(shape, other) for other in result):
            continue
        result.append(shape)
    return result


class BRepGraphIndex:
    """Lazy incidence index for one imported OCCT shape.

    The Python binding does not expose ``BRepGraphInc::NodeInstance`` as a
    usable proxy object.  Consequently this class never consumes
    ``ChildExplorer.Current()`` or ``ParentExplorer.Current()``.  Instead it
    performs one recursive child walk and records exact immediate incidence
    from the fully wrapped ``NodeAt`` and ``CurrentParent`` accessors.
    """

    def __init__(self, shape: Any):
        self.graph = None
        self.root = None
        self.roots = []
        self.shape_signature = _shape_signature(shape)
        self._incidence_ready = None
        self._children_by_parent = {}
        self._parents_by_child = {}
        self._nodes_by_key = {}
        self._build(shape)

    @property
    def valid(self) -> bool:
        return self.graph is not None and _node_valid(self.root, self.graph)

    def _reset_incidence(self) -> None:
        self._incidence_ready = None
        self._children_by_parent = {}
        self._parents_by_child = {}
        self._nodes_by_key = {}

    def _build(self, shape: Any) -> None:
        if not is_available() or _is_null_shape(shape):
            return
        try:
            graph = brepgraph()
            result = graph.Shapes().Add(shape)
            if hasattr(result, "IsOk") and not bool(result.IsOk()):
                return
            root = getattr(result, "TopologyRoot", None)
            if not _node_valid(root, graph):
                return
            self.graph = graph
            self.root = root
            self.roots = [root]
            self._reset_incidence()
        except Exception:
            self.graph = None
            self.root = None
            self.roots = []
            self._reset_incidence()

    @classmethod
    def _from_shapes(cls, shapes: Iterable[Any]):
        """Build one graph containing several roots; used for sharing queries."""
        obj = cls.__new__(cls)
        obj.graph = None
        obj.root = None
        obj.roots = []
        obj.shape_signature = None
        obj._incidence_ready = None
        obj._children_by_parent = {}
        obj._parents_by_child = {}
        obj._nodes_by_key = {}

        if not is_available():
            return obj, []

        roots = []
        try:
            graph = brepgraph()
            for shape in shapes or []:
                if _is_null_shape(shape):
                    return obj, []
                result = graph.Shapes().Add(shape)
                if hasattr(result, "IsOk") and not bool(result.IsOk()):
                    return obj, []
                root = getattr(result, "TopologyRoot", None)
                if not _node_valid(root, graph):
                    return obj, []
                roots.append(root)

            obj.graph = graph
            obj.root = roots[0] if roots else None
            obj.roots = list(roots)
            return obj, roots
        except Exception:
            return obj, []

    def _remember_node(self, node: Any) -> bool:
        if not _node_valid(node, self.graph):
            return False
        key = _node_key(node)
        if key is None:
            return False
        self._nodes_by_key[key] = node
        return True

    @staticmethod
    def _append_unique_node(mapping: dict, key: Any, node: Any) -> None:
        if key is None or node is None:
            return
        node_key = _node_key(node)
        if node_key is None:
            return
        bucket = mapping.setdefault(key, [])
        if all(_node_key(existing) != node_key for existing in bucket):
            bucket.append(node)

    def _ensure_incidence(self) -> bool:
        """Build direct parent/child incidence once using only wrapped accessors.

        ``None``/False means the BRepGraph path cannot be trusted and callers
        must fall back to the established TopExp/TopTools implementation.
        """
        if self._incidence_ready is not None:
            return bool(self._incidence_ready)
        if not self.valid:
            self._incidence_ready = False
            return False

        self._children_by_parent = {}
        self._parents_by_child = {}
        self._nodes_by_key = {}

        roots = [r for r in (self.roots or [self.root]) if _node_valid(r, self.graph)]
        if not roots:
            self._incidence_ready = False
            return False

        for root in roots:
            self._remember_node(root)

        emitted_count = 0
        try:
            for root in roots:
                explorer = BRepGraph_ChildExplorer(self.graph, root)
                while explorer.More():
                    child = _current_child_node(explorer)
                    parent = explorer.CurrentParent()

                    if not _node_valid(child, self.graph):
                        self._incidence_ready = False
                        return False
                    if not _node_valid(parent, self.graph):
                        # The unfiltered two-argument explorer does not emit the
                        # root, so every emitted item must have a real parent.
                        self._incidence_ready = False
                        return False

                    self._remember_node(parent)
                    self._remember_node(child)
                    parent_key = _node_key(parent)
                    child_key = _node_key(child)
                    self._append_unique_node(self._children_by_parent, parent_key, child)
                    self._append_unique_node(self._parents_by_child, child_key, parent)
                    emitted_count += 1
                    explorer.Next()
        except Exception:
            self._incidence_ready = False
            return False

        # A graph with nodes beyond its explicit roots must expose at least one
        # traversal edge.  This catches binding/proxy mismatches and, crucially,
        # returns "cannot answer" rather than a false authoritative empty list.
        try:
            total_nodes = int(self.graph.Topo().Gen().NbNodes())
        except Exception:
            total_nodes = None
        unique_root_count = len({_node_key(r) for r in roots if _node_key(r) is not None})
        if total_nodes is not None and total_nodes > unique_root_count and emitted_count == 0:
            self._incidence_ready = False
            return False

        self._incidence_ready = True
        return True

    def _shape_of_node(self, node: Any):
        if not self.valid or not _node_valid(node, self.graph):
            return None
        shapes = self.graph.Shapes()
        try:
            if shapes.HasOriginal(node):
                original = shapes.Original(node)
                if not _is_null_shape(original):
                    return original
        except Exception:
            pass
        try:
            shape = shapes.Shape(node)
            if not _is_null_shape(shape):
                return shape
        except Exception:
            pass
        try:
            shape = shapes.Reconstruct(node)
            if not _is_null_shape(shape):
                return shape
        except Exception:
            pass
        return None

    def _children(self, node: Any, target_kind: Any) -> Optional[list]:
        """Return all descendant definition nodes of ``target_kind``."""
        if (
            not self.valid
            or not _node_valid(node, self.graph)
            or target_kind is None
            or not self._ensure_incidence()
        ):
            return None

        result = []
        seen_nodes = set()
        stack = list(self._children_by_parent.get(_node_key(node), []))

        while stack:
            child = stack.pop()
            child_key = _node_key(child)
            if child_key is None or child_key in seen_nodes:
                continue
            seen_nodes.add(child_key)

            if _same_kind(getattr(child, "NodeKind", None), target_kind):
                result.append(child)

            stack.extend(self._children_by_parent.get(child_key, []))

        return result

    def _parents(self, node: Any, target_kind: Any) -> Optional[list]:
        """Return all ancestor definition nodes of ``target_kind``."""
        if (
            not self.valid
            or not _node_valid(node, self.graph)
            or target_kind is None
            or not self._ensure_incidence()
        ):
            return None

        result = []
        seen_nodes = set()
        stack = list(self._parents_by_child.get(_node_key(node), []))

        while stack:
            parent = stack.pop()
            parent_key = _node_key(parent)
            if parent_key is None or parent_key in seen_nodes:
                continue
            seen_nodes.add(parent_key)

            if _same_kind(getattr(parent, "NodeKind", None), target_kind):
                result.append(parent)

            stack.extend(self._parents_by_child.get(parent_key, []))

        return result

    def _nodes_from_root(
        self,
        root: Any,
        target_kind: Any,
        include_root: bool = True,
    ) -> Optional[list]:
        if not self.valid or not _node_valid(root, self.graph) or target_kind is None:
            return None

        result = []
        try:
            if include_root and _same_kind(root.NodeKind, target_kind):
                result.append(root)
        except Exception:
            pass

        children = self._children(root, target_kind)
        if children is None:
            return None
        result.extend(children)

        seen = set()
        unique = []
        for node in result:
            key = _node_key(node)
            if key is not None and key not in seen:
                seen.add(key)
                unique.append(node)
        return unique

    def node_for_shape(self, shape: Any):
        """Return a graph node for an exact host-side OCCT shape, or None."""
        if not self.valid or _is_null_shape(shape):
            return None
        try:
            node = self.graph.Shapes().FindNode(shape)
            if _node_valid(node, self.graph):
                return node
        except Exception:
            pass

        target_kind = _kind_for_shape(shape)
        if target_kind is None or not self._ensure_incidence():
            return None

        for node in self._nodes_by_key.values():
            if not _same_kind(getattr(node, "NodeKind", None), target_kind):
                continue
            candidate = self._shape_of_node(node)
            if _same_shape(candidate, shape):
                return node
        return None

    def subshapes(self, source_shape: Any, target_shape_type: Any) -> Optional[list]:
        """Return exact descendant shapes, or None when this index cannot answer."""
        if not self.valid:
            return None
        target_kind = _kind_for_shape_type(target_shape_type)
        if target_kind is None:
            return None
        source_node = self.node_for_shape(source_shape)
        if source_node is None:
            return None
        nodes = self._nodes_from_root(source_node, target_kind, include_root=True)
        if nodes is None:
            return None
        shapes = _unique_shapes(self._shape_of_node(node) for node in nodes)
        if nodes and not shapes:
            return None
        return shapes

    def super_shapes(self, source_shape: Any, target_shape_type: Any) -> Optional[list]:
        """Return exact ancestors of source_shape, or None if this index cannot answer."""
        if not self.valid:
            return None
        target_kind = _kind_for_shape_type(target_shape_type)
        if target_kind is None:
            return None
        source_node = self.node_for_shape(source_shape)
        if source_node is None:
            return None
        nodes = self._parents(source_node, target_kind)
        if nodes is None:
            return None
        shapes = _unique_shapes(self._shape_of_node(node) for node in nodes)
        if nodes and not shapes:
            return None
        return shapes

    def ancestor_count(self, source_shape: Any, ancestor_shape_type: Any) -> Optional[int]:
        """Return the number of distinct ancestor definitions of a requested kind.

        ``None`` means the graph cannot answer.  A numeric zero is authoritative.
        Definition-node counting is intentional: occurrence paths are not double-counted.
        """
        if not self.valid:
            return None
        ancestor_kind = _kind_for_shape_type(ancestor_shape_type)
        if ancestor_kind is None:
            return None
        source_node = self.node_for_shape(source_shape)
        if source_node is None:
            return None
        ancestors = self._parents(source_node, ancestor_kind)
        if ancestors is None:
            return None
        return len({_node_key(node) for node in ancestors if _node_key(node) is not None})

    def subshapes_by_ancestor_count(
        self,
        sub_shape_type: Any,
        ancestor_shape_type: Any,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ) -> Optional[list]:
        """Classify host subshapes by distinct ancestor incidence.

        This is the core Tranche-2 CellComplex primitive.  For example,
        ``Face`` subshapes with exactly one ``Solid`` ancestor are external
        boundary faces, while faces with two or more ``Solid`` ancestors are
        internal/non-manifold faces.

        The method is deliberately single-root.  Multi-root temporary graphs
        are used by ``shared_shapes`` and do not have one meaningful host
        incidence classification.
        """
        if not self.valid:
            return None
        roots = [r for r in (self.roots or [self.root]) if _node_valid(r, self.graph)]
        if len(roots) != 1:
            return None

        sub_kind = _kind_for_shape_type(sub_shape_type)
        ancestor_kind = _kind_for_shape_type(ancestor_shape_type)
        if sub_kind is None or ancestor_kind is None:
            return None

        try:
            min_value = None if min_count is None else int(min_count)
            max_value = None if max_count is None else int(max_count)
        except Exception:
            return None
        if min_value is not None and min_value < 0:
            return None
        if max_value is not None and max_value < 0:
            return None
        if min_value is not None and max_value is not None and min_value > max_value:
            return None

        nodes = self._nodes_from_root(roots[0], sub_kind, include_root=True)
        if nodes is None:
            return None

        result = []
        for node in nodes:
            ancestors = self._parents(node, ancestor_kind)
            if ancestors is None:
                return None
            count = len({_node_key(a) for a in ancestors if _node_key(a) is not None})
            if min_value is not None and count < min_value:
                continue
            if max_value is not None and count > max_value:
                continue
            shape = self._shape_of_node(node)
            if _is_null_shape(shape):
                return None
            result.append(shape)
        return _unique_shapes(result)

    def _edge_incidence_for_node(self, edge_node: Any) -> Optional[dict]:
        """Return CoEdge-use and unique-Face incidence for an Edge node."""
        if not self.valid or not _node_valid(edge_node, self.graph):
            return None
        edge_kind = _kind_constant("Edge")
        coedge_kind = _kind_constant("CoEdge")
        face_kind = _kind_constant("Face")
        if edge_kind is None or coedge_kind is None or face_kind is None:
            return None
        try:
            if not _same_kind(edge_node.NodeKind, edge_kind):
                return None
        except Exception:
            return None

        coedges = self._parents(edge_node, coedge_kind)
        faces = self._parents(edge_node, face_kind)
        if coedges is None or faces is None:
            return None

        coedge_keys = {_node_key(node) for node in coedges if _node_key(node) is not None}
        face_nodes = []
        seen_faces = set()
        for node in faces:
            key = _node_key(node)
            if key is None or key in seen_faces:
                continue
            seen_faces.add(key)
            face_nodes.append(node)

        face_shapes = []
        for node in face_nodes:
            shape = self._shape_of_node(node)
            if _is_null_shape(shape):
                return None
            face_shapes.append(shape)

        return {
            "use_count": len(coedge_keys),
            "face_count": len(face_nodes),
            "face_shapes": _unique_shapes(face_shapes),
        }

    def edge_incidence(self, edge_shape: Any) -> Optional[dict]:
        """Return exact BRepGraph incidence for one Edge.

        ``use_count`` counts distinct CoEdges (uses of the Edge in Wires/Faces),
        while ``face_count`` counts distinct owning Faces.  The distinction is
        essential for seam edges: a seam commonly has two CoEdge uses but only
        one unique owning Face.
        """
        if not self.valid:
            return None
        edge_node = self.node_for_shape(edge_shape)
        if edge_node is None:
            return None
        return self._edge_incidence_for_node(edge_node)

    def edge_shapes_by_incidence(
        self,
        min_use_count: Optional[int] = None,
        max_use_count: Optional[int] = None,
        min_face_count: Optional[int] = None,
        max_face_count: Optional[int] = None,
    ) -> Optional[list]:
        """Return host Edges whose CoEdge/Face incidence is within the ranges.

        Typical queries:
          * free/open-shell boundary edges: use_count == 1
          * true shared/internal shell edges: face_count >= 2
          * seam edges: use_count >= 2 and face_count == 1
        """
        if not self.valid:
            return None
        roots = [r for r in (self.roots or [self.root]) if _node_valid(r, self.graph)]
        if len(roots) != 1:
            return None
        edge_kind = _kind_constant("Edge")
        if edge_kind is None:
            return None

        try:
            limits = [
                None if value is None else int(value)
                for value in (
                    min_use_count,
                    max_use_count,
                    min_face_count,
                    max_face_count,
                )
            ]
        except Exception:
            return None
        min_use, max_use, min_faces, max_faces = limits
        if any(value is not None and value < 0 for value in limits):
            return None
        if min_use is not None and max_use is not None and min_use > max_use:
            return None
        if min_faces is not None and max_faces is not None and min_faces > max_faces:
            return None

        nodes = self._nodes_from_root(roots[0], edge_kind, include_root=True)
        if nodes is None:
            return None

        result = []
        for node in nodes:
            info = self._edge_incidence_for_node(node)
            if info is None:
                return None
            uses = int(info["use_count"])
            face_count = int(info["face_count"])
            if min_use is not None and uses < min_use:
                continue
            if max_use is not None and uses > max_use:
                continue
            if min_faces is not None and face_count < min_faces:
                continue
            if max_faces is not None and face_count > max_faces:
                continue
            shape = self._shape_of_node(node)
            if _is_null_shape(shape):
                return None
            result.append(shape)
        return _unique_shapes(result)

    def adjacent_shapes(self, source_shape: Any, target_shape_type: Any) -> Optional[list]:
        """Return same-dimensional topological neighbours inside this indexed host."""
        if not self.valid:
            return None
        target_kind = _kind_for_shape_type(target_shape_type)
        source_node = self.node_for_shape(source_shape)
        if target_kind is None or source_node is None:
            return None
        try:
            if not _same_kind(source_node.NodeKind, target_kind):
                return None
        except Exception:
            return None

        vertex_kind = _kind_constant("Vertex")
        edge_kind = _kind_constant("Edge")
        wire_kind = _kind_constant("Wire")
        face_kind = _kind_constant("Face")
        shell_kind = _kind_constant("Shell")
        solid_kind = _kind_constant("Solid")
        neighbours = []

        if _same_kind(target_kind, vertex_kind):
            edges = self._parents(source_node, edge_kind)
            if edges is None:
                return None
            for edge in edges:
                nodes = self._children(edge, vertex_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        elif _same_kind(target_kind, edge_kind):
            vertices = self._children(source_node, vertex_kind)
            if vertices is None:
                return None
            for vertex in vertices:
                nodes = self._parents(vertex, edge_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        elif _same_kind(target_kind, wire_kind):
            edges = self._children(source_node, edge_kind)
            if edges is None:
                return None
            for edge in edges:
                nodes = self._parents(edge, wire_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        elif _same_kind(target_kind, face_kind):
            edges = self._children(source_node, edge_kind)
            if edges is None:
                return None
            for edge in edges:
                nodes = self._parents(edge, face_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        elif _same_kind(target_kind, shell_kind):
            faces = self._children(source_node, face_kind)
            if faces is None:
                return None
            for face in faces:
                nodes = self._parents(face, shell_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        elif _same_kind(target_kind, solid_kind):
            faces = self._children(source_node, face_kind)
            if faces is None:
                return None
            for face in faces:
                nodes = self._parents(face, solid_kind)
                if nodes is None:
                    return None
                neighbours.extend(nodes)
        else:
            return None

        source_key = _node_key(source_node)
        seen = set()
        shapes = []
        for node in neighbours:
            key = _node_key(node)
            if key is None or key == source_key or key in seen:
                continue
            seen.add(key)
            shape = self._shape_of_node(node)
            if not _is_null_shape(shape):
                shapes.append(shape)
        return _unique_shapes(shapes)

def cached_index(owner: Any, shape: Any) -> Optional[BRepGraphIndex]:
    """Return a lazily built per-wrapper BRepGraph index.

    The cache lives on the host TopologicPy wrapper rather than globally.  This
    avoids retaining arbitrary CAD models and makes invalidation trivial when a
    wrapper's ``shape`` reference changes.
    """
    if not is_available() or _is_null_shape(shape):
        return None

    signature = _shape_signature(shape)
    if owner is not None:
        try:
            cached = getattr(owner, _CACHE_ATTR, None)
            cached_signature = getattr(owner, _CACHE_SIGNATURE_ATTR, None)
            if (
                isinstance(cached, BRepGraphIndex)
                and cached.valid
                and cached_signature == signature
            ):
                return cached
        except Exception:
            pass

    index = BRepGraphIndex(shape)
    if not index.valid:
        return None

    if owner is not None:
        try:
            setattr(owner, _CACHE_ATTR, index)
            setattr(owner, _CACHE_SIGNATURE_ATTR, signature)
        except Exception:
            pass
    return index


def shared_shapes(shape_a: Any, shape_b: Any, target_shape_type: Any) -> Optional[list]:
    """Return exact OCCT subshapes genuinely shared by two topology roots.

    One temporary BRepGraph is populated with both roots.  The descendants are
    obtained through graph traversal; matching is then done with hash buckets
    plus ``TopoDS_Shape.IsSame``.  The bucketed confirmation keeps the method
    correct even if an OCCT build chooses not to canonicalise a repeated input
    definition during ``Shapes().Add``.
    """
    if not is_available() or _is_null_shape(shape_a) or _is_null_shape(shape_b):
        return None
    target_kind = _kind_for_shape_type(target_shape_type)
    if target_kind is None:
        return None

    index, roots = BRepGraphIndex._from_shapes((shape_a, shape_b))
    if not index.valid or len(roots) != 2:
        return None

    def node_shapes(root):
        nodes = index._nodes_from_root(root, target_kind, include_root=True)
        if nodes is None:
            return None
        result = []
        for node in nodes:
            shape = index._shape_of_node(node)
            if not _is_null_shape(shape):
                result.append(shape)
        return _unique_shapes(result)

    left = node_shapes(roots[0])
    right = node_shapes(roots[1])
    if left is None or right is None:
        return None

    right_buckets = {}
    right_unhashed = []
    for shape in right:
        try:
            right_buckets.setdefault(hash(shape), []).append(shape)
        except Exception:
            right_unhashed.append(shape)

    result = []
    for shape in left:
        candidates = right_unhashed
        try:
            candidates = right_buckets.get(hash(shape), [])
        except Exception:
            pass
        # IsSame shapes normally hash identically, but do not make correctness
        # depend on that implementation detail across OCCT wrapper versions.
        if not candidates:
            candidates = right
        if any(_same_shape(shape, candidate) for candidate in candidates):
            result.append(shape)

    return _unique_shapes(result)
