# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# Private PythonOCC support for topologicpy.CSG.

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Optional


_ACTIVE_CAPTURE = ContextVar("topologicpy_csg_lineage_capture", default=None)


@contextmanager
def capture(operation_node: int, role_nodes: dict, *, stage: int = 0, sink: Optional[list] = None):
    """Activate exact native lineage capture for one CSG evaluation stage."""
    state = {
        "operation_node": operation_node,
        "role_nodes": {str(k).lower(): v for k, v in (role_nodes or {}).items()},
        "stage": int(stage),
        "sink": sink if isinstance(sink, list) else [],
    }
    token = _ACTIVE_CAPTURE.set(state)
    try:
        yield state
    finally:
        _ACTIVE_CAPTURE.reset(token)


def is_active() -> bool:
    return _ACTIVE_CAPTURE.get() is not None


def _normalise_sources(sources: Iterable[Any]) -> list[dict]:
    result = []
    for index, source in enumerate(sources or []):
        role = f"source{index}"
        shape = None
        if isinstance(source, dict):
            role = str(source.get("role", role))
            shape = source.get("shape")
        elif isinstance(source, (tuple, list)):
            if len(source) > 0:
                role = str(source[0])
            if len(source) > 1:
                shape = source[1]
        else:
            shape = source
        if shape is not None:
            result.append({"role": role, "shape": shape})
    return result


def _shape_type_name(shape) -> Optional[str]:
    if shape is None:
        return None
    try:
        from OCC.Core.TopAbs import (
            TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE,
            TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPSOLID, TopAbs_COMPOUND,
        )
        mapping = {
            TopAbs_VERTEX: "Vertex",
            TopAbs_EDGE: "Edge",
            TopAbs_WIRE: "Wire",
            TopAbs_FACE: "Face",
            TopAbs_SHELL: "Shell",
            TopAbs_SOLID: "Cell",
            TopAbs_COMPSOLID: "CellComplex",
            TopAbs_COMPOUND: "Cluster",
        }
        return mapping.get(shape.ShapeType(), str(shape.ShapeType()))
    except Exception:
        return None


def materialise_record(record: dict) -> dict:
    """Convert one private native lineage record to a public neutral record."""
    if not isinstance(record, dict):
        return {}
    result = {
        key: value
        for key, value in record.items()
        if key not in {"_sourceShape", "_resultShape"}
    }
    try:
        from .topology import Topology
        source_shape = record.get("_sourceShape")
        result_shape = record.get("_resultShape")
        result["source"] = Topology.ByOcctShape(source_shape) if source_shape is not None else None
        result["result"] = Topology.ByOcctShape(result_shape) if result_shape is not None else None
    except Exception:
        result["source"] = None
        result["result"] = None
    return result

def _source_entities(shape) -> list:
    """Return unique root + native descendants for history interrogation."""
    try:
        from OCC.Core.TopAbs import (
            TopAbs_COMPOUND,
            TopAbs_COMPSOLID,
            TopAbs_SOLID,
            TopAbs_SHELL,
            TopAbs_FACE,
            TopAbs_WIRE,
            TopAbs_EDGE,
            TopAbs_VERTEX,
        )
        from ._provenance import _iter_unique_subshapes, _is_null_shape, _same_shape, _shape_hash
    except Exception:
        return []

    if _is_null_shape(shape):
        return []

    result = []
    buckets = {}

    def add(item):
        if _is_null_shape(item):
            return
        try:
            shape_type = item.ShapeType()
        except Exception:
            shape_type = None
        key = (shape_type, _shape_hash(item))
        bucket = buckets.setdefault(key, [])
        if any(_same_shape(item, existing) for existing in bucket):
            return
        bucket.append(item)
        result.append(item)

    add(shape)
    for shape_type in (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_SOLID,
        TopAbs_SHELL,
        TopAbs_FACE,
        TopAbs_WIRE,
        TopAbs_EDGE,
        TopAbs_VERTEX,
    ):
        for item in _iter_unique_subshapes(shape, shape_type):
            add(item)
    return result


def _unique_targets(items):
    try:
        from ._provenance import _is_null_shape, _same_shape, _shape_hash
    except Exception:
        return list(items or [])
    result = []
    buckets = {}
    for shape, relation in items or []:
        if _is_null_shape(shape):
            continue
        try:
            shape_type = shape.ShapeType()
        except Exception:
            shape_type = None
        key = (shape_type, _shape_hash(shape))
        bucket = buckets.setdefault(key, [])
        if any(_same_shape(shape, existing) for existing in bucket):
            continue
        bucket.append(shape)
        result.append((shape, relation))
    return result


def _source_node_for(state: dict, role: str, operation: str):
    role_l = str(role or "").lower()
    # A provenance phase such as Union.UnifySameDomain consumes an internal
    # intermediate produced by the same expression node, not the original A.
    if "." in str(operation or "") and role_l == "source":
        return state.get("operation_node")
    mapping = state.get("role_nodes") or {}
    if role_l in mapping:
        return mapping[role_l]
    if role_l in {"other", "tool"}:
        return mapping.get("other", mapping.get("tool"))
    if role_l in {"self", "source"}:
        return mapping.get("self", mapping.get("source"))
    return state.get("operation_node")


def _append_record(
    state: dict,
    *,
    operation: str,
    role: str,
    relation: str,
    source_shape,
    result_shape,
    used_brepgraph: bool,
):
    sink = state.get("sink")
    if not isinstance(sink, list):
        return
    sink.append(
        {
            "operationNode": state.get("operation_node"),
            "sourceNode": _source_node_for(state, role, operation),
            "sourceRole": role,
            "stage": state.get("stage", 0),
            "operation": str(operation or ""),
            "relation": str(relation),
            "_sourceShape": source_shape,
            "_resultShape": result_shape,
            "sourceType": _shape_type_name(source_shape),
            "resultType": _shape_type_name(result_shape),
            "usedBRepGraph": bool(used_brepgraph),
        }
    )


def capture_history_if_active(result_shape, history, sources, operation: str = "History") -> bool:
    """Capture exact BRepTools_History lineage when a CSG context is active.

    Unlike dictionary propagation, lineage intentionally retains cross-
    dimensional Generated/Modified images.  CSG history is descriptive; it is
    not used to decide automatic semantic dictionary transfer.
    """
    state = _ACTIVE_CAPTURE.get()
    if state is None:
        return False

    try:
        from OCC.Core.BRepTools import BRepTools_History
        from ._provenance import _ResultShapeIndex, _toptools_to_list
    except Exception:
        return False

    try:
        target_index = _ResultShapeIndex(result_shape)
    except Exception:
        return False

    for source in _normalise_sources(sources):
        role = source["role"]
        for source_entity in _source_entities(source["shape"]):
            modified = []
            generated = []
            deleted = False
            supported = history is not None
            if supported:
                try:
                    supported = bool(BRepTools_History.IsSupportedType(source_entity))
                except Exception:
                    supported = True
            if history is not None and supported:
                try:
                    modified = _toptools_to_list(history.Modified(source_entity))
                except Exception:
                    modified = []
                try:
                    generated = _toptools_to_list(history.Generated(source_entity))
                except Exception:
                    generated = []
                try:
                    deleted = bool(history.IsRemoved(source_entity))
                except Exception:
                    deleted = False

            targets = []
            for image in modified:
                target = target_index.resolve(image)
                if target is not None:
                    targets.append((target, "modified"))
            for image in generated:
                target = target_index.resolve(image)
                if target is not None:
                    targets.append((target, "generated"))

            if not targets and not deleted:
                unchanged = target_index.resolve(source_entity)
                if unchanged is not None:
                    targets.append((unchanged, "unchanged"))

            if deleted and not targets:
                _append_record(
                    state,
                    operation=operation,
                    role=role,
                    relation="deleted",
                    source_shape=source_entity,
                    result_shape=None,
                    used_brepgraph=target_index.used_brepgraph,
                )
                continue

            for target, relation in _unique_targets(targets):
                _append_record(
                    state,
                    operation=operation,
                    role=role,
                    relation=relation,
                    source_shape=source_entity,
                    result_shape=target,
                    used_brepgraph=target_index.used_brepgraph,
                )
    return True


def capture_modifier_if_active(
    source_shape,
    result_shape,
    modifier,
    operation: str = "Modifier",
) -> bool:
    """Capture exact ModifiedShape lineage for transforms/copies."""
    state = _ACTIVE_CAPTURE.get()
    if state is None:
        return False
    try:
        from ._provenance import _ResultShapeIndex, _same_shape
        target_index = _ResultShapeIndex(result_shape)
    except Exception:
        return False

    for source_entity in _source_entities(source_shape):
        target = None
        relation = "modified"
        try:
            candidate = modifier.ModifiedShape(source_entity)
            target = target_index.resolve(candidate)
        except Exception:
            target = None
        if target is None:
            target = target_index.resolve(source_entity)
            if target is not None:
                relation = "unchanged"
        elif _same_shape(source_entity, target):
            relation = "unchanged"
        if target is None:
            continue
        _append_record(
            state,
            operation=operation,
            role="source",
            relation=relation,
            source_shape=source_entity,
            result_shape=target,
            used_brepgraph=target_index.used_brepgraph,
        )
    return True
