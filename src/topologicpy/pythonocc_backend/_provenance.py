#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Private metadata-provenance engine for the PythonOCC backend.

Tranche 3 establishes one internal contract for dictionary lineage:

* AttributeManager remains the metadata store.
* OCCT history / modifier mappings are the authority for newly-created shapes.
* BRepGraph is used as the preferred result-membership index when available.
* Exact native identity handles unchanged shapes.
* No geometric-nearest/centroid heuristic is used in this module.

The module is intentionally private.  It does not add a new public TopologicPy
API and is safe to import when BRepGraph is unavailable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .attribute_manager import AttributeManager

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
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
    from OCC.Core.BRepTools import BRepTools_History
except Exception:  # pragma: no cover - permits backend import without PythonOCC
    TopAbs_VERTEX = TopAbs_EDGE = TopAbs_WIRE = TopAbs_FACE = None
    TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPSOLID = TopAbs_COMPOUND = None
    TopExp_Explorer = None
    TopTools_ListIteratorOfListOfShape = None
    BRepTools_History = None


_SHAPE_TYPES = (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_WIRE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPSOLID,
    TopAbs_COMPOUND,
)

_HISTORY_TYPES = (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SOLID,
)


@dataclass
class ProvenanceReport:
    operation: str = ""
    source_entities: int = 0
    mapped_entities: int = 0
    unchanged_entities: int = 0
    modified_images: int = 0
    generated_images: int = 0
    deleted_entities: int = 0
    target_entities_written: int = 0
    conflicts: int = 0
    used_brepgraph_index: bool = False

    def as_dict(self) -> dict:
        return {
            "operation": self.operation,
            "source_entities": self.source_entities,
            "mapped_entities": self.mapped_entities,
            "unchanged_entities": self.unchanged_entities,
            "modified_images": self.modified_images,
            "generated_images": self.generated_images,
            "deleted_entities": self.deleted_entities,
            "target_entities_written": self.target_entities_written,
            "conflicts": self.conflicts,
            "used_brepgraph_index": self.used_brepgraph_index,
        }


def _is_null_shape(shape: Any) -> bool:
    if shape is None:
        return True
    try:
        return bool(shape.IsNull())
    except Exception:
        return False


def _same_shape(a: Any, b: Any) -> bool:
    if _is_null_shape(a) or _is_null_shape(b):
        return False
    try:
        return bool(a.IsSame(b))
    except Exception:
        return False


def _shape_hash(shape: Any):
    if _is_null_shape(shape):
        return None
    try:
        return hash(shape)
    except Exception:
        return None


def _iter_unique_subshapes(shape: Any, shape_type: Any) -> list:
    if _is_null_shape(shape) or TopExp_Explorer is None or shape_type is None:
        return []
    result = []
    buckets = {}
    try:
        explorer = TopExp_Explorer(shape, shape_type)
        while explorer.More():
            current = explorer.Current()
            key = _shape_hash(current)
            if key is None:
                if not any(_same_shape(current, existing) for existing in result):
                    result.append(current)
            else:
                bucket = buckets.setdefault(key, [])
                if not any(_same_shape(current, existing) for existing in bucket):
                    bucket.append(current)
                    result.append(current)
            explorer.Next()
    except Exception:
        return []
    return result


def _toptools_to_list(value: Any) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except Exception:
        pass
    if TopTools_ListIteratorOfListOfShape is None:
        return []
    result = []
    try:
        iterator = TopTools_ListIteratorOfListOfShape(value)
        while iterator.More():
            result.append(iterator.Value())
            iterator.Next()
    except Exception:
        return []
    return result


def _unwrap_dictionary_value(value: Any):
    """Return a canonical Python value for a backend/Core attribute value.

    TopologicPy's PythonOCC Dictionary stores scalar/list values as
    IntAttribute, DoubleAttribute, StringAttribute and ListAttribute objects.
    Provenance must never persist those wrappers inside a plain Python dict:
    doing so leaks backend implementation objects through Dictionary.ValueAtKey
    and also breaks equality/sorting/merge semantics.
    """
    if value is None:
        return None

    # PythonOCC / topologic_core attribute protocol.
    for accessor in ("IntValue", "DoubleValue", "StringValue"):
        fn = getattr(value, accessor, None)
        if callable(fn):
            try:
                result = fn()
                if accessor == "StringValue" and result == "__NONE__":
                    return None
                return result
            except Exception:
                pass

    fn = getattr(value, "ListValue", None)
    if callable(fn):
        try:
            return [_unwrap_dictionary_value(item) for item in list(fn() or [])]
        except Exception:
            pass

    # Compatibility with other backend attribute protocols.
    fn = getattr(value, "Value", None)
    if callable(fn):
        try:
            return _unwrap_dictionary_value(fn())
        except Exception:
            pass

    if hasattr(value, "value"):
        try:
            return _unwrap_dictionary_value(value.value)
        except Exception:
            pass

    if isinstance(value, tuple):
        return [_unwrap_dictionary_value(item) for item in value]
    if isinstance(value, list):
        return [_unwrap_dictionary_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap_dictionary_value(item) for key, item in value.items()}

    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _to_python_dict(dictionary: Any) -> dict:
    """Normalize any supported dictionary representation to Python values."""
    if dictionary is None:
        return {}

    if isinstance(dictionary, dict):
        return {
            key: _unwrap_dictionary_value(value)
            for key, value in dictionary.items()
        }

    # Prefer a backend dictionary's own Python conversion when present, then
    # normalize again defensively in case that converter returns attributes.
    fn = getattr(dictionary, "PythonDictionary", None)
    if callable(fn):
        try:
            converted = fn()
            if isinstance(converted, dict):
                return {
                    key: _unwrap_dictionary_value(value)
                    for key, value in converted.items()
                }
        except Exception:
            pass

    raw_data = None
    if hasattr(dictionary, "_data") and isinstance(getattr(dictionary, "_data"), dict):
        raw_data = getattr(dictionary, "_data")
    elif hasattr(dictionary, "data") and isinstance(getattr(dictionary, "data"), dict):
        raw_data = getattr(dictionary, "data")

    if raw_data is not None:
        return {
            key: _unwrap_dictionary_value(value)
            for key, value in raw_data.items()
        }

    if hasattr(dictionary, "Keys") and hasattr(dictionary, "ValueAtKey"):
        try:
            return {
                key: _unwrap_dictionary_value(dictionary.ValueAtKey(key))
                for key in dictionary.Keys()
            }
        except Exception:
            return {}

    if hasattr(dictionary, "keys") and hasattr(dictionary, "__getitem__"):
        try:
            return {
                key: _unwrap_dictionary_value(dictionary[key])
                for key in dictionary.keys()
            }
        except Exception:
            return {}

    return {}


def merge_dictionaries(dictionaries: Iterable[Any], conflict: str = "first") -> tuple[dict, int]:
    """Merge dictionaries deterministically and return ``(dictionary, conflicts)``.

    Tranche 3 intentionally does not use ``dict.update`` for provenance merges.
    The default ``first`` policy means earlier provenance sources have priority.
    Equal repeated values are not counted as conflicts.
    """
    conflict = str(conflict or "first").strip().lower()
    if conflict not in {"first", "last"}:
        conflict = "first"

    result = {}
    conflicts = 0
    for dictionary in dictionaries or []:
        incoming = _to_python_dict(dictionary)
        for key, value in incoming.items():
            if key not in result:
                try:
                    result[key] = copy.deepcopy(value)
                except Exception:
                    result[key] = value
                continue
            if result[key] == value:
                continue
            conflicts += 1
            if conflict == "last":
                try:
                    result[key] = copy.deepcopy(value)
                except Exception:
                    result[key] = value
    return result, conflicts


def _normalize_sources(sources: Iterable[Any]) -> list[dict]:
    result = []
    for index, source in enumerate(sources or []):
        role = f"source{index}"
        shape = None
        dictionary = None
        root_dictionary_value = None

        if isinstance(source, dict):
            role = str(source.get("role", role))
            shape = source.get("shape", None)
            dictionary = source.get("dictionary", None)
            if "root_dictionary" in source:
                root_dictionary_value = source.get("root_dictionary", None)
            else:
                root_dictionary_value = dictionary
        elif isinstance(source, (tuple, list)):
            if len(source) >= 1:
                role = str(source[0])
            if len(source) >= 2:
                shape = source[1]
            if len(source) >= 3:
                dictionary = source[2]
                root_dictionary_value = dictionary
            if len(source) >= 4:
                root_dictionary_value = source[3]
        else:
            shape = source

        if _is_null_shape(shape):
            continue
        result.append(
            {
                "role": role,
                "shape": shape,
                "dictionary": dictionary,
                "root_dictionary": root_dictionary_value,
            }
        )
    return result


def root_dictionary(sources: Iterable[Any], policy: str = "merge", conflict: str = "first") -> tuple[dict, int]:
    sources = _normalize_sources(sources)
    policy = str(policy or "merge").strip().lower()

    if not sources or policy in {"none", "empty"}:
        return {}, 0

    def root_value(source):
        value = source.get("root_dictionary", None)
        if value is None:
            value = source.get("dictionary", None)
        return value

    if policy in {"self", "source", "first"}:
        for source in sources:
            value = root_value(source)
            if value is not None:
                return _to_python_dict(value), 0
        return {}, 0

    if policy.startswith("role:"):
        role = policy.split(":", 1)[1]
        for source in sources:
            if source["role"].lower() == role:
                value = root_value(source)
                if value is not None:
                    return _to_python_dict(value), 0
        return {}, 0

    root_values = [root_value(source) for source in sources if root_value(source) is not None]
    return merge_dictionaries(root_values, conflict=conflict)



class _ResultShapeIndex:
    """Exact final-result membership index with O(1)-style native lookup.

    BRepTools_History may report transient images that do not survive into the
    selected final result. Membership therefore has to be checked before
    metadata is written.

    OCCT 8 BRepGraph already owns an exact shape->node index through
    ``Shapes().FindNode(shape)``. Do not materialize every result subshape and
    linearly scan it for each history image: that turns provenance into O(n^2).

    When BRepGraph is unavailable, a per-shape-type hash bucket is built lazily
    from TopExp and used as the fallback identity index.
    """

    def __init__(self, result_shape: Any):
        self.result_shape = result_shape
        self.used_brepgraph = False
        self._graph_index = None
        self._fallback_by_type = {}
        self._build()

    def _build(self):
        if _is_null_shape(self.result_shape):
            return

        try:
            from ._brepgraph import BRepGraphIndex, is_available
            if is_available():
                candidate = BRepGraphIndex(self.result_shape)
                if candidate.valid:
                    self._graph_index = candidate
                    self.used_brepgraph = True
        except Exception:
            self._graph_index = None
            self.used_brepgraph = False

    @staticmethod
    def _identity_bucket_add(buckets: dict, shape: Any) -> None:
        if _is_null_shape(shape):
            return
        key = _shape_hash(shape)
        bucket = buckets.setdefault(key, [])
        if not any(_same_shape(shape, existing) for existing in bucket):
            bucket.append(shape)

    def _ensure_fallback_type(self, shape_type: Any):
        if shape_type in self._fallback_by_type:
            return self._fallback_by_type[shape_type]

        buckets = {}
        shapes = _iter_unique_subshapes(self.result_shape, shape_type)

        try:
            if self.result_shape.ShapeType() == shape_type:
                shapes = [self.result_shape] + list(shapes or [])
        except Exception:
            pass

        for shape in shapes or []:
            self._identity_bucket_add(buckets, shape)

        self._fallback_by_type[shape_type] = buckets
        return buckets

    def resolve(self, shape: Any):
        """Return *shape* if it is an exact member of the final result.

        On OCCT 8 this is normally a direct BRepGraph ``FindNode`` query and
        does not enumerate or scan all result subshapes.
        """
        if _is_null_shape(shape):
            return None

        if self._graph_index is not None:
            try:
                node = self._graph_index.node_for_shape(shape)
                if node is not None:
                    # ``node_for_shape`` first uses Shapes().FindNode(shape).
                    # The supplied history shape is already the exact TopoDS
                    # entity we want to register in AttributeManager, so there
                    # is no need to reconstruct another shape from the node.
                    return shape
            except Exception:
                pass

        try:
            shape_type = shape.ShapeType()
        except Exception:
            return None

        buckets = self._ensure_fallback_type(shape_type)
        key = _shape_hash(shape)

        # IsSame-compatible shapes should normally share the OCCT hash. Keep
        # collision handling exact inside the bucket.
        if key in buckets:
            for candidate in buckets.get(key, []):
                if _same_shape(shape, candidate):
                    return candidate

        # Only the rare/no-hash fallback performs a broader scan.
        if key is None:
            for bucket in buckets.values():
                for candidate in bucket:
                    if _same_shape(shape, candidate):
                        return candidate

        return None


def _source_shapes_with_dictionaries(
    source_shape: Any,
    explicit_root_dictionary: Any,
    manager: AttributeManager,
) -> list[tuple[Any, Any, bool]]:
    """Return metadata-bearing ``(shape, dictionary, is_root)`` entries.

    De-duplication is hash-bucketed by shape type and OCCT identity. The old
    implementation compared every discovered subshape with every previously
    seen subshape using ``IsSame()``, which made large CellComplexes quadratic
    before history processing had even started.
    """
    result = []
    seen = {}

    def already_seen(shape: Any) -> bool:
        if _is_null_shape(shape):
            return True
        try:
            shape_type = shape.ShapeType()
        except Exception:
            shape_type = None
        key = (shape_type, _shape_hash(shape))
        bucket = seen.setdefault(key, [])
        if any(_same_shape(shape, existing) for existing in bucket):
            return True
        bucket.append(shape)
        return False

    def add(shape, dictionary, is_root=False):
        if _is_null_shape(shape) or already_seen(shape):
            return
        dictionary_py = _to_python_dict(dictionary)
        if not dictionary_py:
            return
        result.append((shape, dictionary_py, is_root))

    root_dictionary_value = explicit_root_dictionary
    if root_dictionary_value is None:
        try:
            if manager.HasDictionary(source_shape):
                root_dictionary_value = manager.GetDictionary(source_shape)
        except Exception:
            root_dictionary_value = None
    add(source_shape, root_dictionary_value, True)

    for shape_type in _SHAPE_TYPES:
        if shape_type is None:
            continue
        for subshape in _iter_unique_subshapes(source_shape, shape_type):
            try:
                if manager.HasDictionary(subshape):
                    add(subshape, manager.GetDictionary(subshape), False)
            except Exception:
                continue
    return result


def set_root_dictionary(result_shape: Any, sources: Iterable[Any], policy: str = "merge", conflict: str = "first") -> tuple[dict, int]:
    """Write the operation's root dictionary through AttributeManager."""
    dictionary, conflicts = root_dictionary(sources, policy=policy, conflict=conflict)
    if _is_null_shape(result_shape) or not dictionary:
        return dictionary, conflicts
    try:
        AttributeManager.GetInstance().SetDictionary(result_shape, dictionary)
    except Exception:
        pass
    return dictionary, conflicts


class _ContributionIndex:
    """Hash-bucketed exact target grouping for provenance contributions."""

    def __init__(self):
        self._buckets = {}
        self.groups = []

    def add(self, target_shape: Any, dictionary: Any, source_order: int, relation: str):
        try:
            shape_type = target_shape.ShapeType()
        except Exception:
            shape_type = None
        key = (shape_type, _shape_hash(target_shape))
        bucket = self._buckets.setdefault(key, [])
        for group in bucket:
            if _same_shape(group["shape"], target_shape):
                group["items"].append((source_order, relation, dictionary))
                return
        group = {
            "shape": target_shape,
            "items": [(source_order, relation, dictionary)],
        }
        bucket.append(group)
        self.groups.append(group)


def _unique_target_pairs(target_images: Iterable[tuple[Any, str]]) -> list[tuple[Any, str]]:
    """De-duplicate target images without an O(n^2) growing identity list."""
    result = []
    buckets = {}
    for target, relation in target_images or []:
        if _is_null_shape(target):
            continue
        try:
            shape_type = target.ShapeType()
        except Exception:
            shape_type = None
        key = (shape_type, _shape_hash(target))
        bucket = buckets.setdefault(key, [])
        if any(_same_shape(target, existing) for existing in bucket):
            continue
        bucket.append(target)
        result.append((target, relation))
    return result


def transfer_by_history(
    result_shape: Any,
    history: Any,
    sources: Iterable[Any],
    *,
    root_policy: str = "merge",
    conflict: str = "first",
    operation: str = "History",
) -> ProvenanceReport:
    """Transfer dictionaries using exact ``BRepTools_History`` lineage.

    Work is proportional to actual metadata.  If neither source root nor any
    source subshape carries a dictionary, no result BRepGraph is built.
    """
    report = ProvenanceReport(operation=operation)
    if _is_null_shape(result_shape):
        return report

    sources = _normalize_sources(sources)
    manager = AttributeManager.GetInstance()

    # Collect metadata-bearing source entities first.  This is intentionally
    # done before constructing the final-result BRepGraph: most TopologicPy
    # modelling operations carry no dictionaries, and provenance must be
    # essentially free in that common case.
    prepared = []
    has_metadata = False
    for source_order, source in enumerate(sources):
        entries = _source_shapes_with_dictionaries(
            source["shape"], source.get("dictionary"), manager
        )
        if entries:
            has_metadata = True
        prepared.append((source_order, source, entries))

    root_dict, root_conflicts = root_dictionary(
        sources, policy=root_policy, conflict=conflict
    )
    if root_dict:
        has_metadata = True

    if not has_metadata:
        return report

    target_index = _ResultShapeIndex(result_shape)
    report.used_brepgraph_index = target_index.used_brepgraph
    groups = _ContributionIndex()

    for source_order, source, entries in prepared:
        for source_entity, dictionary, is_root in entries:
            report.source_entities += 1
            target_images = []
            modified = []
            generated = []
            deleted = False

            history_supported = False
            if history is not None:
                try:
                    history_supported = (
                        bool(BRepTools_History.IsSupportedType(source_entity))
                        if BRepTools_History is not None else True
                    )
                except Exception:
                    history_supported = True

            if history is not None and history_supported:
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

            try:
                source_kind = source_entity.ShapeType()
            except Exception:
                source_kind = None

            # Automatic provenance remains same-kind. Cross-dimensional
            # semantic transfer belongs to explicit selector/key APIs.
            for image in modified:
                try:
                    if source_kind is not None and image.ShapeType() != source_kind:
                        continue
                except Exception:
                    continue
                target = target_index.resolve(image)
                if target is not None:
                    target_images.append((target, "modified"))
                    report.modified_images += 1

            for image in generated:
                try:
                    if source_kind is not None and image.ShapeType() != source_kind:
                        continue
                except Exception:
                    continue
                target = target_index.resolve(image)
                if target is not None:
                    target_images.append((target, "generated"))
                    report.generated_images += 1

            # History omits unchanged entities. Exact OCCT identity is their
            # authoritative continuation.
            if not target_images and not deleted:
                unchanged = target_index.resolve(source_entity)
                if unchanged is not None:
                    target_images.append((unchanged, "unchanged"))
                    report.unchanged_entities += 1

            if deleted and not target_images:
                report.deleted_entities += 1
                continue

            if target_images:
                report.mapped_entities += 1

            for target, relation in _unique_target_pairs(target_images):
                groups.add(target, dictionary, source_order, relation)

    for group in groups.groups:
        ordered = sorted(group["items"], key=lambda item: item[0])
        dictionary, conflicts = merge_dictionaries(
            (item[2] for item in ordered), conflict=conflict
        )
        report.conflicts += conflicts
        if not dictionary:
            continue
        try:
            manager.SetDictionary(group["shape"], dictionary)
            report.target_entities_written += 1
        except Exception:
            pass

    if root_dict:
        try:
            manager.SetDictionary(result_shape, root_dict)
        except Exception:
            pass
    report.conflicts += root_conflicts
    return report

def transfer_by_modifier(
    source_shape: Any,
    result_shape: Any,
    modifier: Any,
    *,
    root_dictionary: Any = None,
    operation: str = "Modifier",
) -> ProvenanceReport:
    """Transfer dictionaries through an OCCT ``ModifiedShape`` mapping.

    Crucially, the source metadata is collected *before* a BRepGraph result
    index is built.  Dictionary-free transforms therefore pay no BRepGraph
    construction cost.
    """
    report = ProvenanceReport(operation=operation)
    if _is_null_shape(source_shape) or _is_null_shape(result_shape) or modifier is None:
        return report

    manager = AttributeManager.GetInstance()
    entries = _source_shapes_with_dictionaries(
        source_shape, root_dictionary, manager
    )
    if not entries:
        return report

    target_index = _ResultShapeIndex(result_shape)
    report.used_brepgraph_index = target_index.used_brepgraph

    for source_entity, dictionary, is_root in entries:
        report.source_entities += 1
        target = None
        try:
            candidate = modifier.ModifiedShape(source_entity)
            if not _is_null_shape(candidate):
                target = target_index.resolve(candidate)
        except Exception:
            target = None

        if target is None:
            target = target_index.resolve(source_entity)
            if target is not None:
                report.unchanged_entities += 1

        if target is None:
            continue

        report.mapped_entities += 1
        if not dictionary:
            continue
        try:
            manager.SetDictionary(target, dictionary)
            report.target_entities_written += 1
        except Exception:
            pass

    # Keep the final root synchronized. Do not register a fabricated empty
    # dictionary when the source has no root metadata.
    root_dict = _to_python_dict(root_dictionary)
    if not root_dict:
        try:
            if manager.HasDictionary(source_shape):
                root_dict = _to_python_dict(manager.GetDictionary(source_shape))
        except Exception:
            root_dict = {}
    if root_dict:
        try:
            manager.SetDictionary(result_shape, root_dict)
        except Exception:
            pass
    return report

def has_any_dictionary(shape: Any) -> bool:
    """Cheaply test whether a root or any native subshape has metadata."""
    if _is_null_shape(shape):
        return False
    manager = AttributeManager.GetInstance()
    try:
        if manager.HasDictionary(shape):
            return True
    except Exception:
        pass
    for shape_type in _SHAPE_TYPES:
        if shape_type is None:
            continue
        for subshape in _iter_unique_subshapes(shape, shape_type):
            try:
                if manager.HasDictionary(subshape):
                    return True
            except Exception:
                continue
    return False
