from __future__ import annotations

import copy
from typing import Any


def _is_null_shape(shape: Any) -> bool:
    if shape is None:
        return True

    if hasattr(shape, "IsNull"):
        try:
            return bool(shape.IsNull())
        except Exception:
            pass

    return False


def _same_shape(shape_a: Any, shape_b: Any) -> bool:
    if shape_a is shape_b:
        return True

    if _is_null_shape(shape_a) or _is_null_shape(shape_b):
        return False

    try:
        return bool(shape_a.IsSame(shape_b))
    except Exception:
        pass

    try:
        return shape_a == shape_b
    except Exception:
        return False


def _shape_hash(shape: Any):
    if _is_null_shape(shape):
        return None

    try:
        return hash(shape)
    except Exception:
        return id(shape)


def _copy_value(value: Any):
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _shape_from_topology(topology: Any):
    if topology is None:
        return None

    if isinstance(topology, dict):
        return topology.get("shape", None)

    if hasattr(topology, "shape"):
        try:
            return getattr(topology, "shape")
        except Exception:
            pass

    if hasattr(topology, "GetOcctShape"):
        try:
            return topology.GetOcctShape()
        except Exception:
            pass

    return None


def _same_relation_item(item_a: Any, item_b: Any) -> bool:
    if item_a is item_b:
        return True

    shape_a = _shape_from_topology(item_a)
    shape_b = _shape_from_topology(item_b)

    if not _is_null_shape(shape_a) and not _is_null_shape(shape_b):
        if _same_shape(shape_a, shape_b):
            return True

    uuid_a = getattr(item_a, "_uuid", None)
    uuid_b = getattr(item_b, "_uuid", None)

    if uuid_a is not None and uuid_b is not None:
        return uuid_a == uuid_b

    return False


def _deduplicate_relationships(items: Any) -> list:
    if items is None:
        return []

    if not isinstance(items, (list, tuple)):
        items = [items]

    result = []
    for item in items:
        if item is None:
            continue

        duplicate = False
        for existing in result:
            if _same_relation_item(item, existing):
                duplicate = True
                break

        if not duplicate:
            result.append(item)

    return result


class AttributeManager:
    """
    Stores topology metadata against OCCT shape identity.

    Dictionaries, contents, contexts, and apertures belong to the underlying
    OCCT topology rather than to a transient Python wrapper. This allows a
    freshly reconstructed wrapper for the same TopoDS_Shape to recover the
    relationships previously attached to that topology.
    """

    _instance = None

    def __init__(self):
        # hash(shape) -> [(TopoDS_Shape, value), ...]
        #
        # Buckets are used because hashes are not assumed to be collision-free.
        # Shape identity is confirmed with IsSame().
        self._dictionaries = {}
        self._contents = {}
        self._contexts = {}
        self._apertures = {}

    @classmethod
    def GetInstance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    # ------------------------------------------------------------------
    # Generic shape-keyed storage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_value(store: dict, shape: Any, value: Any, deep_copy: bool = False):
        if _is_null_shape(shape):
            return

        key = _shape_hash(shape)
        if key is None:
            return

        if deep_copy:
            stored_value = _copy_value(value)
        else:
            stored_value = list(value) if isinstance(value, (list, tuple)) else value

        bucket = store.setdefault(key, [])

        for i, (stored_shape, _) in enumerate(bucket):
            if _same_shape(shape, stored_shape):
                bucket[i] = (stored_shape, stored_value)
                return

        bucket.append((shape, stored_value))

    @staticmethod
    def _get_value(store: dict, shape: Any, default: Any, deep_copy: bool = False):
        if _is_null_shape(shape):
            return _copy_value(default) if deep_copy else list(default) if isinstance(default, list) else default

        key = _shape_hash(shape)
        if key is None:
            return _copy_value(default) if deep_copy else list(default) if isinstance(default, list) else default

        bucket = store.get(key, [])

        for stored_shape, value in bucket:
            if _same_shape(shape, stored_shape):
                if deep_copy:
                    return _copy_value(value)
                if isinstance(value, list):
                    return list(value)
                return value

        return _copy_value(default) if deep_copy else list(default) if isinstance(default, list) else default

    @staticmethod
    def _has_value(store: dict, shape: Any) -> bool:
        if _is_null_shape(shape):
            return False

        key = _shape_hash(shape)
        if key is None:
            return False

        for stored_shape, _ in store.get(key, []):
            if _same_shape(shape, stored_shape):
                return True

        return False

    @staticmethod
    def _clear_value(store: dict, shape: Any):
        if _is_null_shape(shape):
            return

        key = _shape_hash(shape)
        if key is None:
            return

        bucket = store.get(key, [])
        bucket = [
            (stored_shape, value)
            for stored_shape, value in bucket
            if not _same_shape(shape, stored_shape)
        ]

        if bucket:
            store[key] = bucket
        else:
            store.pop(key, None)

    # ------------------------------------------------------------------
    # Dictionaries
    # ------------------------------------------------------------------

    def SetDictionary(self, shape: Any, dictionary: Any):
        self._set_value(
            self._dictionaries,
            shape,
            dictionary if dictionary is not None else {},
            deep_copy=True,
        )

    def GetDictionary(self, shape: Any):
        return self._get_value(
            self._dictionaries,
            shape,
            {},
            deep_copy=True,
        )

    def HasDictionary(self, shape: Any) -> bool:
        return self._has_value(self._dictionaries, shape)

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    def SetContents(self, shape: Any, contents: Any):
        self._set_value(
            self._contents,
            shape,
            _deduplicate_relationships(contents),
        )

    def GetContents(self, shape: Any) -> list:
        return self._get_value(self._contents, shape, [])

    def HasContents(self, shape: Any) -> bool:
        return self._has_value(self._contents, shape)

    # ------------------------------------------------------------------
    # Contexts
    # ------------------------------------------------------------------

    def SetContexts(self, shape: Any, contexts: Any):
        self._set_value(
            self._contexts,
            shape,
            _deduplicate_relationships(contexts),
        )

    def GetContexts(self, shape: Any) -> list:
        return self._get_value(self._contexts, shape, [])

    def HasContexts(self, shape: Any) -> bool:
        return self._has_value(self._contexts, shape)

    # ------------------------------------------------------------------
    # Apertures
    # ------------------------------------------------------------------

    def SetApertures(self, shape: Any, apertures: Any):
        self._set_value(
            self._apertures,
            shape,
            _deduplicate_relationships(apertures),
        )

    def GetApertures(self, shape: Any) -> list:
        return self._get_value(self._apertures, shape, [])

    def HasApertures(self, shape: Any) -> bool:
        return self._has_value(self._apertures, shape)

    # ------------------------------------------------------------------
    # Clearing / copying
    # ------------------------------------------------------------------

    def ClearOne(self, shape: Any):
        self._clear_value(self._dictionaries, shape)
        self._clear_value(self._contents, shape)
        self._clear_value(self._contexts, shape)
        self._clear_value(self._apertures, shape)

    def ClearAll(self):
        self._dictionaries.clear()
        self._contents.clear()
        self._contexts.clear()
        self._apertures.clear()

    def CopyDictionary(self, source_shape: Any, target_shape: Any):
        if self.HasDictionary(source_shape):
            self.SetDictionary(
                target_shape,
                self.GetDictionary(source_shape),
            )

    def CopyRelationships(self, source_shape: Any, target_shape: Any):
        if self.HasContents(source_shape):
            self.SetContents(
                target_shape,
                self.GetContents(source_shape),
            )

        if self.HasContexts(source_shape):
            self.SetContexts(
                target_shape,
                self.GetContexts(source_shape),
            )

        if self.HasApertures(source_shape):
            self.SetApertures(
                target_shape,
                self.GetApertures(source_shape),
            )

    def CopyAttributes(self, source_shape: Any, target_shape: Any):
        self.CopyDictionary(source_shape, target_shape)
        self.CopyRelationships(source_shape, target_shape)
