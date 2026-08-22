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


class AttributeManager:
    """
    Stores topology dictionaries against OCCT shape identity.

    This mirrors the role of topologic_core's AttributeManager: metadata
    belongs to the underlying OCCT topology rather than to a transient
    Python wrapper object.
    """

    _instance = None

    def __init__(self):
        # hash(shape) -> [(TopoDS_Shape, dictionary), ...]
        #
        # Buckets are used because hashes are not assumed to be collision-free.
        # Shape identity is confirmed with IsSame().
        self._dictionaries = {}

    @classmethod
    def GetInstance(cls):
        if cls._instance is None:
            cls._instance = cls()

        return cls._instance

    def SetDictionary(self, shape: Any, dictionary: Any):
        if _is_null_shape(shape):
            return

        key = _shape_hash(shape)

        if key is None:
            return

        bucket = self._dictionaries.setdefault(key, [])

        for i, (stored_shape, _) in enumerate(bucket):
            if _same_shape(shape, stored_shape):
                bucket[i] = (
                    stored_shape,
                    _copy_value(dictionary if dictionary is not None else {})
                )
                return

        bucket.append(
            (
                shape,
                _copy_value(dictionary if dictionary is not None else {})
            )
        )

    def GetDictionary(self, shape: Any):
        if _is_null_shape(shape):
            return {}

        key = _shape_hash(shape)

        if key is None:
            return {}

        bucket = self._dictionaries.get(key, [])

        for stored_shape, dictionary in bucket:
            if _same_shape(shape, stored_shape):
                return _copy_value(dictionary)

        return {}

    def HasDictionary(self, shape: Any) -> bool:
        if _is_null_shape(shape):
            return False

        key = _shape_hash(shape)

        if key is None:
            return False

        for stored_shape, _ in self._dictionaries.get(key, []):
            if _same_shape(shape, stored_shape):
                return True

        return False

    def ClearOne(self, shape: Any):
        if _is_null_shape(shape):
            return

        key = _shape_hash(shape)

        if key is None:
            return

        bucket = self._dictionaries.get(key, [])

        bucket = [
            (stored_shape, dictionary)
            for stored_shape, dictionary in bucket
            if not _same_shape(shape, stored_shape)
        ]

        if bucket:
            self._dictionaries[key] = bucket
        else:
            self._dictionaries.pop(key, None)

    def ClearAll(self):
        self._dictionaries.clear()

    def CopyDictionary(
        self,
        source_shape: Any,
        target_shape: Any
    ):
        if self.HasDictionary(source_shape):
            self.SetDictionary(
                target_shape,
                self.GetDictionary(source_shape)
            )