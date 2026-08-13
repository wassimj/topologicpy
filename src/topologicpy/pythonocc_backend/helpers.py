from __future__ import annotations

import math
import uuid
from typing import Iterable, Any


def new_uuid() -> str:
    return str(uuid.uuid4())


def is_vertex(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("x", "y", "z")) and obj.__class__.__name__ == "Vertex"


def distance3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def same_vertex(a: Any, b: Any, tolerance: float = 0.0001) -> bool:
    return is_vertex(a) and is_vertex(b) and distance3(a, b) <= tolerance


def unique_by_uuid(items: Iterable[Any]) -> list:
    """
    Two wrapper objects extracted from the same underlying OCCT (sub-)shape
    (e.g. the shared endpoint of two adjacent edges) get distinct Python
    identity and distinct `_uuid`s, but should still be treated as the same
    topological entity. Prefer OCCT shape identity (HashCode) over `_uuid`.
    """
    result = []
    seen = set()
    for item in items:
        shape = getattr(item, "shape", None)
        key = None
        if shape is not None and hasattr(shape, "IsNull"):
            try:
                if not shape.IsNull():
                    key = ("shape", hash(shape))
            except Exception:
                key = None
        if key is None:
            key = ("uuid", getattr(item, "_uuid", id(item)))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def round_coord(value: float, tolerance: float = 0.0001) -> float:
    if tolerance <= 0:
        return value
    return round(value / tolerance) * tolerance


def vertex_key(v, tolerance: float = 0.0001):
    return (round_coord(v.x, tolerance), round_coord(v.y, tolerance), round_coord(v.z, tolerance))


def edge_key(edge, tolerance: float = 0.0001):
    a = vertex_key(edge.start, tolerance)
    b = vertex_key(edge.end, tolerance)
    return tuple(sorted([a, b]))


def dedupe_vertices_by_distance(vertices: Iterable[Any], tolerance: float = 0.0001) -> list:
    """
    Merge vertices within tolerance using a floor-based spatial grid sized to
    the tolerance (round-bucketing can straddle bucket boundaries), confirming
    each merge with a real distance3 check.
    """
    if tolerance is None or tolerance <= 0:
        tolerance = 0.0001
    kept: list = []
    grid: dict = {}

    for v in vertices:
        if not is_vertex(v):
            continue
        cx = math.floor(v.x / tolerance)
        cy = math.floor(v.y / tolerance)
        cz = math.floor(v.z / tolerance)
        duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for idx in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if same_vertex(v, kept[idx], tolerance):
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            grid.setdefault((cx, cy, cz), []).append(len(kept))
            kept.append(v)
    return kept


def not_implemented(name: str, return_value=None):
    """Print a uniform not-implemented message and return a safe value."""
    print(f"{name} - Not implemented.")
    return return_value
