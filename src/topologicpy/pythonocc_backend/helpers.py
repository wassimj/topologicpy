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
    result = []
    seen = set()
    for item in items:
        key = getattr(item, "_uuid", id(item))
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


def not_implemented(name: str, return_value=None):
    """Print a uniform not-implemented message and return a safe value."""
    print(f"{name} - Not implemented.")
    return return_value
