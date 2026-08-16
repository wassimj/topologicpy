# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import heapq
import logging
import math
from collections import namedtuple
from itertools import chain, cycle, islice, tee
from typing import Iterable, List, Optional, Sequence, Tuple, Union

log = logging.getLogger(__name__)
EPSILON = 1e-5

Subtree = namedtuple("Subtree", "source, height, sinks")


# -----------------------------------------------------------------------------
# Optional TopologicPy bridge
# -----------------------------------------------------------------------------


def _vertex_class():
    try:
        from topologicpy.Vertex import Vertex
        return Vertex
    except Exception:
        return None


def _edge_class():
    try:
        from topologicpy.Edge import Edge
        return Edge
    except Exception:
        return None


def _is_coordinate_pair(value) -> bool:
    """Return True for plain coordinate-like pairs that should remain Point2 values."""
    if isinstance(value, Point2):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            float(value[0])
            float(value[1])
            return True
        except Exception:
            return False
    return False


def _is_topologic_vertex(value) -> bool:
    """Return True only for real TopologicPy/Core vertices, never for coordinate pairs."""
    if _is_coordinate_pair(value):
        return False
    Vertex = _vertex_class()
    if Vertex is None:
        return False
    try:
        x = Vertex.X(value)
        y = Vertex.Y(value)
        if x is None or y is None:
            return False
        float(x)
        float(y)
        return True
    except Exception:
        return False


def _to_vertex(point):
    """Converts a Point2 to a TopologicPy Vertex when available."""
    if not isinstance(point, Point2) and _is_topologic_vertex(point):
        return point
    Vertex = _vertex_class()
    if Vertex is None:
        return point
    try:
        p = _point_from_any(point)
        return Vertex.ByCoordinates(float(p.x), float(p.y), 0.0)
    except Exception:
        return point


def _point_from_any(value):
    """Returns a Point2 from a tuple/list, Point2, or TopologicPy Vertex."""
    if isinstance(value, Point2):
        return value
    # Coordinate pairs must be handled before TopologicPy accessors. In a real
    # TopologicPy environment, Vertex.X/Y may print diagnostics and return None
    # for tuples/lists rather than raising, which can otherwise misclassify
    # coordinate pairs as Topologic vertices.
    if _is_coordinate_pair(value):
        return Point2(float(value[0]), float(value[1]))
    Vertex = _vertex_class()
    if Vertex is not None:
        try:
            x = Vertex.X(value)
            y = Vertex.Y(value)
            if x is not None and y is not None:
                return Point2(float(x), float(y))
        except Exception:
            pass
    try:
        return Point2(float(value[0]), float(value[1]))
    except Exception as exc:
        raise ValueError("Contour vertices must be Point2, TopologicPy vertices, or coordinate-like pairs.") from exc


# -----------------------------------------------------------------------------
# 2-D geometry primitives
# -----------------------------------------------------------------------------


class Point2:
    """A lightweight 2-D point/vector used by the straight-skeleton solver."""

    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return f"Point2({self.x:.6g}, {self.y:.6g})"

    def __str__(self):
        return f"({self.x:.6g}, {self.y:.6g})"

    def __eq__(self, other):
        if not isinstance(other, Point2):
            return False
        return abs(self.x - other.x) <= EPSILON and abs(self.y - other.y) <= EPSILON

    def __hash__(self):
        return hash((round(self.x / EPSILON), round(self.y / EPSILON)))

    def __add__(self, other):
        return Point2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point2(self.x * float(scalar), self.y * float(scalar))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        scalar = float(scalar)
        if abs(scalar) <= EPSILON:
            return Point2(self.x, self.y)
        return Point2(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return Point2(-self.x, -self.y)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def distance(self, other):
        return abs(self - other)

    def normalized(self):
        length = abs(self)
        if length <= EPSILON:
            return Point2(0.0, 0.0)
        return self / length


class Ray2:
    """A half-line represented by origin p and unit direction v."""

    __slots__ = ("p", "v")

    def __init__(self, p, v):
        self.p = _point_from_any(p)
        self.v = _point_from_any(v).normalized()

    def intersect(self, other):
        denom = self.v.cross(other.v)
        if abs(denom) <= EPSILON:
            return None
        t = (other.p - self.p).cross(other.v) / denom
        if t < -EPSILON:
            return None
        return self.p + self.v * max(0.0, t)

    def __repr__(self):
        return f"Ray2({self.p!r}, {self.v!r})"


class Line2:
    """An infinite 2-D line represented by point p and unit direction v."""

    __slots__ = ("p", "v")

    def __init__(self, p1, p2=None):
        if p2 is None and isinstance(p1, Ray2):
            self.p = p1.p
            self.v = p1.v.normalized()
        elif p2 is None and isinstance(p1, Line2):
            self.p = p1.p
            self.v = p1.v.normalized()
        else:
            self.p = _point_from_any(p1)
            self.v = (_point_from_any(p2) - self.p).normalized()

    def intersect(self, other):
        denom = self.v.cross(other.v)
        if abs(denom) <= EPSILON:
            return None
        t = (other.p - self.p).cross(other.v) / denom
        return self.p + self.v * t

    def distance(self, point):
        p = _point_from_any(point)
        return abs((p - self.p).cross(self.v))

    def __repr__(self):
        return f"Line2({self.p!r}, {self.v!r})"


class LineSegment2(Line2):
    """A finite 2-D line segment."""

    __slots__ = ("p2",)

    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.p2 = _point_from_any(p2)

    def intersect(self, other):
        inter = super().intersect(other)
        if inter is None:
            return None
        if self.on_segment(inter) and (not isinstance(other, LineSegment2) or other.on_segment(inter)):
            return inter
        return None

    def on_segment(self, point):
        p = _point_from_any(point)
        return (
            min(self.p.x, self.p2.x) - EPSILON <= p.x <= max(self.p.x, self.p2.x) + EPSILON
            and min(self.p.y, self.p2.y) - EPSILON <= p.y <= max(self.p.y, self.p2.y) + EPSILON
            and self.distance(p) <= EPSILON
        )

    def __repr__(self):
        return f"LineSegment2({self.p!r}, {self.p2!r})"


# -----------------------------------------------------------------------------
# Compatibility helpers exposed by the old module
# -----------------------------------------------------------------------------


def distance(v1, v2):
    """Returns the XY distance between two coordinate-like or TopologicPy vertices."""
    p1 = _point_from_any(v1)
    p2 = _point_from_any(v2)
    return p1.distance(p2)


def normalize(v):
    """Returns a normalized vector. Coordinate pairs always return Point2."""
    if isinstance(v, Point2) or _is_coordinate_pair(v):
        return _point_from_any(v).normalized()
    p = _point_from_any(v).normalized()
    return _to_vertex(p) if _is_topologic_vertex(v) else p


def bisector(v1, v2, v3):
    """Returns the internal angle bisector direction at v2 for v1-v2-v3."""
    p1, p2, p3 = _point_from_any(v1), _point_from_any(v2), _point_from_any(v3)
    e1 = (p1 - p2).normalized()
    e2 = (p3 - p2).normalized()
    b = (e1 + e2).normalized()
    # Coordinate-pair inputs are a pure-Python API and should not be promoted to
    # TopologicPy vertices merely because TopologicPy is installed.
    if all((isinstance(v, Point2) or _is_coordinate_pair(v)) for v in (v1, v2, v3)):
        return b
    return _to_vertex(b) if any(_is_topologic_vertex(v) for v in (v1, v2, v3)) else b


def _window(lst):
    if not lst:
        return iter(())
    prevs, items, nexts = tee(lst, 3)
    prevs = islice(cycle(prevs), len(lst) - 1, None)
    nexts = islice(cycle(nexts), 1, None)
    return zip(prevs, items, nexts)


def _approximately_equals(a, b):
    return abs(float(a) - float(b)) <= max(1.0, abs(float(a)), abs(float(b))) * EPSILON


def _approximately_same(a, b):
    a = _point_from_any(a)
    b = _point_from_any(b)
    return _approximately_equals(a.x, b.x) and _approximately_equals(a.y, b.y)


def _signed_area(points: Sequence[Point2]) -> float:
    area = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        area += a.x * b.y - b.x * a.y
    return 0.5 * area


def _normalize_contour(contour):
    """
    Normalizes a polygon contour by converting vertices to Point2 objects,
    removing duplicate/collinear vertices, and preserving the input orientation.
    """
    if contour is None:
        return []
    pts = [_point_from_any(p) for p in contour]
    if len(pts) > 1 and _approximately_same(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    cleaned = []
    for prev, point, nxt in _window(pts):
        if _approximately_same(point, nxt) or _approximately_same(prev, point):
            continue
        v1 = (point - prev).normalized()
        v2 = (nxt - point).normalized()
        if abs(v1.cross(v2)) <= EPSILON and v1.dot(v2) > 0:
            continue
        cleaned.append(point)
    return cleaned


# -----------------------------------------------------------------------------
# Straight skeleton event model
# -----------------------------------------------------------------------------


class _SplitEvent(namedtuple("_SplitEvent", "distance intersection_point vertex opposite_edge")):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance


class _EdgeEvent(namedtuple("_EdgeEvent", "distance intersection_point vertex_a vertex_b")):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance


_OriginalEdge = namedtuple("_OriginalEdge", "edge bisector_left bisector_right")


class _EventQueue:
    def __init__(self):
        self.__data = []

    def put(self, item):
        if item is not None:
            heapq.heappush(self.__data, item)

    def put_all(self, iterable):
        for item in iterable or []:
            self.put(item)

    def get(self):
        return heapq.heappop(self.__data)

    def empty(self):
        return len(self.__data) == 0

    def peek(self):
        return self.__data[0] if self.__data else None


class _LAVertex:
    def __init__(self, point, edge_left, edge_right, direction_vectors=None):
        self.point = _point_from_any(point)
        self.edge_left = edge_left
        self.edge_right = edge_right
        self.prev = None
        self.next = None
        self.lav = None
        self._valid = True

        creator_vectors = ((edge_left.v * -1.0).normalized(), edge_right.v.normalized())
        if direction_vectors is None:
            direction_vectors = creator_vectors
        left, right = direction_vectors
        self._is_reflex = left.cross(right) < -EPSILON

        bisector_vector = (creator_vectors[0] + creator_vectors[1])
        if abs(bisector_vector) <= EPSILON:
            # 180-degree case: use a perpendicular to the incoming edge.
            e = edge_left.v.normalized()
            bisector_vector = Point2(-e.y, e.x)
        if self._is_reflex:
            bisector_vector = -bisector_vector
        self._bisector = Ray2(self.point, bisector_vector)

    @property
    def bisector(self):
        return self._bisector

    @property
    def is_reflex(self):
        return self._is_reflex

    @property
    def is_valid(self):
        return self._valid

    @property
    def original_edges(self):
        return self.lav._slav._original_edges if self.lav is not None else []

    def invalidate(self):
        if self.lav is not None:
            self.lav.invalidate(self)
        else:
            self._valid = False

    def next_event(self):
        events = []

        if self.prev is not None:
            i_prev = self.bisector.intersect(self.prev.bisector)
            if i_prev is not None:
                dist = Line2(self.edge_left).distance(i_prev)
                if dist >= -EPSILON:
                    events.append(_EdgeEvent(max(0.0, dist), i_prev, self.prev, self))

        if self.next is not None:
            i_next = self.bisector.intersect(self.next.bisector)
            if i_next is not None:
                dist = Line2(self.edge_right).distance(i_next)
                if dist >= -EPSILON:
                    events.append(_EdgeEvent(max(0.0, dist), i_next, self, self.next))

        if self.is_reflex:
            events.extend(self._split_events())

        if not events:
            return None
        return min(events, key=lambda event: self.point.distance(event.intersection_point))

    def _split_events(self):
        events = []
        for original in self.original_edges:
            edge = original.edge
            if edge is self.edge_left or edge is self.edge_right:
                continue
            try:
                leftdot = abs(self.edge_left.v.normalized().dot(edge.v.normalized()))
                rightdot = abs(self.edge_right.v.normalized().dot(edge.v.normalized()))
                self_edge = self.edge_left if leftdot < rightdot else self.edge_right
                other_edge = self.edge_left if leftdot > rightdot else self.edge_right
                intersection = Line2(self_edge).intersect(Line2(edge))
                if intersection is None or _approximately_same(intersection, self.point):
                    continue
                linvec = (self.point - intersection).normalized()
                edvec = edge.v.normalized()
                if linvec.dot(edvec) < 0:
                    edvec = -edvec
                bisecvec = edvec + linvec
                if abs(bisecvec) <= EPSILON:
                    continue
                candidate = Line2(intersection, intersection + bisecvec).intersect(self.bisector)
                if candidate is None:
                    continue
                xleft = original.bisector_left.v.normalized().cross((candidate - original.bisector_left.p).normalized()) >= -EPSILON
                xright = original.bisector_right.v.normalized().cross((candidate - original.bisector_right.p).normalized()) <= EPSILON
                xedge = edge.v.normalized().cross((candidate - edge.p).normalized()) <= EPSILON
                if xleft and xright and xedge:
                    events.append(_SplitEvent(Line2(edge).distance(candidate), candidate, self, edge))
            except Exception:
                continue
        return events

    def __repr__(self):
        kind = "reflex" if self.is_reflex else "convex"
        return f"_LAVertex({kind}, {self.point!r})"


class _LAV:
    def __init__(self, slav):
        self.head = None
        self._slav = slav
        self._len = 0

    @classmethod
    def from_polygon(cls, polygon, slav):
        lav = cls(slav)
        for prev, point, nxt in _window(polygon):
            vertex = _LAVertex(point, LineSegment2(prev, point), LineSegment2(point, nxt))
            vertex.lav = lav
            lav._len += 1
            if lav.head is None:
                lav.head = vertex
                vertex.prev = vertex.next = vertex
            else:
                vertex.next = lav.head
                vertex.prev = lav.head.prev
                vertex.prev.next = vertex
                lav.head.prev = vertex
        return lav

    @classmethod
    def from_chain(cls, head, slav):
        lav = cls(slav)
        lav.head = head
        seen = set()
        cur = head
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            cur.lav = lav
            lav._len += 1
            cur = cur.next
            if cur is head:
                break
        return lav

    def __iter__(self):
        cur = self.head
        if cur is None:
            return
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = cur.next
            if cur is self.head:
                return

    def __len__(self):
        return self._len

    def invalidate(self, vertex):
        if vertex.lav is not self:
            vertex._valid = False
            return
        vertex._valid = False
        if self.head is vertex:
            self.head = vertex.next if vertex.next is not vertex else None
        vertex.lav = None

    def unify(self, vertex_a, vertex_b, point):
        replacement = _LAVertex(
            point,
            vertex_a.edge_left,
            vertex_b.edge_right,
            (vertex_b.bisector.v.normalized(), vertex_a.bisector.v.normalized()),
        )
        replacement.lav = self
        replacement.prev = vertex_a.prev
        replacement.next = vertex_b.next
        if replacement.prev is not None:
            replacement.prev.next = replacement
        if replacement.next is not None:
            replacement.next.prev = replacement
        if self.head in (vertex_a, vertex_b):
            self.head = replacement
        vertex_a.invalidate()
        vertex_b.invalidate()
        self._len = max(0, self._len - 1)
        return replacement


class _SLAV:
    def __init__(self, polygon, holes=None):
        holes = holes or []
        contours = [_normalize_contour(polygon)]
        contours.extend(_normalize_contour(hole) for hole in holes)
        self._lavs = [_LAV.from_polygon(contour, self) for contour in contours if len(contour) >= 3]
        self._original_edges = [
            _OriginalEdge(LineSegment2(vertex.prev.point, vertex.point), vertex.prev.bisector, vertex.bisector)
            for vertex in chain.from_iterable(self._lavs)
        ]

    def __iter__(self):
        for lav in self._lavs:
            yield lav

    def __len__(self):
        return len(self._lavs)

    def empty(self):
        return len(self._lavs) == 0

    def handle_edge_event(self, event):
        if event.vertex_a.lav is None or event.vertex_b.lav is None:
            return None, []
        lav = event.vertex_a.lav
        sinks = []
        events = []

        if event.vertex_a.prev == event.vertex_b.next:
            if lav in self._lavs:
                self._lavs.remove(lav)
            for vertex in list(lav):
                sinks.append(vertex.point)
                vertex.invalidate()
        else:
            new_vertex = lav.unify(event.vertex_a, event.vertex_b, event.intersection_point)
            sinks.extend((event.vertex_a.point, event.vertex_b.point))
            next_event = new_vertex.next_event()
            if next_event is not None:
                events.append(next_event)

        return Subtree(event.intersection_point, event.distance, sinks), events

    def handle_split_event(self, event):
        if event.vertex.lav is None or not event.vertex.is_valid:
            return None, []
        lav = event.vertex.lav
        sinks = [event.vertex.point]
        x = None
        y = None
        norm = event.opposite_edge.v.normalized()

        for vertex in chain.from_iterable(self._lavs):
            if norm == vertex.edge_left.v.normalized() and event.opposite_edge.p == vertex.edge_left.p:
                x = vertex
                y = x.prev
            elif norm == vertex.edge_right.v.normalized() and event.opposite_edge.p == vertex.edge_right.p:
                y = vertex
                x = y.next
            if x is not None and y is not None:
                xleft = y.bisector.v.normalized().cross((event.intersection_point - y.point).normalized()) >= -EPSILON
                xright = x.bisector.v.normalized().cross((event.intersection_point - x.point).normalized()) <= EPSILON
                if xleft and xright:
                    break
                x = None
                y = None

        if x is None or y is None:
            return None, []

        v1 = _LAVertex(event.intersection_point, event.vertex.edge_left, event.opposite_edge)
        v2 = _LAVertex(event.intersection_point, event.opposite_edge, event.vertex.edge_right)

        v1.prev = event.vertex.prev
        v1.next = x
        event.vertex.prev.next = v1
        x.prev = v1

        v2.prev = y
        v2.next = event.vertex.next
        event.vertex.next.prev = v2
        y.next = v2

        if lav in self._lavs:
            self._lavs.remove(lav)
        if x.lav in self._lavs and x.lav is not lav:
            self._lavs.remove(x.lav)

        new_lavs = [_LAV.from_chain(v1, self)] if lav is not x.lav else [_LAV.from_chain(v1, self), _LAV.from_chain(v2, self)]
        vertices = []
        for new_lav in new_lavs:
            if len(new_lav) > 2:
                self._lavs.append(new_lav)
                vertices.append(new_lav.head)
            else:
                for vertex in list(new_lav):
                    sinks.append(vertex.point)
                    vertex.invalidate()

        event.vertex.invalidate()
        events = []
        for vertex in vertices:
            next_event = vertex.next_event()
            if next_event is not None:
                events.append(next_event)
        return Subtree(event.intersection_point, event.distance, sinks), events


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def _merge_sources(skeleton: List[Subtree]) -> List[Subtree]:
    merged = []
    for subtree in skeleton:
        match_index = None
        for i, existing in enumerate(merged):
            if _approximately_same(existing.source, subtree.source):
                match_index = i
                break
        if match_index is None:
            merged.append(Subtree(subtree.source, subtree.height, list(subtree.sinks)))
        else:
            existing = merged[match_index]
            sinks = list(existing.sinks)
            for sink in subtree.sinks:
                if not any(_approximately_same(sink, other) for other in sinks):
                    sinks.append(sink)
            merged[match_index] = Subtree(existing.source, min(existing.height, subtree.height), sinks)
    return merged


def _fallback_convex_skeleton(contour: Sequence[Point2]) -> List[Subtree]:
    """
    Deterministic fallback for convex or numerically difficult polygons.

    It connects each edge midpoint to the polygon centroid. This is not a full
    straight-skeleton solution, but it is preferable to returning an exception
    for simple convex inputs if event propagation degenerates numerically.
    """
    if len(contour) < 3:
        return []
    cx = sum(p.x for p in contour) / float(len(contour))
    cy = sum(p.y for p in contour) / float(len(contour))
    centre = Point2(cx, cy)
    out = []
    for a, b in zip(contour, contour[1:] + contour[:1]):
        mid = Point2((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)
        out.append(Subtree(centre, Line2(a, b).distance(centre), [mid]))
    return out


def _point_in_polygon(point: Point2, contour: Sequence[Point2], include_boundary: bool = True) -> bool:
    """Returns True if point is inside a polygon contour by ray casting."""
    if len(contour) < 3:
        return False
    p = _point_from_any(point)
    inside = False
    j = len(contour) - 1
    for i in range(len(contour)):
        a = contour[i]
        b = contour[j]
        if include_boundary and LineSegment2(a, b).on_segment(p):
            return True
        intersects = ((a.y > p.y) != (b.y > p.y)) and (p.x < (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) if abs(b.y - a.y) > EPSILON else EPSILON) + a.x)
        if intersects:
            inside = not inside
        j = i
    return inside


def _valid_skeleton(subtrees: Sequence[Subtree], outer: Sequence[Point2], holes: Sequence[Sequence[Point2]]) -> bool:
    """Returns True if skeleton sources are finite and lie inside the domain."""
    if not subtrees:
        return False
    for subtree in subtrees:
        try:
            source = _point_from_any(subtree.source)
            if not (math.isfinite(source.x) and math.isfinite(source.y)):
                return False
            if not _point_in_polygon(source, outer, include_boundary=True):
                return False
            for hole in holes or []:
                if _point_in_polygon(source, hole, include_boundary=False):
                    return False
        except Exception:
            return False
    return True


def _convert_subtrees(subtrees: Iterable[Subtree], asTopologic: bool = True) -> List[Subtree]:
    out = []
    for subtree in subtrees or []:
        if asTopologic:
            source = _to_vertex(subtree.source)
            sinks = [_to_vertex(sink) for sink in subtree.sinks]
        else:
            source = subtree.source
            sinks = list(subtree.sinks)
        out.append(Subtree(source, float(subtree.height), sinks))
    return out


def skeletonize(polygon, holes=None, asTopologic: bool = True, fallback: bool = True, silent: bool = False):
    """
    Computes a straight skeleton approximation for a simple polygon.

    Parameters
    ----------
    polygon : list
        The outer polygon contour as coordinate pairs, Point2 objects, or
        TopologicPy vertices. Repeated first/last vertices are accepted.
    holes : list , optional
        Optional inner contours. Hole support follows the original polyskel
        event model and is best-effort for non-degenerate simple holes.
    asTopologic : bool , optional
        If True and topologicpy.Vertex can be imported, returned Subtree sources
        and sinks are TopologicPy vertices. If False, Point2 objects are returned.
        Default is True.
    fallback : bool , optional
        If True, a centroid-to-edge-midpoint fallback is returned for simple
        polygons when the event algorithm degenerates. Default is True.
    silent : bool , optional
        If True, warning messages are suppressed. Default is False.

    Returns
    -------
    list
        A list of Subtree(source, height, sinks) records.
    """
    try:
        outer = _normalize_contour(polygon)
        inner = [_normalize_contour(hole) for hole in (holes or [])]
    except Exception as exc:
        if not silent:
            print(f"Polyskel.skeletonize - Error: {exc}. Returning [].")
        return []

    if len(outer) < 3:
        if not silent:
            print("Polyskel.skeletonize - Error: The polygon must contain at least three distinct vertices. Returning [].")
        return []

    try:
        slav = _SLAV(outer, inner)
        queue = _EventQueue()
        output = []
        for lav in slav:
            for vertex in lav:
                queue.put(vertex.next_event())
        guard = 0
        max_events = max(32, 64 * (len(outer) + sum(len(h) for h in inner)))
        while not (queue.empty() or slav.empty()) and guard < max_events:
            guard += 1
            event = queue.get()
            if isinstance(event, _EdgeEvent):
                if not event.vertex_a.is_valid or not event.vertex_b.is_valid:
                    continue
                arc, events = slav.handle_edge_event(event)
            elif isinstance(event, _SplitEvent):
                if not event.vertex.is_valid:
                    continue
                arc, events = slav.handle_split_event(event)
            else:
                continue
            queue.put_all(events)
            if arc is not None:
                output.append(arc)
        output = _merge_sources(output)
        if _valid_skeleton(output, outer, inner):
            return _convert_subtrees(output, asTopologic=asTopologic)
    except Exception as exc:
        if not silent:
            print(f"Polyskel.skeletonize - Warning: Event solver failed: {exc}.")

    if fallback:
        return _convert_subtrees(_fallback_convex_skeleton(outer), asTopologic=asTopologic)
    return []


def Skeletonize(polygon, holes=None, asTopologic: bool = True, fallback: bool = True, silent: bool = False):
    """CamelCase alias for skeletonize."""
    return skeletonize(polygon, holes=holes, asTopologic=asTopologic, fallback=fallback, silent=silent)


__all__ = [
    "EPSILON",
    "Point2",
    "Ray2",
    "Line2",
    "LineSegment2",
    "Subtree",
    "bisector",
    "distance",
    "normalize",
    "skeletonize",
    "Skeletonize",
]
