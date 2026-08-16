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

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple
import math

# TopologicPy imports are intentionally defensive so that this module remains
# importable for static analysis even when topologic_core is unavailable.
try:
    from topologicpy.Vertex import Vertex
    from topologicpy.Edge import Edge
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
except Exception:
    Vertex = Edge = Face = Topology = object  # type: ignore


# ----------------------------
# Axis-Aligned Bounding Box
# ----------------------------
@dataclass
class AABB:
    """Axis-aligned bounding box: [minx, miny, minz] .. [maxx, maxy, maxz]."""

    minx: float
    miny: float
    minz: float
    maxx: float
    maxy: float
    maxz: float

    @staticmethod
    def from_points(pts: Iterable[Tuple[float, float, float]], pad: float = 0.0) -> "AABB":
        """
        Creates an AABB from an iterable of 3D points.

        Empty input returns a degenerate AABB at the origin.
        """
        it = iter(pts)
        try:
            x, y, z = next(it)
        except StopIteration:
            return AABB(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        minx = maxx = float(x)
        miny = maxy = float(y)
        minz = maxz = float(z)

        for x, y, z in it:
            x = float(x)
            y = float(y)
            z = float(z)
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            if z < minz:
                minz = z
            if z > maxz:
                maxz = z

        if pad:
            pad = float(pad)
            minx -= pad
            miny -= pad
            minz -= pad
            maxx += pad
            maxy += pad
            maxz += pad

        return AABB(minx, miny, minz, maxx, maxy, maxz)

    @staticmethod
    def union(a: "AABB", b: "AABB") -> "AABB":
        """Returns the smallest AABB containing both input AABBs."""
        return AABB(
            min(a.minx, b.minx),
            min(a.miny, b.miny),
            min(a.minz, b.minz),
            max(a.maxx, b.maxx),
            max(a.maxy, b.maxy),
            max(a.maxz, b.maxz),
        )

    def extent(self) -> Tuple[float, float, float]:
        """Returns the X, Y, and Z extents of this AABB."""
        return (self.maxx - self.minx, self.maxy - self.miny, self.maxz - self.minz)

    def center(self) -> Tuple[float, float, float]:
        """Returns the centroid of this AABB."""
        return (
            (self.minx + self.maxx) * 0.5,
            (self.miny + self.maxy) * 0.5,
            (self.minz + self.maxz) * 0.5,
        )

    def overlaps(self, other: "AABB") -> bool:
        """Returns True if this AABB overlaps the input AABB."""
        if not isinstance(other, AABB):
            return False
        return not (
            self.maxx < other.minx
            or self.minx > other.maxx
            or self.maxy < other.miny
            or self.miny > other.maxy
            or self.maxz < other.minz
            or self.minz > other.maxz
        )

    def contains_point(self, p: Tuple[float, float, float]) -> bool:
        """Returns True if this AABB contains the input point."""
        try:
            x, y, z = p
        except Exception:
            return False
        return (
            self.minx <= x <= self.maxx
            and self.miny <= y <= self.maxy
            and self.minz <= z <= self.maxz
        )

    def ray_intersect(
        self,
        ro: Tuple[float, float, float],
        rd: Tuple[float, float, float],
    ) -> Tuple[bool, float, float]:
        """
        Tests ray-box intersection using the slab method.

        Returns
        -------
        tuple
            (hit, tmin, tmax), where point = ro + t*rd.
        """
        try:
            ox, oy, oz = ro
            dx, dy, dz = rd
        except Exception:
            return False, -math.inf, math.inf

        tmin = -math.inf
        tmax = math.inf

        def axis(o, d, mn, mx, tmin, tmax):
            if abs(d) < 1e-15:
                if o < mn or o > mx:
                    return False, tmin, tmax
                return True, tmin, tmax

            inv_d = 1.0 / d
            t0 = (mn - o) * inv_d
            t1 = (mx - o) * inv_d
            if t0 > t1:
                t0, t1 = t1, t0
            tmin = max(tmin, t0)
            tmax = min(tmax, t1)
            if tmax < tmin:
                return False, tmin, tmax
            return True, tmin, tmax

        ok, tmin, tmax = axis(ox, dx, self.minx, self.maxx, tmin, tmax)
        if not ok:
            return False, tmin, tmax
        ok, tmin, tmax = axis(oy, dy, self.miny, self.maxy, tmin, tmax)
        if not ok:
            return False, tmin, tmax
        ok, tmin, tmax = axis(oz, dz, self.minz, self.maxz, tmin, tmax)
        if not ok:
            return False, tmin, tmax
        return True, tmin, tmax


# ----------------------------
# BVH Node
# ----------------------------
@dataclass
class _BVHNode:
    bbox: AABB
    left: Optional[int]
    right: Optional[int]
    start: int
    count: int

    def is_leaf(self) -> bool:
        return self.count > 0


# ----------------------------
# BVH
# ----------------------------
class BVH:
    """
    Basic Bounding Volume Hierarchy over TopologicPy topologies.

    Usage
    -----
    faces = Topology.Faces(some_topology)
    bvh = BVH.ByTopologies(faces, maxLeafSize=4, tolerance=0.0001, silent=False)

    hits = BVH.QueryAABB(bvh, AABB(minx, miny, minz, maxx, maxy, maxz))
    candidates = BVH.Raycast(bvh, origin_vertex, [dx, dy, dz])
    nearest = BVH.Nearest(bvh, query_vertex)
    """

    def __init__(self):
        self.nodes: List[_BVHNode] = []
        self.items: List[Any] = []
        self.bboxes: List[AABB] = []
        self.centroids: List[Tuple[float, float, float]] = []
        self.aabbs: List[Tuple[float, float, float, float, float, float]] = []
        self._leaf_items: List[int] = []
        self._root: Optional[int] = None

    # ---------- Public API ----------

    @staticmethod
    def ByTopologies(
        *topologies,
        maxLeafSize: int = 4,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> Optional["BVH"]:
        """
        Creates a BVH tree from the input topologies.

        Parameters
        ----------
        *topologies : topologic_core.Topology
            One or more topologies, or nested lists of topologies, to include in the BVH.
        maxLeafSize : int , optional
            The maximum number of primitives stored in a leaf node. Default is 4.
        tolerance : float , optional
            Padding around each AABB. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        BVH
            The created BVH tree.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if isinstance(maxLeafSize, bool) or not isinstance(maxLeafSize, int) or maxLeafSize < 1:
            if not silent:
                print("BVH.ByTopologies - Error: The input maxLeafSize parameter must be a positive integer. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("BVH.ByTopologies - Error: The input tolerance parameter is not numeric. Returning None.")
            return None

        topology_list = Helper.Flatten(list(topologies))
        topology_list = [t for t in topology_list if Topology.IsInstance(t, "Topology")]

        if len(topology_list) == 0:
            if not silent:
                print("BVH.ByTopologies - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None

        bvh = BVH()
        aabb_cache = {}

        def _topology_aabb(topo):
            tid = id(topo)
            if tid in aabb_cache:
                return aabb_cache[tid]

            xmin = ymin = zmin = float("inf")
            xmax = ymax = zmax = float("-inf")

            try:
                verts = Topology.Vertices(topo, silent=True)
            except TypeError:
                try:
                    verts = Topology.Vertices(topo)
                except Exception:
                    verts = []
            except Exception:
                verts = []

            if not isinstance(verts, list) or len(verts) == 0:
                return None

            for v in verts:
                try:
                    x, y, z = Vertex.Coordinates(v)
                    x = float(x)
                    y = float(y)
                    z = float(z)
                except Exception:
                    continue

                xmin = min(xmin, x)
                ymin = min(ymin, y)
                zmin = min(zmin, z)
                xmax = max(xmax, x)
                ymax = max(ymax, y)
                zmax = max(zmax, z)

            if (
                xmin == float("inf")
                or ymin == float("inf")
                or zmin == float("inf")
                or xmax == float("-inf")
                or ymax == float("-inf")
                or zmax == float("-inf")
            ):
                return None

            aabb = (xmin, ymin, zmin, xmax, ymax, zmax)
            aabb_cache[tid] = aabb
            return aabb

        for topo in topology_list:
            aabb = _topology_aabb(topo)

            if aabb is None:
                if not silent:
                    print("BVH.ByTopologies - Warning: Invalid topology with no vertices. Skipping.")
                continue

            xmin, ymin, zmin, xmax, ymax, zmax = aabb
            box = AABB.from_points([(xmin, ymin, zmin), (xmax, ymax, zmax)], pad=tolerance)

            bvh.items.append(topo)
            bvh.bboxes.append(box)
            bvh.centroids.append(box.center())
            bvh.aabbs.append(aabb)

        indices = list(range(len(bvh.items)))
        if not indices:
            if not silent:
                print("BVH.ByTopologies - Warning: no items to build. Returning an empty BVH.")
            return bvh

        bvh.nodes = []
        bvh._leaf_items = []
        bvh._root = bvh._build_recursive(indices, maxLeafSize)

        if not silent:
            depth = BVH.Depth(bvh)
            print(f"BVH.ByTopologies - Information: Built with {len(bvh.items)} items, {len(bvh.nodes)} nodes, depth ~{depth}.")

        return bvh

    @staticmethod
    def Depth(bvh) -> Optional[int]:
        """
        Returns the depth of the BVH tree.

        Parameters
        ----------
        bvh : BVH
            The BVH tree.

        Returns
        -------
        int
            The depth of the BVH tree.
        """
        if not isinstance(bvh, BVH):
            return None

        if bvh._root is None:
            return 0

        def _depth(i: int) -> int:
            node = bvh.nodes[i]
            if node.is_leaf():
                return 1
            return 1 + max(_depth(node.left), _depth(node.right))  # type: ignore[arg-type]

        return _depth(bvh._root)

    @staticmethod
    def QueryAABB(bvh, query_box: AABB):
        """
        Returns item indices whose AABBs overlap the input query AABB.

        Invalid input returns None. A valid but empty BVH returns an empty list.
        """
        if not isinstance(bvh, BVH):
            return None
        if not isinstance(query_box, AABB):
            return None

        out: List[int] = []
        if bvh._root is None:
            return out

        stack = [bvh._root]
        while stack:
            ni = stack.pop()
            node = bvh.nodes[ni]
            if not node.bbox.overlaps(query_box):
                continue

            if node.is_leaf():
                for k in range(node.start, node.start + node.count):
                    idx = bvh._leaf_items[k]
                    if bvh.bboxes[idx].overlaps(query_box):
                        out.append(idx)
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        return out

    @staticmethod
    def Clashes(
        bvh,
        *topologies,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns BVH items whose AABBs overlap the AABBs of the input topologies.

        This is an AABB-level broad-phase query. Use precise geometric predicates or
        Boolean operations afterwards when exact intersection classification is required.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not isinstance(bvh, BVH):
            if not silent:
                print("BVH.Clashes - Error: The input bvh parameter is not a valid BVH tree. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("BVH.Clashes - Error: The input tolerance parameter is not numeric. Returning None.")
            return None

        topology_list = Helper.Flatten(list(topologies))
        topology_list = [t for t in topology_list if Topology.IsInstance(t, "Topology")]

        if len(topology_list) == 0:
            if not silent:
                print("BVH.Clashes - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None

        return_topologies = []

        for topology in topology_list:
            if Topology.IsInstance(topology, "Vertex"):
                try:
                    x, y, z = Vertex.Coordinates(topology, mantissa=mantissa)
                except Exception:
                    continue
                aabb_box = AABB(
                    x - tolerance,
                    y - tolerance,
                    z - tolerance,
                    x + tolerance,
                    y + tolerance,
                    z + tolerance,
                )
            else:
                try:
                    vertices = Topology.Vertices(topology, silent=True)
                except TypeError:
                    vertices = Topology.Vertices(topology)
                except Exception:
                    vertices = []

                points = []
                for v in vertices:
                    try:
                        points.append(Vertex.Coordinates(v, mantissa=mantissa))
                    except Exception:
                        continue

                if len(points) == 0:
                    continue

                aabb_box = AABB.from_points(points, pad=tolerance)

            hit_indices = BVH.QueryAABB(bvh, aabb_box)
            if isinstance(hit_indices, list):
                return_topologies.extend([bvh.items[i] for i in hit_indices])

        return return_topologies

    @staticmethod
    def Raycast(
        bvh,
        origin,
        direction: Tuple[float, float, float],
        mantissa: int = 6,
        silent: bool = False,
    ) -> Optional[List[int]]:
        """
        Returns candidate item indices whose AABBs are intersected by the input ray.

        Parameters
        ----------
        bvh : BVH
            The BVH tree.
        origin : topologic_core.Vertex
            The origin of the ray.
        direction : list or tuple
            A three-component direction vector.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Candidate item indices.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(bvh, BVH):
            if not silent:
                print("BVH.Raycast - Error: The input bvh parameter is not a valid BVH tree. Returning None.")
            return None

        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("BVH.Raycast - Error: The input origin parameter is not a valid topologic Vertex. Returning None.")
            return None

        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            if not silent:
                print("BVH.Raycast - Error: The input direction parameter is not a valid vector. Returning None.")
            return None

        try:
            dx, dy, dz = [float(v) for v in direction]
        except Exception:
            if not silent:
                print("BVH.Raycast - Error: The input direction parameter contains non-numeric values. Returning None.")
            return None

        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag <= 0:
            if not silent:
                print("BVH.Raycast - Error: The input direction parameter is a zero vector. Returning None.")
            return None

        direction_tuple = (dx / mag, dy / mag, dz / mag)

        try:
            o_coords = tuple(Vertex.Coordinates(origin, mantissa=mantissa))
        except Exception:
            if not silent:
                print("BVH.Raycast - Error: Could not read the input origin coordinates. Returning None.")
            return None

        out: List[int] = []
        if bvh._root is None:
            if not silent:
                print("BVH.Raycast - Warning: The input bvh parameter is empty. Returning an empty list.")
            return out

        stack = [bvh._root]
        while stack:
            ni = stack.pop()
            node = bvh.nodes[ni]
            hit, _tmin, tmax = node.bbox.ray_intersect(o_coords, direction_tuple)
            if not hit or tmax < 0:
                continue

            if node.is_leaf():
                for k in range(node.start, node.start + node.count):
                    idx = bvh._leaf_items[k]
                    h2, _tmin2, tmax2 = bvh.bboxes[idx].ray_intersect(o_coords, direction_tuple)
                    if h2 and tmax2 >= 0:
                        out.append(idx)
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        return out

    @staticmethod
    def Nearest(bvh, vertex, mantissa: int = 6, silent: bool = False):
        """
        Returns the BVH item whose AABB centroid is nearest to the input vertex.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(bvh, BVH):
            if not silent:
                print("BVH.Nearest - Error: The input bvh parameter is not a valid BVH tree. Returning None.")
            return None

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("BVH.Nearest - Error: The input vertex parameter is not a valid topologic Vertex. Returning None.")
            return None

        if bvh._root is None or not bvh.items:
            if not silent:
                print("BVH.Nearest - Warning: The input BVH tree is empty. Returning None.")
            return None

        try:
            point = tuple(Vertex.Coordinates(vertex, mantissa=mantissa))
        except Exception:
            if not silent:
                print("BVH.Nearest - Error: Could not read the input vertex coordinates. Returning None.")
            return None

        best_idx = -1
        best_d2 = float("inf")

        def d2_point_aabb(p: Tuple[float, float, float], b: AABB) -> float:
            px, py, pz = p
            dx = 0.0
            if px < b.minx:
                dx = b.minx - px
            elif px > b.maxx:
                dx = px - b.maxx

            dy = 0.0
            if py < b.miny:
                dy = b.miny - py
            elif py > b.maxy:
                dy = py - b.maxy

            dz = 0.0
            if pz < b.minz:
                dz = b.minz - pz
            elif pz > b.maxz:
                dz = pz - b.maxz

            return dx * dx + dy * dy + dz * dz

        stack = [bvh._root]
        while stack:
            ni = stack.pop()
            node = bvh.nodes[ni]
            if d2_point_aabb(point, node.bbox) >= best_d2:
                continue

            if node.is_leaf():
                for k in range(node.start, node.start + node.count):
                    idx = bvh._leaf_items[k]
                    cx, cy, cz = bvh.centroids[idx]
                    dx = cx - point[0]
                    dy = cy - point[1]
                    dz = cz - point[2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = idx
            else:
                if node.left is None or node.right is None:
                    continue

                left = bvh.nodes[node.left]
                right = bvh.nodes[node.right]
                dl = d2_point_aabb(point, left.bbox)
                dr = d2_point_aabb(point, right.bbox)

                if dl < dr:
                    stack.append(node.right)
                    stack.append(node.left)
                else:
                    stack.append(node.left)
                    stack.append(node.right)

        if best_idx < 0:
            return None

        return bvh.items[best_idx]

    # ---------- Internal build ----------

    def _build_recursive(self, indices: List[int], max_leaf_size: int) -> int:
        """Builds a subtree for indices and returns its node index."""
        if len(indices) == 0:
            raise ValueError("BVH._build_recursive - indices cannot be empty.")

        if max_leaf_size < 1:
            raise ValueError("BVH._build_recursive - max_leaf_size must be >= 1.")

        node_bbox = self.bboxes[indices[0]]
        cx_min = cx_max = self.centroids[indices[0]][0]
        cy_min = cy_max = self.centroids[indices[0]][1]
        cz_min = cz_max = self.centroids[indices[0]][2]

        for i in indices[1:]:
            node_bbox = AABB.union(node_bbox, self.bboxes[i])
            cx, cy, cz = self.centroids[i]
            if cx < cx_min:
                cx_min = cx
            if cx > cx_max:
                cx_max = cx
            if cy < cy_min:
                cy_min = cy
            if cy > cy_max:
                cy_max = cy
            if cz < cz_min:
                cz_min = cz
            if cz > cz_max:
                cz_max = cz

        if len(indices) <= max_leaf_size:
            start = len(self._leaf_items)
            self._leaf_items.extend(indices)
            node = _BVHNode(node_bbox, None, None, start, len(indices))
            self.nodes.append(node)
            return len(self.nodes) - 1

        ex = cx_max - cx_min
        ey = cy_max - cy_min
        ez = cz_max - cz_min
        if ex >= ey and ex >= ez:
            axis = 0
        elif ey >= ex and ey >= ez:
            axis = 1
        else:
            axis = 2

        indices.sort(key=lambda i: self.centroids[i][axis])
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]

        if not left_indices or not right_indices:
            left_indices = indices[: len(indices) // 2]
            right_indices = indices[len(indices) // 2 :]

        if not left_indices or not right_indices:
            # This should only be reachable when len(indices) == 1, which the leaf
            # case above handles. Keep a defensive fallback.
            start = len(self._leaf_items)
            self._leaf_items.extend(indices)
            node = _BVHNode(node_bbox, None, None, start, len(indices))
            self.nodes.append(node)
            return len(self.nodes) - 1

        left_node = self._build_recursive(left_indices, max_leaf_size)
        right_node = self._build_recursive(right_indices, max_leaf_size)
        node = _BVHNode(node_bbox, left_node, right_node, start=0, count=0)
        self.nodes.append(node)
        return len(self.nodes) - 1
