# Copyright (C) 2025
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any, Iterable
import math

# TopologicPy imports (used defensively to keep this file standalone-friendly)
try:
    from topologicpy.Vertex import Vertex
    from topologicpy.Edge import Edge
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
except Exception:
    # If TopologicPy isn't present in the current environment, we still allow type checking.
    Vertex = Edge = Face = Topology = object  # type: ignore


# ----------------------------
# Axis-Aligned Bounding Box
# ----------------------------
@dataclass
class AABB:
    """Axis-aligned bounding box: [minx,miny,minz]..[maxx,maxy,maxz]."""
    minx: float; miny: float; minz: float
    maxx: float; maxy: float; maxz: float

    @staticmethod
    def from_points(pts: Iterable[Tuple[float, float, float]], pad: float = 0.0) -> "AABB":
        it = iter(pts)
        try:
            x, y, z = next(it)
        except StopIteration:
            # Empty: return a degenerate box at origin
            return AABB(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        minx = maxx = float(x)
        miny = maxy = float(y)
        minz = maxz = float(z)
        for x, y, z in it:
            if x < minx: minx = x
            if x > maxx: maxx = x
            if y < miny: miny = y
            if y > maxy: maxy = y
            if z < minz: minz = z
            if z > maxz: maxz = z
        if pad:
            minx -= pad; miny -= pad; minz -= pad
            maxx += pad; maxy += pad; maxz += pad
        return AABB(minx, miny, minz, maxx, maxy, maxz)

    @staticmethod
    def union(a: "AABB", b: "AABB") -> "AABB":
        return AABB(
            min(a.minx, b.minx), min(a.miny, b.miny), min(a.minz, b.minz),
            max(a.maxx, b.maxx), max(a.maxy, b.maxy), max(a.maxz, b.maxz),
        )

    def extent(self) -> Tuple[float, float, float]:
        return (self.maxx - self.minx, self.maxy - self.miny, self.maxz - self.minz)

    def center(self) -> Tuple[float, float, float]:
        return ((self.minx + self.maxx) * 0.5, (self.miny + self.maxy) * 0.5, (self.minz + self.maxz) * 0.5)

    def overlaps(self, other: "AABB") -> bool:
        return not (self.maxx < other.minx or self.minx > other.maxx or
                    self.maxy < other.miny or self.miny > other.maxy or
                    self.maxz < other.minz or self.minz > other.maxz)

    def contains_point(self, p: Tuple[float, float, float]) -> bool:
        x, y, z = p
        return (self.minx <= x <= self.maxx and
                self.miny <= y <= self.maxy and
                self.minz <= z <= self.maxz)

    def ray_intersect(self, ro: Tuple[float, float, float], rd: Tuple[float, float, float]) -> Tuple[bool, float, float]:
        """Ray-box intersection using the 'slab' method.
        Returns (hit, tmin, tmax) in ray param t, where point = ro + t*rd."""
        (ox, oy, oz) = ro
        (dx, dy, dz) = rd
        tmin = -math.inf
        tmax = math.inf

        def axis(o, d, mn, mx, tmin, tmax):
            if abs(d) < 1e-15:
                # Ray parallel to slab: reject if origin not within slab
                if o < mn or o > mx:
                    return False, tmin, tmax
                return True, tmin, tmax
            invD = 1.0 / d
            t0 = (mn - o) * invD
            t1 = (mx - o) * invD
            if t0 > t1:
                t0, t1 = t1, t0
            tmin = max(tmin, t0)
            tmax = min(tmax, t1)
            if tmax < tmin:
                return False, tmin, tmax
            return True, tmin, tmax

        ok, tmin, tmax = axis(ox, dx, self.minx, self.maxx, tmin, tmax)
        if not ok: return (False, tmin, tmax)
        ok, tmin, tmax = axis(oy, dy, self.miny, self.maxy, tmin, tmax)
        if not ok: return (False, tmin, tmax)
        ok, tmin, tmax = axis(oz, dz, self.minz, self.maxz, tmin, tmax)
        if not ok: return (False, tmin, tmax)
        return True, tmin, tmax


# ----------------------------
# BVH Node
# ----------------------------
@dataclass
class _BVHNode:
    bbox: AABB
    left: Optional[int]   # index into nodes list
    right: Optional[int]  # index into nodes list
    start: int            # start index into items array (for leaves)
    count: int            # number of items (for leaves). If count>0, node is leaf.

    def is_leaf(self) -> bool:
        return self.count > 0


# ----------------------------
# BVH
# ----------------------------
class BVH:
    """
    Basic Bounding Volume Hierarchy over TopologicPy topologies.

    Usage:
        # 1) Prepare your primitives (Faces, Edges, Cells, etc.)
        faces = Topology.Faces(some_topology)  # or any list of topologies

        # 2) Build the BVH
        bvh = BVH.FromTopologies(faces, max_leaf_size=4, pad=0.0, silent=False)

        # 3) AABB query
        hits = bvh.QueryAABB(AABB(minx, miny, minz, maxx, maxy, maxz))

        # 4) Raycast (rough): returns candidate primitive indices
        cand = bvh.Raycast((ox,oy,oz), (dx,dy,dz))

        # 5) Nearest by centroid (coarse)
        idx, dist = bvh.Nearest((x,y,z))
        primitive = bvh.items[idx]
    """

    def __init__(self):
        self.nodes: List[_BVHNode] = []
        self.items: List[Any] = []       # original topologies
        self.bboxes: List[AABB] = []     # per item bbox
        self.centroids: List[Tuple[float, float, float]] = []  # per item centroid
        self._root: Optional[int] = None

    # ---------- Public API ----------

    @staticmethod
    def ByTopologies(
        *topologies,
        maxLeafSize: int = 4,
        tolerance: float = 0.0001,
        silent: bool = False
    ) -> "BVH":
        """
        Creates a BVH Tree from the input list of topologies. The input can be individual topologies each as an input argument or a list of topologies stored in one input argument.

        Parameters
        ----------
        *topologies: (tuple of Topologic topologies)
            One or more TopologicPy topologies to include in the BVH.
            Each topology is automatically analyzed to extract its vertices and compute an axis-aligned bounding box (AABB)
            for hierarchical spatial indexing.

        maxLeafSize: int , optional
            The maximum number of primitives (topologies) that can be stored in a single leaf node of the BVH.
            Smaller values result in deeper trees with finer spatial subdivision (potentially faster queries but slower build times),
            while larger values produce shallower trees with coarser spatial grouping (faster builds but less precise queries).
            Default is 4.
        tolerance : float , optional
            The desired tolerance. Tolerance is used for an optional margin added to all sides of each topology's axis-aligned bounding box (AABB).
            This helps account for numerical precision errors or slight geometric inaccuracies.
            A small positive value ensures that closely adjacent or nearly touching primitives are
            properly enclosed within their bounding boxes. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        BVH tree
            The created BVH tree.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        topologyList = Helper.Flatten(list(topologies))
        topologyList = [t for t in topologyList if Topology.IsInstance(t, "Topology")]

        if len(topologyList) == 0:
            if not silent:
                print("BVH.ByTopologies - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None
        
        bvh = BVH()

        # Precompute per-item AABBs & centroids
        bvh.items = topologyList
        bvh.bboxes = []
        bvh.centroids = []

        for topo in bvh.items:
            pts = [Vertex.Coordinates(v) for v in Topology.Vertices(topo)]
            if not pts:
                # Degenerate: keep a tiny box at (0,0,0) to avoid crashes
                box = AABB.from_points([(0.0, 0.0, 0.0)], pad=tolerance)
                c = (0.0, 0.0, 0.0)
            else:
                box = AABB.from_points(pts, pad=tolerance)
                c = box.center()
            bvh.bboxes.append(box)
            bvh.centroids.append(c)

        # Build using indices
        indices = list(range(len(bvh.items)))
        if not indices:
            if not silent:
                print("BVH.Topologies - Warning: no items to build.")
            return bvh

        # Reserve nodes list
        bvh.nodes = []
        bvh._root = bvh._build_recursive(indices, maxLeafSize)
        if not silent:
            depth = bvh.Depth(bvh)
            print(f"BVH.ByTopologies - Information: Built with {len(bvh.items)} items, {len(bvh.nodes)} nodes, depth ~{depth}.")
        return bvh

    @staticmethod
    def Depth(bvh) -> int:
        """
        Returns an approximate depth of the BVH.
        
        Parameters
        ----------
        bvh : BVH
            The bvh tree.
        
        Returns
        -------
        int
            The approximate depth of the input bvh tree.

        """
        def _depth(i: int) -> int:
            n = bvh.nodes[i]
            if n.is_leaf(): return 1
            return 1 + max(_depth(n.left), _depth(n.right))  # type: ignore
        if bvh._root is None: return 0
        return _depth(bvh._root)

    @staticmethod
    def QueryAABB(bvh, query_box: AABB):
        """Return indices of items whose AABBs overlap query_box."""
        out: List[int] = []
        if bvh._root is None: return out
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
                stack.append(node.left)   # type: ignore
                stack.append(node.right)  # type: ignore
        return out
    
    @staticmethod
    def Clashes(bvh, *topologies, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns candidate primitives (topologies) overlapping the BVH (AABB-level) of the input topologies list.
        You can follow up with precise TopologicPy geometry intersection if needed.
        
        Parameters
        ----------
        bvh : BVH
            The bvh tree.
        *topologies: (tuple of Topologic topologies)
            One or more TopologicPy topologies to include in the BVH.
            Each topology is automatically analyzed to extract its vertices and compute an axis-aligned bounding box (AABB)
            for hierarchical spatial indexing.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001. Tolerance is used for an optional margin added to all sides of each topology's axis-aligned bounding box (AABB).
            This helps account for numerical precision errors or slight geometric inaccuracies.
            A small positive value ensures that closely adjacent or nearly touching primitives are
            properly enclosed within their bounding boxes. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of topologies that broadly interest the input list of topologies.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        topologyList = Helper.Flatten(list(topologies))
        topologyList = [t for t in topologyList if Topology.IsInstance(t, "Topology")]

        if len(topologyList) == 0:
            if not silent:
                print("BVH.Clashes - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None
        
        return_topologies = []
        for topology in topologyList:
            if Topology.IsInstance(topology, "vertex"):
                x,y,z = Vertex.Coordinates(topology, mantissa=mantissa)
                points = [[x-tolerance, y-tolerance, z-tolerance], [x+tolerance, y+tolerance, z+tolerance]]
            else:
                points = [Vertex.Coordinates(v, mantissa=mantissa) for v in Topology.Vertices(topology)]
            aabb_box = AABB.from_points(points, pad = tolerance)
            return_topologies.extend([bvh.items[i] for i in BVH.QueryAABB(bvh, aabb_box)])
        return return_topologies              

    @staticmethod
    def Raycast(bvh, origin, direction: Tuple[float, float, float], mantissa: int = 6, silent: bool = False) -> List[int]:
        """
        Returns candidate primitives intersecting the BVH (AABB-level).
        You can follow up with precise TopologicPy geometry intersection if needed.
        
        Parameters
        ----------
        bvh : BVH
            The bvh tree.
        origin : topologic_core.Vertex
            The origin of the ray vector
        direction : topologic_core.Vector
            The direction of the raycast vector.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The list of the indices of the possible candidates interesecting the input ray vector.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(bvh, BVH):
            if not silent:
                print("BVH.Raycast - Error: The input bvh parameter is not a valid BVH tree. Returning None.")
            return None
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("BVH.Raycast - Error: The input origin parameter is not a valid topologic Vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("BVH.Raycast - Error: The input direction parameter is not a valid vector. Returning None.")
            return None
        if not len(direction) < 3:
            if not silent:
                print("BVH.Raycast - Error: The input direction parameter is not a valid vector. Returning None.")
            return None
        o_coords = Vertex.Coordinates(origin, mantissa=mantissa)
        out: List[int] = []
        if bvh._root is None:
            if not silent:
                print("BVH.Raycast - Warning: The input bvh parameter is empty. Returning an empty list.")
            return out

        # Normalize direction if possible (not strictly required)
        dx, dy, dz = direction
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            direction = (dx/mag, dy/mag, dz/mag)

        stack = [bvh._root]
        while stack:
            ni = stack.pop()
            node = bvh.nodes[ni]
            hit, tmin, tmax = node.bbox.ray_intersect(o_coords, direction)
            if not hit or tmax < 0:
                continue
            if node.is_leaf():
                for k in range(node.start, node.start + node.count):
                    idx = bvh._leaf_items[k]
                    h2, _, _ = bvh.bboxes[idx].ray_intersect(o_coords, direction)
                    if h2:
                        out.append(idx)
            else:
                stack.append(node.left)   # type: ignore
                stack.append(node.right)  # type: ignore
        return out

    @staticmethod
    def Nearest(bvh, vertex, mantissa: int = 6, silent: bool = False):
        """
        Returns the topology with centroid nearest to the input vertex.
        Uses AABB distance lower-bounds to prune search.
        Parameters
        ----------
        bvh : BVH
            The bvh tree.
        vertex : topologic_core.Vertex
            The input vertex.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Topology
            The topology with centroid nearest to the input vertex.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not isinstance(bvh, BVH):
            if not silent:
                print("BVH.Nearest - Error: The input bvh parameter is not a valid BVH tree. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "vertex"):
            if not silent:
                print("BVH.Nearest - Error: The input vertex parameter is not a valid topologic Vertex. Returning None.")
            return None
        if bvh._root is None or not bvh.items:
            if not silent:
                print("BVH.Nearest - Warning: The input bhv tree is empty. Returning None.")
            return None

        best_idx = -1
        best_d2 = float("inf")

        def d2_point_aabb(p: Tuple[float, float, float], b: AABB) -> float:
            px, py, pz = p
            dx = 0.0
            if px < b.minx: dx = b.minx - px
            elif px > b.maxx: dx = px - b.maxx
            dy = 0.0
            if py < b.miny: dy = b.miny - py
            elif py > b.maxy: dy = py - b.maxy
            dz = 0.0
            if pz < b.minz: dz = b.minz - pz
            elif pz > b.maxz: dz = pz - b.maxz
            return dx*dx + dy*dy + dz*dz

        stack = [bvh._root]
        point = Vertex.Coordinates(vertex, mantissa=mantissa)
        while stack:
            ni = stack.pop()
            node = bvh.nodes[ni]
            if d2_point_aabb(point, node.bbox) >= best_d2:
                continue
            if node.is_leaf():
                for k in range(node.start, node.start + node.count):
                    idx = bvh._leaf_items[k]
                    cx, cy, cz = bvh.centroids[idx]
                    dx = cx - point[0]; dy = cy - point[1]; dz = cz - point[2]
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = idx
            else:
                # Visit child likely nearer first to improve pruning
                l = bvh.nodes[node.left]  # type: ignore
                r = bvh.nodes[node.right] # type: ignore
                dl = d2_point_aabb(point, l.bbox)
                dr = d2_point_aabb(point, r.bbox)
                if dl < dr:
                    stack.append(node.right) # type: ignore
                    stack.append(node.left)  # type: ignore
                else:
                    stack.append(node.left)  # type: ignore
                    stack.append(node.right) # type: ignore

        return bvh.items[best_idx]

    # ---------- Internal build ----------

    def _build_recursive(self, indices: List[int], max_leaf_size: int) -> int:
        """Builds a subtree for 'indices' and returns node index."""
        # Compute node bbox and centroid bbox
        node_bbox = self.bboxes[indices[0]]
        cx_min = cx_max = self.centroids[indices[0]][0]
        cy_min = cy_max = self.centroids[indices[0]][1]
        cz_min = cz_max = self.centroids[indices[0]][2]
        for i in indices[1:]:
            node_bbox = AABB.union(node_bbox, self.bboxes[i])
            cx, cy, cz = self.centroids[i]
            if cx < cx_min: cx_min = cx
            if cx > cx_max: cx_max = cx
            if cy < cy_min: cy_min = cy
            if cy > cy_max: cy_max = cy
            if cz < cz_min: cz_min = cz
            if cz > cz_max: cz_max = cz

        if len(indices) <= max_leaf_size:
            start = len(getattr(self, "_leaf_items", []))
            if not hasattr(self, "_leaf_items"):
                self._leaf_items: List[int] = []
            self._leaf_items.extend(indices)
            node = _BVHNode(node_bbox, None, None, start, len(indices))
            self.nodes.append(node)
            return len(self.nodes) - 1

        # Choose split axis (longest centroid axis)
        ex = cx_max - cx_min
        ey = cy_max - cy_min
        ez = cz_max - cz_min
        if ex >= ey and ex >= ez:
            axis = 0
        elif ey >= ex and ey >= ez:
            axis = 1
        else:
            axis = 2

        # Median split by centroid along chosen axis
        mid = len(indices) // 2
        indices.sort(key=lambda i: self.centroids[i][axis])
        left_idx = indices[:mid]
        right_idx = indices[mid:]

        # Handle pathological case (all centroids equal) by forcing a split
        if not left_idx or not right_idx:
            left_idx = indices[:len(indices)//2]
            right_idx = indices[len(indices)//2:]

        left_node = self._build_recursive(left_idx, max_leaf_size)
        right_node = self._build_recursive(right_idx, max_leaf_size)
        node = _BVHNode(node_bbox, left_node, right_node, start=0, count=0)
        self.nodes.append(node)
        return len(self.nodes) - 1