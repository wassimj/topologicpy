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

    @staticmethod
    def PrepareScene(topologies, linearDeflection=0.01, angularDeflection=5.0,
                     tolerance=0.0001, maxLeafSize=8, cancelled=None):
        """Prepare two-sided surface triangles for repeated actual-hit queries.

        Unlike Raycast, RayHits/IsOccluded intersect trimmed geometry, not boxes.
        Deflection is absolute in model units; angles are degrees. Curved faces
        are approximated by the active backend's Tessellate. Holes are retained.
        Each triangle retains its input topology index and face index. Empty
        input is a valid clear scene; failed/empty face tessellation raises.
        The returned scene is an opaque snapshot; rebuild after geometry edits.
        """
        import numpy as np
        from topologicpy.Topology import Topology
        for name, value in [('linearDeflection', linearDeflection),
                            ('angularDeflection', angularDeflection), ('tolerance', tolerance)]:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(name + ' must be positive and finite')
        if angularDeflection > 180:
            raise ValueError('angularDeflection must not exceed 180 degrees')
        inputs = list(topologies) if isinstance(topologies, (list, tuple)) else [topologies]
        triangles, sources, faces = [], [], []
        face_index = 0
        for source, topology in enumerate(inputs):
            if not Topology.IsInstance(topology, 'Topology'):
                raise ValueError('Every scene item must be a topology')
            items = [topology] if Topology.IsInstance(topology, 'Face') else Topology.Faces(topology)
            if not items:
                raise ValueError('Scene topology contains no shading faces')
            for face in items:
                if cancelled is not None and cancelled():
                    raise InterruptedError('Scene preparation cancelled')
                mesh = Topology.Tessellate(face, linearDeflection=linearDeflection,
                    angularDeflection=angularDeflection, relative=False,
                    weld=False, mantissa=12, silent=True)
                if not mesh or not mesh.get('faces'):
                    raise ValueError('Could not tessellate shading face ' + str(face_index))
                xyz = np.asarray(mesh['vertices'], dtype=float)
                for indices in mesh['faces']:
                    if len(indices) != 3:
                        raise ValueError('Tessellator returned a non-triangle')
                    triangles.append(xyz[indices].tolist())
                    sources.append(source)
                    faces.append(face_index)
                face_index += 1
        scene = BVH.ByTriangles(triangles, sourceIds=sources, faceIds=faces,
                                tolerance=tolerance, maxLeafSize=maxLeafSize)
        scene['metadata'].update(linearDeflection=linearDeflection,
            angularDeflection=angularDeflection, representation='backend tessellation')
        return scene

    @staticmethod
    def ByTriangles(triangles, sourceIds=None, faceIds=None, tolerance=0.0001, maxLeafSize=8):
        """Prepare explicit (N,3,3) triangles, without requiring a geometry backend.

        Coordinates must be finite and triangles nondegenerate. Source/face IDs
        are optional integer arrays. Distances use model units. Triangle indices
        are stable primitive IDs. Input arrays are copied and made read-only.
        """
        import numpy as np
        if not math.isfinite(float(tolerance)) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        if isinstance(maxLeafSize, bool) or not isinstance(maxLeafSize, int) or maxLeafSize < 1:
            raise ValueError('maxLeafSize must be a positive integer')
        tri = np.array(triangles, dtype=float, copy=True)
        if tri.size == 0:
            tri = np.empty((0, 3, 3), dtype=float)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3) or not np.isfinite(tri).all():
            raise ValueError('triangles must have finite shape (N,3,3)')
        n = len(tri)
        if n and np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) <= tolerance*tolerance):
            raise ValueError('Degenerate shading triangle')
        offset = (tri.min(axis=(0,1)) + tri.max(axis=(0,1))) / 2 if n else np.zeros(3)
        tri -= offset
        def ids(values, default):
            arr = np.asarray(default if values is None else values)
            if arr.shape != (n,) or (n and not np.issubdtype(arr.dtype, np.integer)):
                raise ValueError('IDs must be integer arrays with one entry per triangle')
            return arr.astype(np.int64, copy=True)
        sources, faces = ids(sourceIds, np.arange(n)), ids(faceIds, np.arange(n))
        low, high = tri.min(axis=1) if n else np.empty((0,3)), tri.max(axis=1) if n else np.empty((0,3))
        centres = (low + high)/2
        nodes = []
        def build(indices):
            lo, hi = low[indices].min(axis=0)-tolerance, high[indices].max(axis=0)+tolerance
            if len(indices) <= maxLeafSize:
                nodes.append((lo, hi, -1, -1, indices))
            else:
                axis = int(np.argmax(np.ptp(centres[indices], axis=0)))
                ordered = indices[np.argsort(centres[indices,axis], kind='stable')]
                mid = len(ordered)//2
                left, right = build(ordered[:mid]), build(ordered[mid:])
                nodes.append((lo, hi, left, right, None))
            return len(nodes)-1
        root = build(np.arange(n)) if n else -1
        for arr in [tri, offset, sources, faces]:
            arr.setflags(write=False)
        return {'schema':'topologicpy.rayscene.v1', 'triangles':tri, 'offset':offset,
                'sourceIds':sources, 'faceIds':faces, 'nodes':nodes, 'root':root,
                'tolerance':float(tolerance), 'metadata':{'triangles':n,
                'representation':'explicit triangles', 'twoSided':True}}

    @staticmethod
    def _RayArrays(origins, directions):
        import numpy as np
        def array(values):
            a = np.asarray(values, dtype=float)
            if a.size == 0:
                a = np.empty((0,3))
            if a.shape == (3,):
                a = a.reshape(1,3)
            if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
                raise ValueError('Rays must be finite XYZ arrays')
            return a
        o, d = array(origins), array(directions)
        if len(o) != len(d):
            if len(o) == 1:
                o = np.broadcast_to(o, d.shape)
            elif len(d) == 1:
                d = np.broadcast_to(d, o.shape)
            else:
                raise ValueError('Ray counts must match, or one input must contain one XYZ')
        scale = np.max(np.abs(d),axis=1)
        if np.any(scale <= 0):
            raise ValueError('Ray direction cannot be zero')
        scaled = d / scale[:,None]
        return o, scaled / np.linalg.norm(scaled,axis=1)[:,None]

    @staticmethod
    def _Trace(scene, origins, directions, minDistance, maxDistance, batchSize, anyHit, cancelled):
        import numpy as np
        if not isinstance(scene, dict) or scene.get('schema') != 'topologicpy.rayscene.v1':
            raise ValueError('Expected a prepared ray scene')
        minimum = scene['tolerance'] if minDistance is None else float(minDistance)
        maximum = float(maxDistance)
        if not math.isfinite(minimum) or minimum < 0 or math.isnan(maximum) or maximum <= minimum:
            raise ValueError('Require finite 0 <= minDistance < maxDistance')
        if isinstance(batchSize, bool) or not isinstance(batchSize,int) or batchSize < 1:
            raise ValueError('batchSize must be a positive integer')
        o, d = BVH._RayArrays(origins, directions)
        distance = np.full(len(o), np.inf)
        primitive = np.full(len(o), -1, dtype=np.int64)
        triangles, tolerance = scene['triangles'], scene['tolerance']
        for start in range(0,len(o),batchSize):
            if cancelled is not None and cancelled():
                raise InterruptedError('Ray query cancelled')
            end = min(start+batchSize,len(o))
            origin = o[start:end]-scene['offset']
            direction = d[start:end]
            stack = [(scene['root'], np.arange(end-start))] if scene['root'] >= 0 else []
            while stack:
                if cancelled is not None and cancelled():
                    raise InterruptedError('Ray query cancelled')
                node_id, rays = stack.pop()
                if anyHit:
                    rays = rays[primitive[start+rays] < 0]
                if len(rays) == 0:
                    continue
                lo, hi, left, right, indices = scene['nodes'][node_id]
                ro, rd = origin[rays], direction[rays]
                near = np.full(len(rays),minimum)
                far = np.minimum(maximum, distance[start+rays])
                for axis in range(3):
                    parallel = np.abs(rd[:,axis]) < 1e-15
                    safe = np.where(parallel,1.0,rd[:,axis])
                    a, b = (lo[axis]-ro[:,axis])/safe, (hi[axis]-ro[:,axis])/safe
                    near = np.maximum(near,np.where(parallel,-np.inf,np.minimum(a,b)))
                    far = np.minimum(far,np.where(parallel,np.inf,np.maximum(a,b)))
                    far = np.where(parallel & ((ro[:,axis]<lo[axis]) | (ro[:,axis]>hi[axis])), -np.inf, far)
                rays = rays[far >= near]
                if not len(rays):
                    continue
                if indices is None:
                    stack.extend([(right,rays),(left,rays)])
                    continue
                # Vectorized Moller-Trumbore, two-sided. Coplanar rays do not
                # cross a surface. Boundary contacts count; distances <= min
                # are ignored to avoid self-hits without hiding the host face.
                ro, rd = origin[rays,None,:], direction[rays,None,:]
                tri = triangles[indices]
                e1, e2 = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
                p = np.cross(rd,e2)
                det = np.einsum('tj,rtj->rt',e1,p)
                scale = np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
                valid = np.abs(det) > 1e-12*scale
                inv = np.divide(1.0,det,out=np.zeros_like(det),where=valid)
                tv = ro-tri[:,0]
                u = np.einsum('rtj,rtj->rt',tv,p)*inv
                q = np.cross(tv,e1)
                v = np.einsum('rtj,rtj->rt',rd,q)*inv
                t = np.einsum('tj,rtj->rt',e2,q)*inv
                # Convert model-unit tolerance to conservative barycentric
                # tolerances using triangle altitudes, including skinny faces.
                twice_area = np.linalg.norm(np.cross(e1,e2),axis=1)
                eu = tolerance*np.linalg.norm(e2,axis=1)/twice_area
                ev = tolerance*np.linalg.norm(e1,axis=1)/twice_area
                ew = tolerance*np.linalg.norm(e2-e1,axis=1)/twice_area
                valid &= (u>=-eu) & (v>=-ev) & (u+v<=1+ew) & (t>minimum) & (t<=maximum)
                t = np.where(valid,t,np.inf)
                best = np.argmin(t,axis=1)
                values = t[np.arange(len(rays)),best]
                global_ids = start+rays
                better = values < distance[global_ids]
                distance[global_ids[better]] = values[better]
                primitive[global_ids[better]] = indices[best[better]]
        return o,d,distance,primitive

    @staticmethod
    def IsOccluded(scene, origins, directions, minDistance=None, maxDistance=math.inf,
                   batchSize=4096, cancelled=None):
        """Return a boolean per ray, stopping at any actual triangle hit.

        XYZ arrays are paired, with one-row broadcasting. Directions normalize
        internally. Positive minDistance defaults to scene tolerance. No entire
        building/face is excluded. maxDistance includes its endpoint. Temporary
        pair storage is bounded by batchSize * maxLeafSize; output is O(rays).
        A true-returning cancellation callback raises InterruptedError.
        """
        return BVH._Trace(scene,origins,directions,minDistance,maxDistance,batchSize,True,cancelled)[3] >= 0

    @staticmethod
    def RayHits(scene, origins, directions, minDistance=None, maxDistance=math.inf,
                batchSize=4096, cancelled=None):
        """Nearest actual hit per ray, or None; retain primitive/source/face IDs.

        Returned points and distances are in original world coordinates/units.
        See IsOccluded for boundary, self-hit, batching and cancellation policy.
        """
        o,d,dist,ids = BVH._Trace(scene,origins,directions,minDistance,maxDistance,batchSize,False,cancelled)
        return [None if primitive < 0 else {'distance':float(dist[i]),
            'point':(o[i]+d[i]*dist[i]).tolist(), 'primitiveId':int(primitive),
            'sourceId':int(scene['sourceIds'][primitive]), 'faceId':int(scene['faceIds'][primitive])}
            for i,primitive in enumerate(ids)]

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
