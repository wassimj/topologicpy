# Copyright (C) 2026
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


class Grid:
    """Utility methods for creating edge and vertex grids."""

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _IsFace(face) -> bool:
        from topologicpy.Topology import Topology
        return Topology.IsInstance(face, "Face")

    @staticmethod
    def _IsVertex(vertex) -> bool:
        from topologicpy.Topology import Topology
        return Topology.IsInstance(vertex, "Vertex")

    @staticmethod
    def _DefaultDistanceRange():
        return [-0.5, -0.25, 0.0, 0.25, 0.5]

    @staticmethod
    def _DefaultParameterRange():
        return [0.0, 0.25, 0.5, 0.75, 1.0]

    @staticmethod
    def _FloatList(values, default=None):
        if values is None:
            values = default
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            return None
        try:
            return sorted([float(v) for v in values])
        except Exception:
            return None

    @staticmethod
    def _ParameterList(values, default=None):
        values = Grid._FloatList(values, default)
        if values is None:
            return None
        if len(values) < 1:
            return values
        if min(values) < 0.0 or max(values) > 1.0:
            return None
        return values

    @staticmethod
    def _Coordinates(vertex, mantissa=6):
        from topologicpy.Vertex import Vertex
        return [
            Vertex.X(vertex, mantissa=mantissa),
            Vertex.Y(vertex, mantissa=mantissa),
            Vertex.Z(vertex, mantissa=mantissa),
        ]

    @staticmethod
    def _Subtract(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    @staticmethod
    def _Add(a, b):
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    @staticmethod
    def _Scale(v, s):
        return [v[0] * s, v[1] * s, v[2] * s]

    @staticmethod
    def _Magnitude(v):
        return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5

    @staticmethod
    def _Normalize(v, tolerance=0.0001):
        mag = Grid._Magnitude(v)
        if mag <= max(float(tolerance), 1e-12):
            return None
        return [v[0] / mag, v[1] / mag, v[2] / mag]

    @staticmethod
    def _Point(origin, u_dir, v_dir, u=0.0, v=0.0, mantissa=6):
        from topologicpy.Vertex import Vertex
        p = Grid._Add(origin, Grid._Add(Grid._Scale(u_dir, u), Grid._Scale(v_dir, v)))
        return Vertex.ByCoordinates(round(p[0], mantissa), round(p[1], mantissa), round(p[2], mantissa))

    @staticmethod
    def _Origin(face=None, origin=None, mantissa=6):
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        if Grid._IsVertex(origin):
            return origin
        if Grid._IsFace(face):
            return Face.VertexByParameters(face, 0, 0)
        return Vertex.ByCoordinates(0, 0, 0)

    @staticmethod
    def _Basis(face=None, mantissa=6, tolerance=0.0001):
        """Return normalized u and v basis vectors for a face or the world XY plane."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face

        if Grid._IsFace(face):
            p00 = Face.VertexByParameters(face, 0, 0)
            p10 = Face.VertexByParameters(face, 1, 0)
            p01 = Face.VertexByParameters(face, 0, 1)
            c00 = Grid._Coordinates(p00, mantissa=mantissa)
            c10 = Grid._Coordinates(p10, mantissa=mantissa)
            c01 = Grid._Coordinates(p01, mantissa=mantissa)
            u_vec = Grid._Subtract(c10, c00)
            v_vec = Grid._Subtract(c01, c00)
        else:
            u_vec = [1.0, 0.0, 0.0]
            v_vec = [0.0, 1.0, 0.0]

        u_dir = Grid._Normalize(u_vec, tolerance=tolerance)
        v_dir = Grid._Normalize(v_vec, tolerance=tolerance)
        return u_dir, v_dir

    @staticmethod
    def _Span(values):
        """Return a usable [min, max] span. Avoid a degenerate span for one-value ranges."""
        if values is None or len(values) < 1:
            return None
        a = min(values)
        b = max(values)
        if abs(b - a) <= 1e-12:
            # A single offset still needs a finite line. Use a symmetric unit span.
            return [a - 0.5, a + 0.5]
        return [a, b]

    @staticmethod
    def _SetDictionary(topology, keys, values):
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        d = Dictionary.ByKeysValues(keys, values)
        if d:
            Topology.SetDictionary(topology, d)
        return topology

    @staticmethod
    def _AppendEdgeResult(result, grid_edges, direction, offset):
        from topologicpy.Topology import Topology
        if not result:
            return
        if Topology.IsInstance(result, "Edge"):
            Grid._SetDictionary(result, ["dir", "offset"], [direction, offset])
            grid_edges.append(result)
            return
        try:
            if Topology.Type(result) > Topology.TypeID("Edge"):
                for edge in Topology.Edges(result):
                    if Topology.IsInstance(edge, "Edge"):
                        Grid._SetDictionary(edge, ["dir", "offset"], [direction, offset])
                        grid_edges.append(edge)
        except Exception:
            return

    @staticmethod
    def _FlattenFor2D(face, cluster, tolerance=0.0001):
        """Flatten a face and a cluster to the XY plane without relying on Topology.Centroid."""
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector

        ref = Face.VertexByParameters(face, 0, 0)
        x_tran = -Vertex.X(ref)
        y_tran = -Vertex.Y(ref)
        z_tran = -Vertex.Z(ref)

        face_2 = Topology.Translate(face, x_tran, y_tran, z_tran)
        cluster_2 = Topology.Translate(cluster, x_tran, y_tran, z_tran)

        face_normal = Face.Normal(face_2)
        if not isinstance(face_normal, list) or len(face_normal) != 3:
            return None, None

        tran_mat = Vector.TransformationMatrix(face_normal, [0, 0, 1])
        flat_face = Topology.Transform(face_2, tran_mat, transferDictionaries=False)
        flat_cluster = Topology.Transform(cluster_2, tran_mat, transferDictionaries=False)
        return flat_face, flat_cluster

    @staticmethod
    def _FilterVerticesByFace(vertices, face, tolerance=0.0001):
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not vertices:
            return []
        if not Grid._IsFace(face):
            return vertices

        cluster = Cluster.ByTopologies(vertices)
        if not cluster:
            return []

        try:
            flat_face, flat_cluster = Grid._FlattenFor2D(face, cluster, tolerance=tolerance)
            if not flat_face or not flat_cluster:
                return []
            flat_vertices = Topology.Vertices(flat_cluster)
            status_list = Vertex.IsInternal2D(flat_vertices, flat_face)
            if not isinstance(status_list, list) or len(status_list) != len(vertices):
                return []
            return [v for i, v in enumerate(vertices) if status_list[i] is True]
        except Exception:
            # Safe but slower fallback for unusual face/transform cases.
            filtered = []
            for v in vertices:
                try:
                    if Vertex.IsInternal(v, face, tolerance=tolerance):
                        filtered.append(v)
                except TypeError:
                    if Vertex.IsInternal(v, face):
                        filtered.append(v)
            return filtered

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    # @staticmethod
    # def EdgesByDistances(face=None,
    #                     uOrigin=None,
    #                     vOrigin=None,
    #                     uRange=None,
    #                     vRange=None,
    #                     clip=False,
    #                     mantissa: int = 6,
    #                     tolerance=0.0001):
    #     """
    #     Creates a grid of edges by distance offsets.

    #     Parameters
    #     ----------
    #     face : topologic_core.Face , optional
    #         The input face. If set to None, the grid is created on the XY plane. Default is None.
    #     uOrigin : topologic_core.Vertex , optional
    #         The origin used for the local grid coordinate system. If None, it is set to the face's
    #         (0,0) parameter vertex or to the world origin when face is None. Default is None.
    #     vOrigin : topologic_core.Vertex , optional
    #         Deprecated/legacy-compatible. If supplied, it is used as the origin for constant-v
    #         lines only. In most cases this should be None so that both u and v lines share one
    #         coordinate frame. Default is None.
    #     uRange : list , optional
    #         A list of distance offsets in the *u* direction. Default is [-0.5,-0.25,0,0.25,0.5].
    #     vRange : list , optional
    #         A list of distance offsets in the *v* direction. Default is [-0.5,-0.25,0,0.25,0.5].
    #     clip : bool , optional
    #         If True, the grid is clipped by the input face. Default is False.
    #     mantissa : int , optional
    #         The number of decimal places to round coordinates to. Default is 6.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.

    #     Returns
    #     -------
    #     topologic_core.Cluster
    #         The created grid. Edges in the grid have a dictionary with keys "dir" and "offset".
    #     """
    #     from topologicpy.Edge import Edge
    #     from topologicpy.Cluster import Cluster
    #     from topologicpy.Topology import Topology

    #     u_vals = Grid._FloatList(uRange, Grid._DefaultDistanceRange())
    #     v_vals = Grid._FloatList(vRange, Grid._DefaultDistanceRange())

    #     if u_vals is None or v_vals is None:
    #         return None

    #     if len(u_vals) < 1 and len(v_vals) < 1:
    #         return None

    #     u_dir, v_dir = Grid._Basis(face=face, mantissa=mantissa, tolerance=tolerance)
    #     if u_dir is None or v_dir is None:
    #         return None

    #     # Use one consistent local coordinate frame.
    #     # uOrigin is the primary origin. vOrigin is kept only for backward compatibility.
    #     origin = Grid._Origin(face=face, origin=uOrigin, mantissa=mantissa)
    #     if origin is None:
    #         return None

    #     origin_coords = Grid._Coordinates(origin, mantissa=mantissa)
    #     if origin_coords is None:
    #         return None

    #     if vOrigin is None:
    #         v_origin_coords = origin_coords
    #     else:
    #         v_origin = Grid._Origin(face=face, origin=vOrigin, mantissa=mantissa)
    #         if v_origin is None:
    #             return None
    #         v_origin_coords = Grid._Coordinates(v_origin, mantissa=mantissa)
    #         if v_origin_coords is None:
    #             return None

    #     u_span = Grid._Span(u_vals)
    #     v_span = Grid._Span(v_vals)

    #     if u_span is None or v_span is None:
    #         return None

    #     has_face = Grid._IsFace(face)
    #     grid_edges = []

    #     # Constant-u lines:
    #     # Offset along u_dir, then span along v_dir.
    #     for u in u_vals:
    #         start = Grid._Point(
    #             origin_coords,
    #             u_dir,
    #             v_dir,
    #             u,
    #             v_span[0],
    #             mantissa=mantissa
    #         )
    #         end = Grid._Point(
    #             origin_coords,
    #             u_dir,
    #             v_dir,
    #             u,
    #             v_span[1],
    #             mantissa=mantissa
    #         )

    #         edge = Edge.ByVertices([start, end], tolerance=tolerance)

    #         if edge and clip and has_face:
    #             edge = Topology.Intersect(edge, face)

    #         Grid._AppendEdgeResult(edge, grid_edges, "u", u)

    #     # Constant-v lines:
    #     # Offset along v_dir, then span along u_dir.
    #     # The same u_dir/v_dir basis is used. Only the origin may differ for legacy compatibility.
    #     for v in v_vals:
    #         start = Grid._Point(
    #             v_origin_coords,
    #             u_dir,
    #             v_dir,
    #             u_span[0],
    #             v,
    #             mantissa=mantissa
    #         )
    #         end = Grid._Point(
    #             v_origin_coords,
    #             u_dir,
    #             v_dir,
    #             u_span[1],
    #             v,
    #             mantissa=mantissa
    #         )

    #         edge = Edge.ByVertices([start, end], tolerance=tolerance)

    #         if edge and clip and has_face:
    #             edge = Topology.Intersect(edge, face)

    #         Grid._AppendEdgeResult(edge, grid_edges, "v", v)

    #     if len(grid_edges) < 1:
    #         return None

    #     return Cluster.ByTopologies(grid_edges)


    @staticmethod
    def EdgesByDistances(face=None,
                        uOrigin=None,
                        vOrigin=None,
                        uRange=None,
                        vRange=None,
                        clip=False,
                        mantissa: int = 6,
                        tolerance=0.0001):
        """
        Creates a grid of edges by distance offsets.

        This method converts physical distances along the face's local u and v
        parameter directions into normalized parameters in the range [0, 1], then
        calls Grid.EdgesByParameters.

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the input uRange and vRange are assumed
            to already be normalized parameters and Grid.EdgesByParameters is called
            directly. Default is None.
        uOrigin : topologic_core.Vertex , optional
            Kept for backward API compatibility. This implementation derives
            distances from Face.VertexByParameters(face, 0, 0). Default is None.
        vOrigin : topologic_core.Vertex , optional
            Kept for backward API compatibility. This implementation derives
            distances from Face.VertexByParameters(face, 0, 0). Default is None.
        uRange : list , optional
            A list of physical distances along the face's u direction. If None, five
            distances are generated between 0 and the full u-direction distance.
        vRange : list , optional
            A list of physical distances along the face's v direction. If None, five
            distances are generated between 0 and the full v-direction distance.
        clip : bool , optional
            Passed to Grid.EdgesByParameters. Default is False.
        mantissa : int , optional
            The number of decimal places to round parameters to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Edges in the grid have a dictionary with keys "dir" and
            "offset", as assigned by Grid.EdgesByParameters.
        """
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex

        def _float_list(values):
            if values is None:
                return None
            if not isinstance(values, list):
                values = [values]
            result = []
            for value in values:
                try:
                    result.append(float(value))
                except:
                    continue
            return result

        def _default_distances(total_distance):
            return [
                0.0,
                0.25 * total_distance,
                0.50 * total_distance,
                0.75 * total_distance,
                total_distance
            ]

        def _clamp(value, low=0.0, high=1.0):
            return max(low, min(high, value))

        def _safe_vertex_by_parameters(face, u, v):
            try:
                return Face.VertexByParameters(face, u, v)
            except:
                return None

        def _safe_distance(vertex_a, vertex_b):
            if vertex_a is None or vertex_b is None:
                return None
            try:
                return float(Vertex.Distance(vertex_a, vertex_b))
            except:
                return None

        def _axis_distance(face, axis, parameter):
            parameter = _clamp(parameter)

            origin = _safe_vertex_by_parameters(face, 0.0, 0.0)
            if origin is None:
                return None

            if axis == "u":
                vertex = _safe_vertex_by_parameters(face, parameter, 0.0)
            else:
                vertex = _safe_vertex_by_parameters(face, 0.0, parameter)

            return _safe_distance(origin, vertex)

        def _distance_to_parameter(face, axis, distance, total_distance):
            if total_distance is None or total_distance <= tolerance:
                return 0.0

            try:
                distance = float(distance)
            except:
                return None

            if distance <= tolerance:
                return 0.0

            if distance >= total_distance - tolerance:
                return 1.0

            low = 0.0
            high = 1.0

            # Binary search assumes that distance increases monotonically along
            # the selected face parameter direction.
            for _ in range(64):
                mid = 0.5 * (low + high)
                mid_distance = _axis_distance(face, axis, mid)

                if mid_distance is None:
                    return None

                if abs(mid_distance - distance) <= tolerance:
                    return round(mid, mantissa)

                if mid_distance < distance:
                    low = mid
                else:
                    high = mid

            return round(0.5 * (low + high), mantissa)

        # If no face is supplied, there is no physical face domain from which to
        # derive distances. In that case, assume the supplied ranges are already
        # normalized parameters.
        if face is None:
            u_params = _float_list(uRange)
            v_params = _float_list(vRange)

            if u_params is None:
                u_params = Grid._DefaultParameterRange()
            if v_params is None:
                v_params = Grid._DefaultParameterRange()

            return Grid.EdgesByParameters(
                face=face,
                uRange=u_params,
                vRange=v_params,
                clip=clip,
                mantissa=mantissa,
                tolerance=tolerance
            )

        origin = _safe_vertex_by_parameters(face, 0.0, 0.0)
        u_end = _safe_vertex_by_parameters(face, 1.0, 0.0)
        v_end = _safe_vertex_by_parameters(face, 0.0, 1.0)

        if origin is None or u_end is None or v_end is None:
            return None

        u_total_distance = _safe_distance(origin, u_end)
        v_total_distance = _safe_distance(origin, v_end)

        if u_total_distance is None or v_total_distance is None:
            return None

        if u_total_distance <= tolerance or v_total_distance <= tolerance:
            return None

        u_vals = _float_list(uRange)
        v_vals = _float_list(vRange)

        if u_vals is None:
            u_vals = _default_distances(u_total_distance)

        if v_vals is None:
            v_vals = _default_distances(v_total_distance)

        if len(u_vals) < 1 and len(v_vals) < 1:
            return None

        u_params = []
        for u_distance in u_vals:
            u_param = _distance_to_parameter(
                face=face,
                axis="u",
                distance=u_distance,
                total_distance=u_total_distance
            )
            if u_param is not None:
                u_params.append(u_param)

        v_params = []
        for v_distance in v_vals:
            v_param = _distance_to_parameter(
                face=face,
                axis="v",
                distance=v_distance,
                total_distance=v_total_distance
            )
            if v_param is not None:
                v_params.append(v_param)

        if len(u_params) < 1 and len(v_params) < 1:
            return None

        return Grid.EdgesByParameters(
            face=face,
            uRange=u_params,
            vRange=v_params,
            clip=clip,
            mantissa=mantissa,
            tolerance=tolerance
        )

    @staticmethod
    def EdgesByParameters(face,
                          uRange=None,
                          vRange=None,
                          clip=False,
                          mantissa=6,
                          tolerance=0.0001):
        """
        Creates a grid of edges by face parameters.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        uRange : list , optional
            A list of *u* parameters for the *u* grid lines. Default is [0,0.25,0.5,0.75,1.0].
        vRange : list , optional
            A list of *v* parameters for the *v* grid lines. Default is [0,0.25,0.5,0.75,1.0].
        clip : bool , optional
            If True, the grid is clipped by the input face. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Edges in the grid have a dictionary with keys "dir" and "offset".
        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Grid._IsFace(face):
            return None

        u_vals = Grid._ParameterList(uRange, Grid._DefaultParameterRange())
        v_vals = Grid._ParameterList(vRange, Grid._DefaultParameterRange())
        if u_vals is None or v_vals is None:
            return None
        if len(u_vals) < 1 and len(v_vals) < 1:
            return None

        grid_edges = []

        for u in u_vals:
            start = Face.VertexByParameters(face, u, 0)
            end = Face.VertexByParameters(face, u, 1)
            edge = Edge.ByVertices([start, end], tolerance=tolerance)
            if edge and clip:
                edge = Topology.Intersect(edge, face)
            Grid._AppendEdgeResult(edge, grid_edges, "u", u)

        for v in v_vals:
            start = Face.VertexByParameters(face, 0, v)
            end = Face.VertexByParameters(face, 1, v)
            edge = Edge.ByVertices([start, end], tolerance=tolerance)
            if edge and clip:
                edge = Topology.Intersect(edge, face)
            Grid._AppendEdgeResult(edge, grid_edges, "v", v)

        if len(grid_edges) < 1:
            return None
        return Cluster.ByTopologies(grid_edges)

    @staticmethod
    def VerticesByDistances_old(face=None,
                                origin=None,
                                uRange=None,
                                vRange=None,
                                clip: bool = False,
                                mantissa: int = 6,
                                tolerance: float = 0.0001):
        """
        Deprecated compatibility wrapper for VerticesByDistances.
        """
        return Grid.VerticesByDistances(face=face,
                                        origin=origin,
                                        uRange=uRange,
                                        vRange=vRange,
                                        clip=clip,
                                        mantissa=mantissa,
                                        tolerance=tolerance,
                                        silent=True)

    @staticmethod
    def VerticesByDistances(face=None,
                            origin=None,
                            uRange=None,
                            vRange=None,
                            clip: bool = False,
                            mantissa: int = 6,
                            tolerance: float = 0.0001,
                            silent: bool = False):
        """
        Creates a grid of vertices by distance offsets.

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid is created on the XY plane. Default is None.
        origin : topologic_core.Vertex , optional
            The origin of the grid vertices. If None, it is set to the face's (0,0) parameter vertex
            or to the world origin when face is None. Default is None.
        uRange : list , optional
            A list of distance offsets in the *u* direction. Default is [-0.5,-0.25,0,0.25,0.5].
        vRange : list , optional
            A list of distance offsets in the *v* direction. Default is [-0.5,-0.25,0,0.25,0.5].
        clip : bool , optional
            If True, vertices outside the input face are removed. Default is False.
        mantissa : int , optional
            The number of decimal places to round coordinates to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warning or error messages are printed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Vertices in the grid have a dictionary with keys "u" and "v".
        """
        from topologicpy.Cluster import Cluster

        u_vals = Grid._FloatList(uRange, Grid._DefaultDistanceRange())
        v_vals = Grid._FloatList(vRange, Grid._DefaultDistanceRange())
        if u_vals is None or v_vals is None or len(u_vals) < 1 or len(v_vals) < 1:
            if not silent:
                print("Grid.VerticesByDistances - Error: The input uRange or vRange parameter is not valid. Returning None.")
            return None

        u_dir, v_dir = Grid._Basis(face=face, mantissa=mantissa, tolerance=tolerance)
        if u_dir is None or v_dir is None:
            if not silent:
                print("Grid.VerticesByDistances - Error: Could not derive a valid grid basis. Returning None.")
            return None

        origin_vertex = Grid._Origin(face=face, origin=origin, mantissa=mantissa)
        origin_coords = Grid._Coordinates(origin_vertex, mantissa=mantissa)

        grid_vertices = []
        for u in u_vals:
            for v in v_vals:
                vertex = Grid._Point(origin_coords, u_dir, v_dir, u, v, mantissa=mantissa)
                Grid._SetDictionary(vertex, ["u", "v"], [u, v])
                grid_vertices.append(vertex)

        if clip and Grid._IsFace(face):
            grid_vertices = Grid._FilterVerticesByFace(grid_vertices, face, tolerance=tolerance)

        if len(grid_vertices) < 1:
            return None
        return Cluster.ByTopologies(grid_vertices)

    @staticmethod
    def VerticesByParameters(face=None,
                             uRange=None,
                             vRange=None,
                             clip=False,
                             tolerance=0.0001,
                             silent: bool = False):
        """
        Creates a grid of vertices by face parameters.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        uRange : list , optional
            A list of *u* parameters. Default is [0,0.25,0.5,0.75,1.0].
        vRange : list , optional
            A list of *v* parameters. Default is [0,0.25,0.5,0.75,1.0].
        clip : bool , optional
            If True, vertices outside the input face are removed. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warning or error messages are printed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Vertices in the grid have a dictionary with keys "u" and "v".
        """
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster

        if not Grid._IsFace(face):
            if not silent:
                print("Grid.VerticesByParameters - Error: The input face parameter is not a valid face. Returning None.")
            return None

        u_vals = Grid._ParameterList(uRange, Grid._DefaultParameterRange())
        v_vals = Grid._ParameterList(vRange, Grid._DefaultParameterRange())
        if u_vals is None or v_vals is None or len(u_vals) < 1 or len(v_vals) < 1:
            if not silent:
                print("Grid.VerticesByParameters - Error: The input uRange or vRange parameter is not valid. Returning None.")
            return None

        grid_vertices = []
        for u in u_vals:
            for v in v_vals:
                vertex = Face.VertexByParameters(face, u, v)
                Grid._SetDictionary(vertex, ["u", "v"], [u, v])
                grid_vertices.append(vertex)

        if clip:
            grid_vertices = Grid._FilterVerticesByFace(grid_vertices, face, tolerance=tolerance)

        if len(grid_vertices) < 1:
            return None
        return Cluster.ByTopologies(grid_vertices)
