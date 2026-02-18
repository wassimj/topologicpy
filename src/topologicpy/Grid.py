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

import topologic_core as topologic

class Grid():
    @staticmethod
    def EdgesByDistances(face=None, uOrigin=None, vOrigin=None, uRange=[-0.5,-0.25,0, 0.25,0.5], vRange=[-0.5,-0.25,0, 0.25,0.5], clip=False, mantissa: int = 6, tolerance=0.0001):
        """
        Creates a grid (cluster of edges).

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. Default is None.
        uOrigin : topologic_core.Vertex , optional
            The origin of the *u* grid lines. If set to None: if the face is set, the uOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the uOrigin will be set to the origin. Default is None.
        vOrigin : topologic_core.Vertex , optional
            The origin of the *v* grid lines. If set to None: if the face is set, the vOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the vOrigin will be set to the origin. Default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Edges in the grid have an identifying dictionary with two keys: "dir" and "offset". The "dir" key can have one of two values: "u" or "v", the "offset" key contains the offset distance of that grid edge from the specified origin.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        if len(uRange) < 1 or len(vRange) < 1:
            return None
        if not uOrigin:
            if not Topology.IsInstance(face, "Face"):
                uOrigin = Vertex.ByCoordinates(0, 0, 0)
            else:
                uOrigin = Face.VertexByParameters(face, 0, 0)
        if not vOrigin:
            if not Topology.IsInstance(face, "Face"):
                vOrigin = Vertex.ByCoordinates(0, 0, 0)
            else:
                vOrigin = Face.VertexByParameters(face, 0, 0)
        
        if Topology.IsInstance(face, "Face"):
            v1 = Face.VertexByParameters(face, 0, 0)
            v2 = Face.VertexByParameters(face, 1, 0)
            v3 = Face.VertexByParameters(face, 0, 0)
            v4 = Face.VertexByParameters(face, 0, 1)
        else:
            v1 = Vertex.ByCoordinates(0, 0, 0)
            v2 = Vertex.ByCoordinates(max(uRange),0,0)
            v3 = Vertex.ByCoordinates(0, 0, 0)
            v4 = Vertex.ByCoordinates(0,max(vRange),0)

        uVector = [Vertex.X(v2, mantissa=mantissa)-Vertex.X(v1, mantissa=mantissa), Vertex.Y(v2, mantissa=mantissa)-Vertex.Y(v1, mantissa=mantissa),Vertex.Z(v2, mantissa=mantissa)-Vertex.Z(v1, mantissa=mantissa)]
        vVector = [Vertex.X(v4, mantissa=mantissa)-Vertex.X(v3, mantissa=mantissa), Vertex.Y(v4, mantissa=mantissa)-Vertex.Y(v3, mantissa=mantissa),Vertex.Z(v4, mantissa=mantissa)-Vertex.Z(v3, mantissa=mantissa)]
        gridEdges = []
        if len(uRange) > 0:
            uRange.sort()
            uuVector = Vector.Normalize(uVector)
            for u in uRange:
                tempVec = Vector.Multiply(uuVector, u, tolerance=tolerance)
                v1 = Vertex.ByCoordinates(Vertex.X(uOrigin, mantissa=mantissa)+tempVec[0], Vertex.Y(uOrigin, mantissa=mantissa)+tempVec[1], Vertex.Z(uOrigin, mantissa=mantissa)+tempVec[2])
                v2 = Vertex.ByCoordinates(Vertex.X(v1, mantissa=mantissa)+vVector[0], Vertex.Y(v1, mantissa=mantissa)+vVector[1], Vertex.Z(v1, mantissa=mantissa)+vVector[2])
                e = Edge.ByVertices([v1, v2], tolerance=tolerance)
                if clip and Topology.IsInstance(face, "Face"):
                    e = e.Intersect(face, False)
                if e:
                    if Topology.IsInstance(e, "Edge"):
                        d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                        e.SetDictionary(d)
                        gridEdges.append(e)
                    elif Topology.Type(e) > Topology.TypeID("Edge"):
                        tempEdges = Topology.Edges(e)
                        for tempEdge in tempEdges:
                            d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                            tempEdge.SetDictionary(d)
                            gridEdges.append(tempEdge)
        if len(vRange) > 0:
            vRange.sort()
            uvVector = Vector.Normalize(vVector)
            for v in vRange:
                tempVec = Vector.Multiply(uvVector, v, tolerance=tolerance)
                v1 = Vertex.ByCoordinates(Vertex.X(vOrigin, mantissa=mantissa)+tempVec[0], Vertex.Y(vOrigin, mantissa=mantissa)+tempVec[1], Vertex.Z(vOrigin, mantissa=mantissa)+tempVec[2])
                v2 = Vertex.ByCoordinates(Vertex.X(v1, mantissa=mantissa)+uVector[0], Vertex.Y(v1, mantissa=mantissa)+uVector[1], Vertex.Z(v1, mantissa=mantissa)+uVector[2])
                e = Edge.ByVertices([v1, v2], tolerance=tolerance)
                if clip and Topology.IsInstance(face, "Face"):
                    e = e.Intersect(face, False)
                if e:
                    if Topology.IsInstance(e, "Edge"):
                        d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                        e.SetDictionary(d)
                        gridEdges.append(e)
                    elif Topology.Type(e) > Topology.TypeID("Edge"):
                        tempEdges = Topology.Edges(e)
                        for tempEdge in tempEdges:
                            d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                            tempEdge.SetDictionary(d)
                            gridEdges.append(tempEdge)
        grid = None
        if len(gridEdges) > 0:
            grid = Cluster.ByTopologies(gridEdges)
        return grid
    
    @staticmethod
    def EdgesByParameters(face, uRange=[0,0.25,0.5,0.75,1.0], vRange=[0,0.25,0.5,0.75,1.0], clip=False, tolerance=0.0001):
        """
        Creates a grid (cluster of edges).

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        uRange : list , optional
            A list of *u* parameters for the *u* grid lines. Default is [0,0.25,0.5, 0.75, 1.0].
        vRange : list , optional
            A list of *v* parameters for the *v* grid lines. Default is [0,0.25,0.5, 0.75, 1.0].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Cluster
            The created grid. Edges in the grid have an identifying dictionary with two keys: "dir" and "offset". The "dir" key can have one of two values: "u" or "v", the "offset" key contains the offset parameter of that grid edge.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        if len(uRange) < 1 and len(vRange) < 1:
            return None
        if len(uRange) > 0:
            if (min(uRange) < 0) or (max(uRange) > 1):
                return None
        if len(vRange) > 0:
            if (min(vRange) < 0) or (max(vRange) > 1):
                return None

        uRange.sort()
        vRange.sort()
        gridEdges = []
        for u in uRange:
            v1 = Face.VertexByParameters(face, u, 0)
            v2 = Face.VertexByParameters(face, u, 1)
            e = Edge.ByVertices([v1, v2], tolerance=tolerance)
            if clip and Topology.IsInstance(face, "Face"):
                e = e.Intersect(face, False)
            if e:
                if Topology.IsInstance(e, "Edge"):
                    d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                    e.SetDictionary(d)
                    gridEdges.append(e)
                elif Topology.Type(e) > Topology.TypeID("Edge"):
                    tempEdges = Topology.Edges(e)
                    for tempEdge in tempEdges:
                        d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                        tempEdge.SetDictionary(d)
                        gridEdges.append(tempEdge)
        for v in vRange:
            v1 = Face.VertexByParameters(face, 0, v)
            v2 = Face.VertexByParameters(face, 1, v)
            e = Edge.ByVertices([v1, v2], tolerance=tolerance)
            if clip and Topology.IsInstance(face, "Face"):
                e = e.Intersect(face, False)
            if e:
                if Topology.IsInstance(e, "Edge"):
                    d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                    e.SetDictionary(d)
                    gridEdges.append(e)
                elif Topology.Type(e) > Topology.TypeID("Edge"):
                    tempEdges = Topology.Edges(e)
                    for tempEdge in tempEdges:
                        d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                        tempEdge = Topology.SetDictionary(tempEdge, d)
                        gridEdges.append(tempEdge)
        grid = None
        if len(gridEdges) > 0:
            grid = Cluster.ByTopologies(gridEdges)
        return grid

    @staticmethod
    def VerticesByDistances_old(face=None,
                            origin=None,
                            uRange: list = [-0.5,-0.25,0, 0.25,0.5],
                            vRange: list = [-0.5,-0.25,0,0.25,0.5],
                            clip: bool = False,
                            mantissa: int = 6,
                            tolerance: float = 0.0001):
        """
        Creates a grid (cluster of vertices).

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. Default is None.
        origin : topologic_core.Vertex , optional
            The origin of the grid vertices. If set to None: if the face is set, the origin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the origin will be set to (0, 0, 0). Default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Vertices in the grid have an identifying dictionary with two keys: "u" and "v". The "dir" key can have one of two values: "u" or "v" that contain the *u* and *v* offset distances of that grid vertex from the specified origin.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        if len(uRange) < 1 or len(vRange) < 1:
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            if not Topology.IsInstance(face, "Face"):
                origin = Vertex.ByCoordinates(0, 0, 0)
            else:
                origin = Face.VertexByParameters(face, 0, 0)
        
        print("ORIGIN:", Vertex.Coordinates(origin))
        
        if Topology.IsInstance(face, "Face"):
            v2 = Face.VertexByParameters(face, 1, 0)
            v3 = Face.VertexByParameters(face, 0, 0)
            v4 = Face.VertexByParameters(face, 0, 1)
        else:
            v2 = Vertex.ByCoordinates(max(uRange),0,0)
            v3 = Vertex.ByCoordinates(0, 0, 0)
            v4 = Vertex.ByCoordinates(0,max(vRange),0)

        uVector = [Vertex.X(v2, mantissa=mantissa)-Vertex.X(origin, mantissa=mantissa), Vertex.Y(v2, mantissa=mantissa)-Vertex.Y(origin, mantissa=mantissa),Vertex.Z(v2, mantissa=mantissa)-Vertex.Z(origin, mantissa=mantissa)]
        vVector = [Vertex.X(v4, mantissa=mantissa)-Vertex.X(v3, mantissa=mantissa), Vertex.Y(v4, mantissa=mantissa)-Vertex.Y(v3, mantissa=mantissa),Vertex.Z(v4, mantissa=mantissa)-Vertex.Z(v3, mantissa=mantissa)]
        gridVertices = []
        if len(uRange) > 0:
            uRange.sort()
            uuVector = Vector.Normalize(uVector)
            uvVector = Vector.Normalize(vVector)
            for u in uRange:
                for v in vRange:
                    uTempVec = Vector.Multiply(uuVector, u, tolerance=tolerance)
                    vTempVec = Vector.Multiply(uvVector, v, tolerance=tolerance)
                    gridVertex = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa)+uTempVec[0], Vertex.Y(origin, mantissa=mantissa)+vTempVec[1], Vertex.Z(origin, mantissa=mantissa)+uTempVec[2])
                    d = Dictionary.ByKeysValues(["u","v"],[u,v])
                    if d:
                        gridVertex.SetDictionary(d)
                    if clip and Topology.IsInstance(face, "Face"):
                        if Vertex.IsInternal(gridVertex, face):
                                gridVertices.append(gridVertex)
                    else:
                        gridVertices.append(gridVertex)

        grid = None
        if len(gridVertices) > 0:
            grid = Cluster.ByTopologies(gridVertices)
        return grid


    @staticmethod
    def VerticesByDistances(face=None,
                            origin=None,
                            uRange: list = [-0.5, -0.25, 0, 0.25, 0.5],
                            vRange: list = [-0.5, -0.25, 0, 0.25, 0.5],
                            clip: bool = False,
                            mantissa: int = 6,
                            tolerance: float = 0.0001,
                            silent: bool = False):
        """
        Creates a grid (cluster of vertices).

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. Default is None.
        origin : topologic_core.Vertex , optional
            The origin of the grid vertices. If set to None: if the face is set, the origin will be set to
            vertex at the face's 0,0 parameter. If the face is set to None, the origin will be set to (0, 0, 0).
            Default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. Default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warning or error messages are printed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The created grid (Cluster of Vertices). Vertices in the grid have an identifying dictionary
            with two keys: "u" and "v" that contain the *u* and *v* offset distances of that grid vertex
            from the specified origin.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector
        import math

        # -----------------------------
        # Helpers
        # -----------------------------
        def _round(x: float) -> float:
            return round(float(x), int(mantissa))

        def _vec_mag(v):
            return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

        def _safe_normalize(v):
            m = _vec_mag(v)
            if m <= max(tolerance, 1e-12):
                return None
            return [v[0]/m, v[1]/m, v[2]/m]

        # -----------------------------
        # Validate ranges
        # -----------------------------
        if not isinstance(uRange, list) or not isinstance(vRange, list):
            return None
        if len(uRange) < 1 or len(vRange) < 1:
            return None

        # Do NOT mutate caller's lists
        u_vals = sorted([float(x) for x in uRange])
        v_vals = sorted([float(x) for x in vRange])

        has_face = Topology.IsInstance(face, "Face")

        # -----------------------------
        # Resolve origin
        # -----------------------------
        if not Topology.IsInstance(origin, "Vertex"):
            if has_face:
                origin = Face.VertexByParameters(face, 0, 0)
            else:
                origin = Vertex.ByCoordinates(0, 0, 0)

        # For faces: build a consistent local basis relative to the param point of the origin
        # If we can't recover parameters at origin, fall back to (0,0).
        if has_face:
            u0 = 0.0
            v0 = 0.0
            try:
                # If available in your Face API, this keeps the basis local to the provided origin.
                uv = Face.ParametersAtVertex(face, origin)  # expected (u, v)
                if isinstance(uv, (list, tuple)) and len(uv) >= 2:
                    u0 = float(uv[0])
                    v0 = float(uv[1])
            except Exception:
                # fallback: treat origin as (0,0) param reference
                u0, v0 = 0.0, 0.0

            # Use the face param point as the geometric reference for basis construction
            ref = Face.VertexByParameters(face, u0, v0)

            # Basis directions from the SAME reference point
            vU = Face.VertexByParameters(face, u0 + 1.0, v0)
            vV = Face.VertexByParameters(face, u0, v0 + 1.0)

            uVector = [
                Vertex.X(vU, mantissa=mantissa) - Vertex.X(ref, mantissa=mantissa),
                Vertex.Y(vU, mantissa=mantissa) - Vertex.Y(ref, mantissa=mantissa),
                Vertex.Z(vU, mantissa=mantissa) - Vertex.Z(ref, mantissa=mantissa)
            ]
            vVector = [
                Vertex.X(vV, mantissa=mantissa) - Vertex.X(ref, mantissa=mantissa),
                Vertex.Y(vV, mantissa=mantissa) - Vertex.Y(ref, mantissa=mantissa),
                Vertex.Z(vV, mantissa=mantissa) - Vertex.Z(ref, mantissa=mantissa)
            ]

            # Use the provided origin as the translation anchor (but keep basis local to ref)
            ox = Vertex.X(origin, mantissa=mantissa)
            oy = Vertex.Y(origin, mantissa=mantissa)
            oz = Vertex.Z(origin, mantissa=mantissa)
        else:
            # XY plane basis
            uVector = [1.0, 0.0, 0.0]
            vVector = [0.0, 1.0, 0.0]
            ox = Vertex.X(origin, mantissa=mantissa)
            oy = Vertex.Y(origin, mantissa=mantissa)
            oz = Vertex.Z(origin, mantissa=mantissa)

        uu = _safe_normalize(uVector)
        vv = _safe_normalize(vVector)
        if uu is None or vv is None:
            return None

        # -----------------------------
        # Build grid vertices
        # -----------------------------
        gridVertices = []
        for u in u_vals:
            for v in v_vals:
                uTempVec = Vector.Multiply(uu, u, tolerance=tolerance)
                vTempVec = Vector.Multiply(vv, v, tolerance=tolerance)

                # Correct: origin + u*û + v*ṽ (component-wise)
                x = _round(ox + uTempVec[0] + vTempVec[0])
                y = _round(oy + uTempVec[1] + vTempVec[1])
                z = _round(oz + uTempVec[2] + vTempVec[2])

                gv = Vertex.ByCoordinates(x, y, z)

                d = Dictionary.ByKeysValues(["u", "v"], [u, v])
                if d:
                    # Prefer TopologicPy's consistent pattern
                    Topology.SetDictionary(gv, d)

                gridVertices.append(gv)
        
        if clip and has_face:
            from topologicpy.Vector import Vector
            cluster = Cluster.ByTopologies(gridVertices)
            centroid = Topology.Centroid(face)
            x_tran = -Vertex.X(centroid)
            y_tran = -Vertex.Y(centroid)
            z_tran = -Vertex.Z(centroid)
            face_2 = Topology.Translate(face, x_tran, y_tran, z_tran)
            cluster_2 = Topology.Translate(cluster, x_tran, y_tran, z_tran)

            face_normal = Face.Normal(face_2)
            up = [0,0,1]
            tran_mat = Vector.TransformationMatrix(face_normal, up)
            flat_face = Topology.Transform(face_2, tran_mat, transferDictionaries=False)
            flat_cluster = Topology.Transform(cluster_2, tran_mat)
            # flat_cluster = Topology.Translate(flat_cluster, 0, 0, -Vertex.Z(flat_vertex))
            status_list = Vertex.IsInternal2D(Topology.Vertices(flat_cluster), flat_face)
            gridVertices = [v for i, v in enumerate(gridVertices) if status_list[i] == True]

        if len(gridVertices) < 1:
            return None

        return Cluster.ByTopologies(gridVertices)

    @staticmethod
    def VerticesByParameters(face=None, uRange=[0.0,0.25,0.5,0.75,1.0], vRange=[0.0,0.25,0.5,0.75,1.0], clip=False, tolerance=0.0001, silent: bool = False):
        """
        Creates a grid (cluster of vertices).

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. Default is None.
        uRange : list , optional
            A list of *u* parameters for the *u* grid lines from the uOrigin. Default is [0.0,0.25,0.5,0.75,1.0].
        vRange : list , optional
            A list of *v* parameters for the *v* grid lines from the vOrigin. Default is [0.0,0.25,0.5,0.75,1.0].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, no warning or error messages are printed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The created grid. Vertices in the grid have an identifying dictionary with two keys: "u" and "v". The "dir" key can have one of two values: "u" or "v" that contain the *u* and *v* offset distances of that grid vertex from the specified origin.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Grid.VerticesByParameters - Error: The input face parameter is not a valid face. Returning None.")
            return None
        if len(uRange) < 1 or len(vRange) < 1:
            if not silent:
                print("Grid.VerticesByParameters - Error: The input uRange or VRange parameter is not valid. Returning None.")
            return None
        if (min(uRange) < 0) or (max(uRange) > 1):
            if not silent:
                print("Grid.VerticesByParameters - Error: The input uRange or VRange parameter is not valid. Returning None.")
            return None
        if (min(vRange) < 0) or (max(vRange) > 1):
            if not silent:
                print("Grid.VerticesByParameters - Error: The input uRange or VRange parameter is not valid. Returning None.")
            return None

        uRange.sort()
        vRange.sort()
        gridVertices = []
        if len(uRange) > 0:
            uRange.sort()
            for u in uRange:
                for v in vRange:
                    gridVertex = Face.VertexByParameters(face, u, v)
                    d = Dictionary.ByKeysValues(["u","v"],[u,v])
                    gridVertex.SetDictionary(d)
                    gridVertices.append(gridVertex)
        if clip:
            from topologicpy.Vector import Vector
            cluster = Cluster.ByTopologies(gridVertices)
            centroid = Topology.Centroid(face)
            x_tran = -Vertex.X(centroid)
            y_tran = -Vertex.Y(centroid)
            z_tran = -Vertex.Z(centroid)
            face_2 = Topology.Translate(face, x_tran, y_tran, z_tran)
            cluster_2 = Topology.Translate(cluster, x_tran, y_tran, z_tran)

            face_normal = Face.Normal(face_2)
            up = [0,0,1]
            tran_mat = Vector.TransformationMatrix(face_normal, up)
            flat_face = Topology.Transform(face_2, tran_mat, transferDictionaries=False)
            flat_cluster = Topology.Transform(cluster_2, tran_mat)
            status_list = Vertex.IsInternal2D(Topology.Vertices(flat_cluster), flat_face)
            gridVertices = [v for i, v in enumerate(gridVertices) if status_list[i] == True]
        
        if len(gridVertices) < 1:
            return None

        return Cluster.ByTopologies(gridVertices)