# Copyright (C) 2024
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
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        uOrigin : topologic_core.Vertex , optional
            The origin of the *u* grid lines. If set to None: if the face is set, the uOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the uOrigin will be set to the origin. The default is None.
        vOrigin : topologic_core.Vertex , optional
            The origin of the *v* grid lines. If set to None: if the face is set, the vOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the vOrigin will be set to the origin. The default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            A list of *u* parameters for the *u* grid lines. The default is [0,0.25,0.5, 0.75, 1.0].
        vRange : list , optional
            A list of *v* parameters for the *v* grid lines. The default is [0,0.25,0.5, 0.75, 1.0].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
    def VerticesByDistances(face=None,
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
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        origin : topologic_core.Vertex , optional
            The origin of the grid vertices. If set to None: if the face is set, the origin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the origin will be set to (0, 0, 0). The default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
                    if clip and Topology.IsInstance(face, "Face"):
                        gridVertex = gridVertex.Intersect(face, False)
                    if Topology.IsInstance(gridVertex, "Vertex"):
                        d = Dictionary.ByKeysValues(["u","v"],[u,v])
                        if d:
                            gridVertex.SetDictionary(d)
                        gridVertices.append(gridVertex)
        grid = None
        if len(gridVertices) > 0:
            grid = Cluster.ByTopologies(gridVertices)
        return grid

    @staticmethod
    def VerticesByParameters(face=None, uRange=[0.0,0.25,0.5,0.75,1.0], vRange=[0.0,0.25,0.5,0.75,1.0], clip=False, tolerance=0.0001):
        """
        Creates a grid (cluster of vertices).

        Parameters
        ----------
        face : topologic_core.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        uRange : list , optional
            A list of *u* parameters for the *u* grid lines from the uOrigin. The default is [0.0,0.25,0.5,0.75,1.0].
        vRange : list , optional
            A list of *v* parameters for the *v* grid lines from the vOrigin. The default is [0.0,0.25,0.5,0.75,1.0].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            return None
        if len(uRange) < 1 or len(vRange) < 1:
            return None
        if (min(uRange) < 0) or (max(uRange) > 1):
            return None
        if (min(vRange) < 0) or (max(vRange) > 1):
            return None

        uRange.sort()
        vRange.sort()
        gridVertices = []
        if len(uRange) > 0:
            uRange.sort()
            for u in uRange:
                for v in vRange:
                    gridVertex = Face.VertexByParameters(face, u, v)
                    if clip and Topology.IsInstance(face, "Face"):
                        gridVertex = gridVertex.Intersect(face, False)
                    if Topology.IsInstance(gridVertex, "Vertex"):
                        d = Dictionary.ByKeysValues(["u","v"],[u,v])
                        if d:
                            gridVertex.SetDictionary(d)
                        gridVertices.append(gridVertex)
        grid = None
        if len(gridVertices) > 0:
            grid = Cluster.ByTopologies(gridVertices)
        return grid