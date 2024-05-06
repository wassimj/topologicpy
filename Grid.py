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

class Grid(topologic.Cluster):
    @staticmethod
    def EdgesByDistances(face=None, uOrigin=None, vOrigin=None, uRange=[-0.5,-0.25,0, 0.25,0.5], vRange=[-0.5,-0.25,0, 0.25,0.5], clip=False, tolerance=0.0001):
        """
        Creates a grid (cluster of edges).

        Parameters
        ----------
        face : topologic.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        uOrigin : topologic.Vertex , optional
            The origin of the *u* grid lines. If set to None: if the face is set, the uOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the uOrigin will be set to the origin. The default is None.
        vOrigin : topologic.Vertex , optional
            The origin of the *v* grid lines. If set to None: if the face is set, the vOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the vOrigin will be set to the origin. The default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cluster
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
            if not isinstance(face, topologic.Face):
                uOrigin = Vertex.ByCoordinates(0, 0, 0)
            else:
                uOrigin = Face.VertexByParameters(face, 0, 0)
        if not vOrigin:
            if not isinstance(face, topologic.Face):
                vOrigin = Vertex.ByCoordinates(0, 0, 0)
            else:
                vOrigin = Face.VertexByParameters(face, 0, 0)
        
        if isinstance(face, topologic.Face):
            v1 = Face.VertexByParameters(face, 0, 0)
            v2 = Face.VertexByParameters(face, 1, 0)
            v3 = Face.VertexByParameters(face, 0, 0)
            v4 = Face.VertexByParameters(face, 0, 1)
        else:
            v1 = Vertex.ByCoordinates(0, 0, 0)
            v2 = Vertex.ByCoordinates(max(uRange),0,0)
            v3 = Vertex.ByCoordinates(0, 0, 0)
            v4 = Vertex.ByCoordinates(0,max(vRange),0)

        uVector = [v2.X()-v1.X(), v2.Y()-v1.Y(),v2.Z()-v1.Z()]
        vVector = [v4.X()-v3.X(), v4.Y()-v3.Y(),v4.Z()-v3.Z()]
        gridEdges = []
        if len(uRange) > 0:
            uRange.sort()
            uuVector = Vector.Normalize(uVector)
            for u in uRange:
                tempVec = Vector.Multiply(uuVector, u, tolerance)
                v1 = Vertex.ByCoordinates(uOrigin.X()+tempVec[0], uOrigin.Y()+tempVec[1], uOrigin.Z()+tempVec[2])
                v2 = Vertex.ByCoordinates(v1.X()+vVector[0], v1.Y()+vVector[1], v1.Z()+vVector[2])
                e = Edge.ByVertices([v1, v2], tolerance=tolerance)
                if clip and isinstance(face, topologic.Face):
                    e = e.Intersect(face, False)
                if e:
                    if isinstance(e, topologic.Edge):
                        d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                        e.SetDictionary(d)
                        gridEdges.append(e)
                    elif e.Type() > topologic.Edge.Type():
                        tempEdges = []
                        _ = e.Edges(None, tempEdges)
                        for tempEdge in tempEdges:
                            d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                            tempEdge.SetDictionary(d)
                            gridEdges.append(tempEdge)
        if len(vRange) > 0:
            vRange.sort()
            uvVector = Vector.Normalize(vVector)
            for v in vRange:
                tempVec = Vector.Multiply(uvVector, v, tolerance)
                v1 = Vertex.ByCoordinates(vOrigin.X()+tempVec[0], vOrigin.Y()+tempVec[1], vOrigin.Z()+tempVec[2])
                v2 = Vertex.ByCoordinates(v1.X()+uVector[0], v1.Y()+uVector[1], v1.Z()+uVector[2])
                e = Edge.ByVertices([v1, v2], tolerance=tolerance)
                if clip and isinstance(face, topologic.Face):
                    e = e.Intersect(face, False)
                if e:
                    if isinstance(e, topologic.Edge):
                        d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                        e.SetDictionary(d)
                        gridEdges.append(e)
                    elif e.Type() > topologic.Edge.Type():
                        tempEdges = []
                        _ = e.Edges(None, tempEdges)
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
        face : topologic.Face
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
        topologic.Cluster
            The created grid. Edges in the grid have an identifying dictionary with two keys: "dir" and "offset". The "dir" key can have one of two values: "u" or "v", the "offset" key contains the offset parameter of that grid edge.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

        if not isinstance(face, topologic.Face):
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
            if clip and isinstance(face, topologic.Face):
                e = e.Intersect(face, False)
            if e:
                if isinstance(e, topologic.Edge):
                    d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                    e.SetDictionary(d)
                    gridEdges.append(e)
                elif e.Type() > topologic.Edge.Type():
                    tempEdges = []
                    _ = e.Edges(None, tempEdges)
                    for tempEdge in tempEdges:
                        d = Dictionary.ByKeysValues(["dir", "offset"],["u",u])
                        tempEdge.SetDictionary(d)
                        gridEdges.append(tempEdge)
        for v in vRange:
            v1 = Face.VertexByParameters(face, 0, v)
            v2 = Face.VertexByParameters(face, 1, v)
            e = Edge.ByVertices([v1, v2], tolerance=tolerance)
            if clip and isinstance(face, topologic.Face):
                e = e.Intersect(face, False)
            if e:
                if isinstance(e, topologic.Edge):
                    d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                    e.SetDictionary(d)
                    gridEdges.append(e)
                elif e.Type() > topologic.Edge.Type():
                    tempEdges = []
                    _ = e.Edges(None, tempEdges)
                    for tempEdge in tempEdges:
                        d = Dictionary.ByKeysValues(["dir", "offset"],["v",v])
                        tempEdge.SetDictionary(d)
                        gridEdges.append(tempEdge)
        grid = None
        if len(gridEdges) > 0:
            grid = Cluster.ByTopologies(gridEdges)
        return grid


    @staticmethod
    def VerticesByDistances(face=None, origin=None, uRange=[-0.5,-0.25,0, 0.25,0.5], vRange=[-0.5,-0.25,0,0.25,0.5], clip=False, tolerance=0.0001):
        """
        Creates a grid (cluster of vertices).

        Parameters
        ----------
        face : topologic.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        origin : topologic.Vertex , optional
            The origin of the grid vertices. If set to None: if the face is set, the origin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the origin will be set to (0, 0, 0). The default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        vRange : list , optional
            A list of distances for the *v* grid lines from the vOrigin. The default is [-0.5,-0.25,0, 0.25,0.5].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cluster
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
        if not origin:
            if not isinstance(face, topologic.Face):
                origin = Vertex.ByCoordinates(0, 0, 0)
            else:
                origin = Face.VertexByParameters(face, 0, 0)
        
        if isinstance(face, topologic.Face):
            v1 = Face.VertexByParameters(face, 0, 0)
            v2 = Face.VertexByParameters(face, 1, 0)
            v3 = Face.VertexByParameters(face, 0, 0)
            v4 = Face.VertexByParameters(face, 0, 1)
        else:
            v1 = Vertex.ByCoordinates(0, 0, 0)
            v2 = Vertex.ByCoordinates(max(uRange),0,0)
            v3 = Vertex.ByCoordinates(0, 0, 0)
            v4 = Vertex.ByCoordinates(0,max(vRange),0)

        uVector = [v2.X()-v1.X(), v2.Y()-v1.Y(),v2.Z()-v1.Z()]
        vVector = [v4.X()-v3.X(), v4.Y()-v3.Y(),v4.Z()-v3.Z()]
        gridVertices = []
        if len(uRange) > 0:
            uRange.sort()
            uuVector = Vector.Normalize(uVector)
            uvVector = Vector.Normalize(vVector)
            for u in uRange:
                for v in vRange:
                    uTempVec = Vector.Multiply(uuVector, u, tolerance)
                    vTempVec = Vector.Multiply(uvVector, v, tolerance)
                    gridVertex = Vertex.ByCoordinates(origin.X()+uTempVec[0], origin.Y()+vTempVec[1], origin.Z()+uTempVec[2])
                    if clip and isinstance(face, topologic.Face):
                        gridVertex = gridVertex.Intersect(face, False)
                    if isinstance(gridVertex, topologic.Vertex):
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
        face : topologic.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        origin : topologic.Vertex , optional
            The origin of the grid vertices. If set to None: if the face is set, the origin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the origin will be set to (0, 0, 0). The default is None.
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
        topologic.Cluster
            The created grid. Vertices in the grid have an identifying dictionary with two keys: "u" and "v". The "dir" key can have one of two values: "u" or "v" that contain the *u* and *v* offset distances of that grid vertex from the specified origin.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector

        if not isinstance(face, topologic.Face):
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
                    if clip and isinstance(face, topologic.Face):
                        gridVertex = gridVertex.Intersect(face, False)
                    if isinstance(gridVertex, topologic.Vertex):
                        d = Dictionary.ByKeysValues(["u","v"],[u,v])
                        if d:
                            gridVertex.SetDictionary(d)
                        gridVertices.append(gridVertex)
        grid = None
        if len(gridVertices) > 0:
            grid = Cluster.ByTopologies(gridVertices)
        return grid