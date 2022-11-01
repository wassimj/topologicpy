import topologicpy
import topologic


class Grid(topologic.Cluster):
    @staticmethod
    def ByDistances(face=None, uOrigin=None, vOrigin=None, uRange=[0,0.25,0.5,0.75,1.0], vRange=[0,0.25,0.5,0.75,1.0], clip=False, tolerance=0.0001):
        """
        Description
        ----------
        Creates a grid (cluster of edges) on the input face.

        Parameters
        ----------
        face : topologic.Face , optional
            The input face. If set to None, the grid will be created on the XY plane. The default is None.
        uOrigin : topologic.Vertex , optional
            The origin of the *u* grid lines. If set to None: if the face is set, the vOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the uOrigin will be set to the origin. The default is None.
        vOrigin : topologic.Vertex , optional
            The origin of the *v* grid lines. If set to None: if the face is set, the vOrigin will be set to vertex at the face's 0,0 paratmer. If the face is set to None, the uOrigin will be set to the origin. The default is None.
        uRange : list , optional
            A list of distances for the *u* grid lines from the uOrigin. The default is [0,0.25,0.5, 0.75, 1.0].
        vRange : list , optional
            A list of distances for the *v* grid lines from the uOrigin. The default is [0,0.25,0.5, 0.75, 1.0].
        clip : bool , optional
            If True the grid will be clipped by the shape of the input face. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Cluster
            The created grid. Edges in the grid have a dicitonary with an identifying dictionary with two keys: "dir" and "offset". The "dir" key can have one of two values: "u" or "v", the offset key contains the offset distance of that grid edge from the specified origin.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector
        if len(uRange) < 1 or len(vRange) < 1:
            return None
        if not uOrigin:
            if not isinstance(face, topologic.Face):
                uOrigin = Vertex.ByCoordinates(0,0,0)
            else:
                uOrigin = Face.VertexByParameters(face, 0, 0)
        if not vOrigin:
            if not isinstance(face, topologic.Face):
                vOrigin = Vertex.ByCoordinates(0,0,0)
            else:
                vOrigin = Face.VertexByParameters(face, 0, 0)
        
        if isinstance(face, topologic.Face):
            v1 = Face.VertexByParameters(face, 0, 0)
            v2 = Face.VertexByParameters(face, 1, 0)
            v3 = Face.VertexByParameters(face, 0, 0)
            v4 = Face.VertexByParameters(face, 0, 1)
        else:
            v1 = Vertex.ByCoordinates(0,0,0)
            v2 = Vertex.ByCoordinates(max(uRange),0,0)
            v3 = Vertex.ByCoordinates(0,0,0)
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
                e = Edge.ByVertices([v1, v2])
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
                e = Edge.ByVertices([v1, v2])
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
    def ByParameters(face, uRange, vRange, clip):
        """
        Parameters
        ----------
        face : TYPE
            DESCRIPTION.
        uRange : TYPE
            DESCRIPTION.
        vRange : TYPE
            DESCRIPTION.
        clip : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        # face = item[0]
        # uRange = item[1]
        # vRange = item[2]
        # clip = item[3]
        if isinstance(clip, list):
            clip = clip[0]
        uvWireEdges = []
        uCluster = None
        vCluster = None
        uvWire = None
        if len(uRange) > 0:
            if (min(uRange) < 0) or (max(uRange) > 1):
                raise Exception("Face.GridByParameters - Error: uRange input values are outside acceptable range (0,1)")
            uRange.sort()
            uRangeEdges = []
            for u in uRange:
                v1 = topologic.FaceUtility.VertexAtParameters(face, u, 0)
                v2 = topologic.FaceUtility.VertexAtParameters(face, u, 1)
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                uRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(uRangeEdges) > 0:
                uCluster = topologic.Cluster.ByTopologies(uRangeEdges)
                if clip:
                    uCluster = uCluster.Intersect(face, False)
        if len(vRange) > 0:
            if (min(vRange) < 0) or (max(vRange) > 1):
                raise Exception("Face.GridByParameters - Error: vRange input values are outside acceptable range (0,1)")
            vRange.sort()
            vRangeEdges = []
            for v in vRange:
                v1 = topologic.FaceUtility.VertexAtParameters(face, 0, v)
                v2 = topologic.FaceUtility.VertexAtParameters(face, 1, v)
                e = topologic.Edge.ByStartVertexEndVertex(v1, v2)
                vRangeEdges.append(e)
                uvWireEdges.append(e)
            if len(vRangeEdges) > 0:
                vCluster = topologic.Cluster.ByTopologies(vRangeEdges)
                if clip:
                    vCluster = vCluster.Intersect(face, False)
        if len(uvWireEdges) > 0 and uCluster and vCluster:
            uvWire = uCluster.Merge(vCluster)
        return [uCluster, vCluster, uvWire]
        