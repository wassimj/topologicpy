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

from binascii import a2b_base64
from re import A
import topologic_core as topologic
from topologicpy.Topology import Topology
import math
import itertools

class Wire(Topology):
    @staticmethod
    def Arc(startVertex, middleVertex, endVertex, sides: int = 16, close: bool = True, tolerance: float = 0.0001):
        """
        Creates an arc. The base chord will be parallel to the x-axis and the height will point in the positive y-axis direction. 

        Parameters
        ----------
        startVertex : topologic_core.Vertex
            The location of the start vertex of the arc.
        middleVertex : topologic_core.Vertex
            The location of the middle vertex (apex) of the arc.
        endVertex : topologic_core.Vertex
            The location of the end vertex of the arc.
        sides : int , optional
            The number of sides of the arc. The default is 16.
        close : bool , optional
            If set to True, the arc will be closed by connecting the last vertex to the first vertex. Otherwise, it will be left open.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created arc.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        def segmented_arc(x1, y1, x2, y2, x3, y3, sides):
            import math
            """
            Generates a segmented arc passing through the three given points.

            Arguments:
            x1, y1: Coordinates of the first point
            x2, y2: Coordinates of the second point
            x3, y3: Coordinates of the third point
            sides: Number of sides to divide the arc

            Returns:
            List of tuples [x, y] representing the segmented arc passing through the points
            """

            # Calculate the center of the circle
            A = x2 - x1
            B = y2 - y1
            C = x3 - x1
            D = y3 - y1
            E = A * (x1 + x2) + B * (y1 + y2)
            F = C * (x1 + x3) + D * (y1 + y3)
            G = 2 * (A * (y3 - y2) - B * (x3 - x2))
            if G == 0:
                center_x = 0
                center_y = 0
            else:
                center_x = (D * E - B * F) / G
                center_y = (A * F - C * E) / G

            # Calculate the radius of the circle
            radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

            # Calculate the angles between the center and the three points
            angle1 = math.atan2(y1 - center_y, x1 - center_x)
            angle3 = math.atan2(y3 - center_y, x3 - center_x)

            # Calculate the angle between points 1 and 3
            angle13 = (angle3 - angle1) % (2 * math.pi)
            if angle13 < 0:
                angle13 += 2 * math.pi

            # Determine the direction of the arc based on the angle between points 1 and 3
            if angle13 < math.pi:
                start_angle = angle3
                end_angle = angle1
            else:
                start_angle = angle1
                end_angle = angle3

            # Calculate the angle increment
            angle_increment = (end_angle - start_angle) / sides

            # Generate the points of the arc passing through the points
            arc_points = []
            for i in range(sides + 1):
                angle = start_angle + i * angle_increment
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                arc_points.append([x, y])

            return arc_points
        if not Topology.IsInstance(startVertex, "Vertex"):
            print("Wire.Arc - Error: The startVertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(middleVertex, "Vertex"):
            print("Wire.Arc - Error: The middleVertex parameter is not a valid vertex. Returning None.")
            return None
        if not Topology.IsInstance(endVertex, "Vertex"):
            print("Wire.Arc - Error: The endVertex parameter is not a valid vertex. Returning None.")
            return None
        if Vertex.AreCollinear([startVertex, middleVertex, endVertex], tolerance = tolerance):
            return Wire.ByVertices([startVertex, middleVertex, endVertex], close=False)
        
        w = Wire.ByVertices([startVertex, middleVertex, endVertex], close=True)
        f = Face.ByWire(w, tolerance=tolerance)
        normal = Face.Normal(f)
        flat_w = Topology.Flatten(w, origin=startVertex, direction=normal)
        v1, v2, v3 = Topology.Vertices(flat_w)
        x1, y1, z1 = Vertex.Coordinates(v1)
        x2, y2, z2 = Vertex.Coordinates(v2)
        x3, y3, z3 = Vertex.Coordinates(v3)
        arc_points = segmented_arc(x1, y1, x2, y2, x3, y3, sides)
        arc_verts = [Vertex.ByCoordinates(coord[0], coord[1], 0) for coord in arc_points]
        arc = Wire.ByVertices(arc_verts, close=close)
        # Unflatten the arc
        arc = Topology.Unflatten(arc, origin=startVertex, direction=normal)
        return arc
    
    @staticmethod
    def BoundingRectangle(topology, optimize: int = 0, mantissa: int = 6, tolerance=0.0001):
        """
        Returns a wire representing a bounding rectangle of the input topology. The returned wire contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting wire will become axis-aligned.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Wire
            The bounding rectangle of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from random import sample
        import time


        def br(topology):
            vertices = []
            _ = topology.Vertices(None, vertices)
            x = []
            y = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
            minX = min(x)
            minY = min(y)
            maxX = max(x)
            maxY = max(y)
            return [minX, minY, maxX, maxY]

        if not Topology.IsInstance(topology, "Topology"):
            return None

        world_origin = Vertex.Origin()

        vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
        start = time.time()
        period = 0
        result = True
        while result and period < 30:
            vList = sample(vertices, 3)
            result = Vertex.AreCollinear(vList)
            end = time.time()
            period = end - start
        if result == True:
            print("Wire.BoundingRectangle - Error: Could not find three vertices that are not colinear within 30 seconds. Returning None.")
            return None
        w = Wire.ByVertices(vList)
        f = Face.ByWire(w, tolerance=tolerance)
        f_origin = Topology.Centroid(f)
        normal = Face.Normal(f, mantissa=mantissa)
        topology = Topology.Flatten(topology, origin=f_origin, direction=normal)
        
        boundingRectangle = br(topology)
        minX = boundingRectangle[0]
        minY = boundingRectangle[1]
        maxX = boundingRectangle[2]
        maxY = boundingRectangle[3]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        best_area = l*w
        orig_area = best_area
        best_z = 0
        best_br = boundingRectangle
        origin = Topology.Centroid(topology)
        optimize = min(max(optimize, 0), 10)
        if optimize > 0:
            factor = (round(((11 - optimize)/30 + 0.57), 2))
            flag = False
            for n in range(10, 0, -1):
                if flag:
                    break
                za = n
                zb = 90+n
                zc = n
                for z in range(za,zb,zc):
                    if flag:
                        break
                    t = Topology.Rotate(topology, origin=origin, axis=[0, 0, 1], angle=z)
                    minX, minY, maxX, maxY = br(t)
                    w = abs(maxX - minX)
                    l = abs(maxY - minY)
                    area = l*w
                    if area < orig_area*factor:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        flag = True
                        break
                    if area < best_area:
                        best_area = area
                        best_z = z
                        best_br = [minX, minY, maxX, maxY]
                        
        else:
            best_br = boundingRectangle

        minX, minY, maxX, maxY = best_br
        vb1 = Vertex.ByCoordinates(minX, minY, 0)
        vb2 = Vertex.ByCoordinates(maxX, minY, 0)
        vb3 = Vertex.ByCoordinates(maxX, maxY, 0)
        vb4 = Vertex.ByCoordinates(minX, maxY, 0)

        boundingRectangle = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        boundingRectangle = Topology.Rotate(boundingRectangle, origin=origin, axis=[0, 0, 1], angle=-best_z)
        boundingRectangle = Topology.Unflatten(boundingRectangle, origin=f_origin, direction=normal)
        dictionary = Dictionary.ByKeysValues(["zrot"], [best_z])
        boundingRectangle = Topology.SetDictionary(boundingRectangle, dictionary)
        return boundingRectangle

    @staticmethod
    def ByEdges(edges: list, orient: bool = False, tolerance: float = 0.0001):
        """
        Creates a wire from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        orient : bool , optional
            If set to True the edges are oriented head to tail. Otherwise, they are not. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(edges, list):
            return None
        edgeList = [x for x in edges if Topology.IsInstance(x, "Edge")]
        if len(edgeList) == 0:
            print("Wire.ByEdges - Error: The input edges list does not contain any valid edges. Returning None.")
            return None
        if len(edgeList) == 1:
            wire = topologic.Wire.ByEdges(edgeList) # Hook to Core
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(edgeList), tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.ByEdges - Error: The operation failed. Returning None.")
            wire = None
        if Wire.IsManifold(wire):
            if orient == True:
                wire = Wire.OrientEdges(wire, Wire.StartVertex(wire), tolerance=tolerance)
        return wire

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a wire from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        if not Topology.IsInstance(cluster, "Cluster"):
            print("Wire.ByEdges - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = []
        _ = cluster.Edges(None, edges)
        return Wire.ByEdges(edges, tolerance=tolerance)

    @staticmethod
    def ByOffset(wire, offset: float = 1.0, bisectors: bool = False, tolerance: float = 0.0001):
        """
        Creates an offset wire from the input wire. A positive offset value results in an offset to the interior of the anti-clockwise wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float , optional
            The desired offset distance. The default is 1.0.
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned wire. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        def compute_h(alpha, offset):
            import math
            alpha = math.radians(alpha) *0.5
            h = offset/math.cos(alpha)
            return h
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vector import Vector
        import math

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.ByOffset - Error: The input wire parameter is not a valid wire. Returning None")
            return None
        if abs(offset) < tolerance:
            return wire
        vertices = Topology.Vertices(wire)
        if len(vertices) < 3:
            print("Wire.ByOffset - Error: The input wire parameter contains less than three vertices. Cannot proceed. Returning None")
            return None
        if not Wire.IsClosed(wire):
            w = Wire.ByVertices(vertices, close=True)
        else:
            w = wire
        three_vertices = Wire.Vertices(w)[0:3]
        temp_w = Wire.ByVertices(three_vertices, close=True)
        flat_face = Face.ByWire(temp_w, tolerance=tolerance)
        origin = Vertex.Origin()
        normal = Face.Normal(flat_face)
        flat_wire = Topology.Flatten(w, origin=origin, direction=normal)
        edges = Wire.Edges(flat_wire)
        vertices = Wire.Vertices(flat_wire)
        int_angles = Wire.InteriorAngles(flat_wire)
        bisector_list = []
        final_vertices = []
        for i in range(len(vertices)):
            v_edges = Topology.SuperTopologies(vertices[i], flat_wire, topologyType="edge")
            if len(v_edges) < 2:
                continue
            bisector = Edge.Bisect(v_edges[0], v_edges[1], length=offset, placement=1)
            if int_angles[i] > 180:
                bisector = Topology.TranslateByDirectionDistance(bisector, Vector.Reverse(Edge.Direction(bisector)), distance=abs(offset))
                bisector = Edge.Reverse(bisector)
            
            h = abs(compute_h(180-int_angles[i], abs(offset)))
            bisector = Edge.SetLength(bisector, length=h, bothSides=False)
            final_vertices.append(Edge.EndVertex(bisector))        
            bisector_list.append(bisector)
        return_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire))
        vertices = Topology.Vertices(return_wire)
        if not Wire.IsClosed(wire):
            sv = Topology.Vertices(flat_wire)[0]
            ev = Topology.Vertices(flat_wire)[-1]
            first_edge = Topology.SuperTopologies(sv, flat_wire, topologyType="edge")[0]
            first_normal = Edge.NormalAsEdge(first_edge, length=abs(offset), u = 0)
            last_edge = Topology.SuperTopologies(ev, flat_wire, topologyType="edge")[0]
            last_normal = Edge.NormalAsEdge(last_edge, length=abs(offset), u = 1.0)
            sv1 = Edge.EndVertex(first_normal)
            ev1 = Edge.EndVertex(last_normal)
            vertices = [sv1] + vertices[1:-1] + [ev1]
            bisector_list = [first_normal] + bisector_list[1:-1] + [last_normal]
        return_wire = Wire.ByVertices(vertices, close=Wire.IsClosed(wire))
        if bisectors:
            return_wire = Topology.SelfMerge(Cluster.ByTopologies(bisector_list, Topology.Edges(return_wire)))
        return_wire = Topology.Unflatten(return_wire, origin=origin, direction=normal)
        return return_wire

    @staticmethod
    def ByVertices(vertices: list, close: bool = True, tolerance: float = 0.0001):
        """
        Creates a wire from the input list of vertices.

        Parameters
        ----------
        vertices : list
            the input list of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 2:
            print("Wire.ByVertices - Error: The number of vertices is less than 2. Returning None.")
            return None
        edges = []
        for i in range(len(vertexList)-1):
            v1 = vertexList[i]
            v2 = vertexList[i+1]
            e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
        if close:
            v1 = vertexList[-1]
            v2 = vertexList[0]
            e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
            if Topology.IsInstance(e, "Edge"):
                edges.append(e)
        if len(edges) < 1:
            print("Wire.ByVertices - Error: The number of edges is less than 1. Returning None.")
            return None
        elif len(edges) == 1:
            wire = Wire.ByEdges(edges, orient=False)
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
        return wire

    @staticmethod
    def ByVerticesCluster(cluster, close: bool = True):
        """
        Creates a wire from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.cluster
            the input cluster of vertices.
        close : bool , optional
            If True the last vertex will be connected to the first vertex to close the wire. The default is True.

        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices)
        return Wire.ByVertices(vertices, close)

    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 0.5.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created circle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Circle - Error: The input origin parameter is not a valid Vertex. Retruning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Circle - Error: The input placement parameter is not a recognised string. Retruning None.")
            return None
        radius = abs(radius)
        if radius < tolerance:
            return None
        
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) < tolerance:
            return None
        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(Vertex.ByCoordinates(x, y, z))

        if angleRange == 360:
            baseWire = Wire.ByVertices(baseV[::-1], close=False) #reversing the list so that the normal points up in Blender
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, radius, radius, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -radius, radius, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire
    
    @staticmethod
    def Close(wire, mantissa=6, tolerance=0.0001):
        """
        Closes the input wire

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The closed version of the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        
        def nearest_vertex(vertex, vertices):
            distances = []
            for v in vertices:
                distances.append(Vertex.Distance(vertex, v))
            new_vertices = Helper.Sort(vertices, distances)
            return new_vertices[1] #The first item is the same vertex, so return the next nearest vertex.
        
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Close - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if Wire.IsClosed(wire):
            return wire
        vertices = Topology.Vertices(wire)
        ends = [v for v in vertices if Vertex.Degree(v, wire) == 1]
        if len(ends) < 2:
            print("Wire.Close - Error: The input wire parameter contains less than two open end vertices. Returning None.")
            return None
        geometry = Topology.Geometry(wire, mantissa=mantissa)
        g_vertices = geometry['vertices']
        g_edges = geometry['edges']
        used = []
        for end in ends:
            nearest = nearest_vertex(end, ends)
            if not nearest in used:
                d = Vertex.Distance(end, nearest)
                i1 = Vertex.Index(end, vertices, tolerance=tolerance)
                i2 = Vertex.Index(nearest, vertices, tolerance=tolerance)
                if i1 == None or i2 == None:
                    print("Wire.Close - Error: Something went wrong. Returning None.")
                    return None
                if d < tolerance:
                    g_vertices[i1] = Vertex.Coordinates(end)
                    g_vertices[i2] = Vertex.Coordinates(end)
                else:
                    if not(([i1, i2] in g_edges) or ([i2, i1] in g_edges)):
                        g_edges.append([i1, i2])
                used.append(end)
        new_wire = Topology.ByGeometry(vertices=g_vertices, edges=g_edges, faces=[], outputMode="wire")
        return new_wire

    @staticmethod
    def ConvexHull(topology, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a wire representing the 2D convex hull of the input topology. The vertices of the topology are assumed to be coplanar.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
                
        Returns
        -------
        topologic_core.Wire
            The convex hull of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from random import sample


        def Left_index(points):
            
            '''
            Finding the left most point
            '''
            minn = 0
            for i in range(1,len(points)):
                if points[i][0] < points[minn][0]:
                    minn = i
                elif points[i][0] == points[minn][0]:
                    if points[i][1] > points[minn][1]:
                        minn = i
            return minn

        def orientation(p, q, r):
            '''
            To find orientation of ordered triplet (p, q, r). 
            The function returns following values 
            0 --> p, q and r are collinear 
            1 --> Clockwise 
            2 --> Counterclockwise 
            '''
            val = (q[1] - p[1]) * (r[0] - q[0]) - \
                (q[0] - p[0]) * (r[1] - q[1])
        
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2
        
        def convex_hull(points, n):
            
            # There must be at least 3 points 
            if n < 3:
                return
        
            # Find the leftmost point
            l = Left_index(points)
        
            hull = []
            
            '''
            Start from leftmost point, keep moving counterclockwise 
            until reach the start point again. This loop runs O(h) 
            times where h is number of points in result or output. 
            '''
            p = l
            q = 0
            while(True):
                
                # Add current point to result 
                hull.append(p)
        
                '''
                Search for a point 'q' such that orientation(p, q, 
                x) is counterclockwise for all points 'x'. The idea 
                is to keep track of last visited most counterclock- 
                wise point in q. If any point 'i' is more counterclock- 
                wise than q, then update q. 
                '''
                q = (p + 1) % n
        
                for i in range(n):
                    
                    # If i is more counterclockwise 
                    # than current q, then update q 
                    if(orientation(points[p], 
                                points[i], points[q]) == 2):
                        q = i
        
                '''
                Now q is the most counterclockwise with respect to p 
                Set p as q for next iteration, so that q is added to 
                result 'hull' 
                '''
                p = q
        
                # While we don't come to first point
                if(p == l):
                    break
        
            # Print Result 
            return hull

        f = None
        # Create a sample face and flatten
        while not Topology.IsInstance(f, "Face"):
            vertices = Topology.SubTopologies(topology=topology, subTopologyType="vertex")
            v = sample(vertices, 3)
            w = Wire.ByVertices(v)
            f = Face.ByWire(w, tolerance=tolerance)
            origin = Topology.Centroid(f)
            normal = Face.Normal(f, mantissa=mantissa)
            f = Topology.Flatten(f, origin=origin, direction=normal)
        topology = Topology.Flatten(topology, origin=origin, direction=normal)
        vertices = Topology.Vertices(topology)
        points = []
        for v in vertices:
            points.append((Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa)))
        hull = convex_hull(points, len(points))
        hull_vertices = []
        for p in hull:
            hull_vertices.append(Vertex.ByCoordinates(points[p][0], points[p][1], 0))
        ch = Wire.ByVertices(hull_vertices)
        ch = Topology.Unflatten(ch, origin=origin, direction=normal)
        return ch

    @staticmethod
    def Cycles(wire, maxVertices: int = 4, tolerance: float = 0.0001) -> list:
        """
        Returns the closed circuits of wires found within the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        maxVertices : int , optional
            The maximum number of vertices of the circuits to be searched. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of circuits (closed wires) found within the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        def vIndex(v, vList, tolerance):
            for i in range(len(vList)):
                if Vertex.Distance(v, vList[i]) < tolerance:
                    return i+1
            return None
        
        #  rotate cycle path such that it begins with the smallest node
        def rotate_to_smallest(path):
            n = path.index(min(path))
            return path[n:]+path[:n]

        def invert(path):
            return rotate_to_smallest(path[::-1])

        def isNew(cycles, path):
            return not path in cycles

        def visited(node, path):
            return node in path

        def findNewCycles(graph, cycles, path, maxVertices):
            if len(path) > maxVertices:
                return
            start_node = path[0]
            next_node= None
            sub = []

            #visit each edge and each node of each edge
            for edge in graph:
                node1, node2 = edge
                if start_node in edge:
                        if node1 == start_node:
                            next_node = node2
                        else:
                            next_node = node1
                        if not visited(next_node, path):
                                # neighbor node not on path yet
                                sub = [next_node]
                                sub.extend(path)
                                # explore extended path
                                findNewCycles(graph, cycles, sub, maxVertices);
                        elif len(path) > 2  and next_node == path[-1]:
                                # cycle found
                                p = rotate_to_smallest(path);
                                inv = invert(p)
                                if isNew(cycles, p) and isNew(cycles, inv):
                                    cycles.append(p)

        def main(graph, cycles, maxVertices):
            returnValue = []
            for edge in graph:
                for node in edge:
                    findNewCycles(graph, cycles, [node], maxVertices)
            for cy in cycles:
                row = []
                for node in cy:
                    row.append(node)
                returnValue.append(row)
            return returnValue

        tEdges = []
        _ = wire.Edges(None, tEdges)
        tVertices = []
        _ = wire.Vertices(None, tVertices)
        tVertices = tVertices

        graph = []
        for anEdge in tEdges:
            graph.append([vIndex(anEdge.StartVertex(), tVertices, tolerance), vIndex(anEdge.EndVertex(), tVertices, tolerance)])

        cycles = []
        resultingCycles = main(graph, cycles, maxVertices)

        result = []
        for aRow in resultingCycles:
            row = []
            for anIndex in aRow:
                row.append(tVertices[anIndex-1])
            result.append(row)

        resultWires = []
        for i in range(len(result)):
            c = result[i]
            resultEdges = []
            for j in range(len(c)-1):
                v1 = c[j]
                v2 = c[j+1]
                e = Edge.ByStartVertexEndVertex(v1, v2, tolerance=tolerance, silent=True)
                resultEdges.append(e)
            e = Edge.ByStartVertexEndVertex(c[len(c)-1], c[0], tolerance=tolerance, silent=True)
            resultEdges.append(e)
            resultWire = Wire.ByEdges(resultEdges, tolerance=tolerance)
            resultWires.append(resultWire)
        return resultWires

    @staticmethod
    def Edges(wire) -> list:
        """
        Returns the edges of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of edges.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None
        edges = []
        _ = wire.Edges(None, edges)
        return edges

    @staticmethod
    def Einstein(origin= None, radius: float = 0.5, direction: list = [0, 0, 1], placement: str = "center", mantissa: int = 6):
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the tile. The default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        Returns
        -------
        topologic_core.Wire
            The created wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import math

        def cos(angle):
            return math.cos(math.radians(angle))
        def sin(angle):
            return math.sin(math.radians(angle))
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        d = cos(30)*radius
        v1 = Vertex.ByCoordinates(0, 0, 0)
        v2 = Vertex.ByCoordinates(cos(30)*d, sin(30)*d, 0)
        v3 = Vertex.ByCoordinates(radius, 0)
        v4 = Vertex.ByCoordinates(2*radius, 0)
        v5 = Vertex.ByCoordinates(2*radius+cos(60)*radius*0.5, sin(30)*d, 0)
        v6 = Vertex.ByCoordinates(1.5*radius, d)
        v7 = Vertex.ByCoordinates(1.5*radius, 2*d)
        v8 = Vertex.ByCoordinates(radius, 2*d)
        v9 = Vertex.ByCoordinates(radius-cos(60)*0.5*radius, 2*d+sin(60)*0.5*radius)
        v10 = Vertex.ByCoordinates(0, 2*d)
        v11 = Vertex.ByCoordinates(0, d)
        v12 = Vertex.ByCoordinates(-radius*0.5, d)
        v13 = Vertex.ByCoordinates(-cos(30)*d, sin(30)*d, 0)
        einstein = Wire.ByVertices([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13], close=True)
        
        if placement.lower() == "lowerleft":
            einstein = Topology.Translate(einstein, radius, d, 0)
        dx = Vertex.X(origin, mantissa=mantissa)
        dy = Vertex.Y(origin, mantissa=mantissa)
        dz = Vertex.Z(origin, mantissa=mantissa)
        einstein = Topology.Translate(einstein, dx, dy, dz)
        if direction != [0, 0, 1]:
            einstein = Topology.Orient(einstein, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return einstein
    
    @staticmethod
    def Ellipse(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: float = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created ellipse

        """
        ellipseAll = Wire.EllipseAll(origin=origin, inputMode=inputMode, width=width, length=length, focalLength=focalLength, eccentricity=eccentricity, majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=close, direction=direction, placement=placement, tolerance=tolerance)
        return ellipseAll["ellipse"]

    @staticmethod
    def EllipseAll(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str ="center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by wich the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dictionary
            A dictionary with the following keys and values:
            1. "ellipse" : The ellipse (topologic_core.Wire)
            2. "foci" : The two focal points (topologic_core.Cluster containing two vertices)
            3. "a" : The major axis length
            4. "b" : The minor axis length
            5. "c" : The focal length
            6. "e" : The eccentricity
            7. "width" : The width
            8. "length" : The length

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        if inputMode not in [1, 2, 3, 4]:
            return None
        if placement.lower() not in ["center", "lowerleft"]:
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            return None
        width = abs(width)
        length = abs(length)
        focalLength= abs(focalLength)
        eccentricity=abs(eccentricity)
        majorAxisLength=abs(majorAxisLength)
        minorAxisLength=abs(minorAxisLength)
        sides = abs(sides)
        if width < tolerance or length < tolerance or focalLength < tolerance or eccentricity < tolerance or majorAxisLength < tolerance or minorAxisLength < tolerance or sides < 3:
            return None
        if inputMode == 1:
            w = width
            l = length
            a = width/2
            b = length/2
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
        elif inputMode == 2:
            c = focalLength
            e = eccentricity
            a = c/e
            b = math.sqrt(abs(a**2 - c**2))
            w = a*2
            l = b*2
        elif inputMode == 3:
            c = focalLength
            b = minorAxisLength
            a = math.sqrt(abs(b**2 + c**2))
            e = c/a
            w = a*2
            l = b*2
        elif inputMode == 4:
            a = majorAxisLength
            b = minorAxisLength
            c = math.sqrt(abs(b**2 - a**2))
            e = c/a
            w = a*2
            l = b*2
        else:
            return None
        baseV = []
        xList = []
        yList = []

        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle - fromAngle) < tolerance:
            return None

        angleRange = toAngle - fromAngle
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        sides = int(math.floor(sides))
        for i in range(sides+1):
            angle = fromAngle + math.radians(angleRange/sides)*i
            x = math.sin(angle)*a + origin.X()
            y = math.cos(angle)*b + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append(Vertex.ByCoordinates(x, y, z))

        if angleRange == 360:
            baseWire = Wire.ByVertices(baseV[::-1], close=False) #reversing the list so that the normal points up in Blender
        else:
            baseWire = Wire.ByVertices(baseV[::-1], close=close) #reversing the list so that the normal points up in Blender

        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, a, b, 0)
        baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        # Create a Cluster of the two foci
        v1 = Vertex.ByCoordinates(c+origin.X(), 0+origin.Y(), 0)
        v2 = Vertex.ByCoordinates(-c+origin.X(), 0+origin.Y(), 0)
        foci = Cluster.ByTopologies([v1, v2])
        if placement.lower() == "lowerleft":
            foci = Topology.Translate(foci, a, b, 0)
        foci = Topology.Orient(foci, origin=origin, dirA=[0, 0, 1], dirB=direction)
        d = {}
        d['ellipse'] = baseWire
        d['foci'] = foci
        d['a'] = a
        d['b'] = b
        d['c'] = c
        d['e'] = e
        d['w'] = w
        d['l'] = l
        return d

    @staticmethod
    def EndVertex(wire):
        """
        Returns the end vertex of the input wire. The wire must be manifold and open.

        """
        sv, ev = Wire.StartEndVertices(wire)
        return ev
    
    @staticmethod
    def ExteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6) -> list:
        """
        Returns the exterior angles of the input wire in degrees. The wire must be planar, manifold, and closed.
        
        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        mantissa : int , optional
            The length of the desired mantissa. The default is 6.
        
        Returns
        -------
        list
            The list of exterior angles.
        """        

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None")
            return None
        if not Wire.IsClosed(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None")
            return None
        
        interior_angles = Wire.InteriorAngles(wire, mantissa=mantissa)
        exterior_angles = [round(360-a, mantissa) for a in interior_angles]
        return exterior_angles
    
    @staticmethod
    def ExternalBoundary(wire):
        """
        Returns the external boundary (cluster of vertices where degree == 1) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input wire. This is a cluster of vertices of degree == 1.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.ExternalBoundary - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        return Cluster.ByTopologies([v for v in Topology.Vertices(wire) if (Vertex.Degree(v, wire, topologyType="edge") == 1)])
    
    @staticmethod
    def Fillet(wire, radius: float = 0, radiusKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input wire given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        radius : float
            The desired radius of the fillet.
        radiusKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired fillet radius. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to False, error and warning messages are printed. Otherwise, they are not. The default is False.

        Returns
        -------
        topologic_core.Wire
            The filleted wire.

        """
        def start_from(edge, v):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(v, ev) < Vertex.Distance(v, sv):
                return Edge.Reverse(edge)
            return edge
        
        def compute_kite_edges(alpha, r):
            # Convert angle to radians
            alpha = math.radians(alpha) *0.5
            h = r/math.cos(alpha)
            a = math.sqrt(h*h - r*r)
            return [a,h]
        
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not manifold. Returning None.")
            return None
        if not Topology.IsPlanar(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not planar. Returning None.")
            return None

        orig_radius = radius
        f = Face.BoundingRectangle(wire, tolerance=tolerance)
        normal = Face.Normal(f)
        flat_wire = Topology.Flatten(wire, origin=Vertex.Origin(), direction=normal)
        vertices = Topology.Vertices(flat_wire)
        final_vertices = []
        fillets = []
        for v in vertices:
            radius = orig_radius
            edges = Topology.SuperTopologies(v, flat_wire, topologyType="edge")
            if len(edges) == 2:
                for edge in edges:
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(v, ev) < tolerance:
                        edge0 = edge
                    else:
                        edge1 = edge
                ang = Edge.Angle(edge0, edge1)
                e1 = start_from(edge0, v)
                e2 = start_from(edge1, v)

                dir1 = Edge.Direction(e1)
                dir2 = Edge.Direction(e2)
                if Vector.IsParallel(dir1, dir2) or Vector.IsAntiParallel(dir1, dir2):
                    pass
                else:
                    if isinstance(radiusKey, str):
                        d = Topology.Dictionary(v)
                        if Topology.IsInstance(d, "Dictionary"):
                            v_radius = Dictionary.ValueAtKey(d, radiusKey)
                            if isinstance(v_radius, float) or isinstance(v_radius, int):
                                if v_radius >= 0:
                                    radius = v_radius
                    if radius > 0:
                        dir_bisector = Vector.Bisect(dir1,dir2)
                        a, h = compute_kite_edges(ang, radius)
                        if a <= Edge.Length(e1) and a <= Edge.Length(e2):
                            v1 = Topology.TranslateByDirectionDistance(v, dir1, a)
                            center = Topology.TranslateByDirectionDistance(v, dir_bisector, h)
                            v2 = Topology.TranslateByDirectionDistance(v, dir2, a)
                            r1 = Edge.ByVertices(center, v1)
                            dir1 = Edge.Direction(r1)
                            r2 = Edge.ByVertices(center, v2)
                            dir2 = Edge.Direction(r2)
                            compass1 = Vector.CompassAngle(Vector.East(), dir1)*-1
                            compass2 = Vector.CompassAngle(Vector.East(), dir2)*-1
                            if compass2 < compass1:
                                temp = compass2
                                compass2 = compass1
                                compass1 = temp
                            w1 = Wire.Circle(origin=center, radius=radius, fromAngle=compass1, toAngle=compass2, close=False)
                            w2 = Wire.Circle(origin=center, radius=radius, fromAngle=compass2, toAngle=compass1, close=False)
                            if Wire.Length(w1) < Wire.Length(w2):
                                fillet = w1
                            else:
                                fillet = w2
                            f_sv = Wire.StartVertex(fillet)
                            if Vertex.Distance(f_sv, edge1) < Vertex.Distance(f_sv, edge0):
                                fillet = Wire.Reverse(fillet)
                            final_vertices += Topology.Vertices(fillet)
                        else:
                            if not silent:
                                print("Wire.Fillet - Error: The specified fillet radius is too large to be applied. Skipping.")
                    else:
                        final_vertices.append(v)
            else:
                final_vertices.append(v)
        flat_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire))
        # Unflatten the wire
        return_wire = Topology.Unflatten(flat_wire, origin=Vertex.Origin(), direction=normal)
        return return_wire
    
    @staticmethod
    def InteriorAngles(wire, tolerance: float = 0.0001, mantissa: int = 6) -> list:
        """
        Returns the interior angles of the input wire in degrees. The wire must be planar, manifold, and closed.
        This code has been contributed by Yidan Xue.
        
        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        list
            The list of interior angles.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.InteriorAngles - Error: The input wire parameter is not a valid wire. Returning None")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is non-manifold. Returning None")
            return None
        if not Wire.IsClosed(wire):
            print("Wire.InteriorAngles - Error: The input wire parameter is not closed. Returning None")
            return None
        
        f = Face.ByWire(wire)
        normal = Face.Normal(f)
        origin = Topology.Centroid(f)
        w = Topology.Flatten(wire, origin=origin, direction=normal)
        angles = []
        edges = Topology.Edges(w)
        e1 = edges[len(edges)-1]
        e2 = edges[0]
        a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
        angles.append(a)
        for i in range(len(edges)-1):
            e1 = edges[i]
            e2 = edges[i+1]
            a = Vector.CompassAngle(Vector.Reverse(Edge.Direction(e1)), Edge.Direction(e2))
            angles.append(round(a, mantissa))
        if abs(sum(angles)-(len(angles)-2)*180)<tolerance:
            return angles
        else:
            angles = [360-ang for ang in angles]
            return angles

    @staticmethod
    def Interpolate(wires: list, n: int = 5, outputType: str = "default", mapping: str = "default", tolerance: float = 0.0001):
        """
        Creates *n* number of wires that interpolate between wireA and wireB.

        Parameters
        ----------
        wireA : topologic_core.Wire
            The first input wire.
        wireB : topologic_core.Wire
            The second input wire.
        n : int , optional
            The number of intermediate wires to create. The default is 5.
        outputType : str , optional
            The desired type of output. The options are case insensitive. The default is "contour". The options are:
                - "Default" or "Contours" (wires are not connected)
                - "Raster or "Zigzag" or "Toolpath" (the wire ends are connected to create a continous path)
                - "Grid" (the wire ends are connected to create a grid). 
        mapping : str , optional
            The desired type of mapping for wires with different number of vertices. It is case insensitive. The default is "default". The options are:
                - "Default" or "Repeat" which repeats the last vertex of the wire with the least number of vertices
                - "Nearest" which maps the vertices of one wire to the nearest vertex of the next wire creating a list of equal number of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        tTopology
            The created interpolated wires as well as the input wires. The return type can be a topologic_core.Cluster or a topologic_core.Wire based on options.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Helper import Helper
        
        outputType = outputType.lower()
        if outputType not in ["default", "contours", "raster", "zigzag", "toolpath", "grid"]:
            return None
        if outputType == "default" or outputType == "contours":
            outputType = "contours"
        if outputType == "raster" or outputType == "zigzag" or outputType == "toolpath":
            outputType = "zigzag"
        
        mapping = mapping.lower()
        if mapping not in ["default", "nearest", "repeat"]:
            print("Wire.Interpolate - Error: The mapping input parameter is not recognized. Returning None.")
            return None
        
        def nearestVertex(v, vertices):
            distances = [Vertex.Distance(v, vertex) for vertex in vertices]
            return vertices[distances.index(sorted(distances)[0])]
        
        def replicate(vertices, mapping="default"):
            vertices = Helper.Repeat(vertices)
            finalList = vertices
            if mapping == "nearest":
                finalList = [vertices[0]]
                for i in range(len(vertices)-1):
                    loopA = vertices[i]
                    loopB = vertices[i+1]
                    nearestVertices = []
                    for j in range(len(loopA)):
                        nv = nearestVertex(loopA[j], loopB)
                        nearestVertices.append(nv)
                    finalList.append(nearestVertices)
            return finalList
        
        def process(verticesA, verticesB, n=5):
            contours = [verticesA]
            for i in range(1, n+1):
                u = float(i)/float(n+1)
                temp_vertices = []
                for j in range(len(verticesA)):
                    temp_v = Edge.VertexByParameter(Edge.ByVertices([verticesA[j], verticesB[j]], tolerance=tolerance), u)
                    temp_vertices.append(temp_v)
                contours.append(temp_vertices)
            return contours
        
        if len(wires) < 2:
            return None
        
        vertices = []
        for wire in wires:
            vertices.append(Topology.SubTopologies(wire, subTopologyType="vertex"))
        vertices = replicate(vertices, mapping=mapping)
        contours = []
        
        finalWires = []
        for i in range(len(vertices)-1):
            verticesA = vertices[i]
            verticesB = vertices[i+1]
            contour = process(verticesA=verticesA, verticesB=verticesB, n=n)
            contours += contour
            for c in contour:
                finalWires.append(Wire.ByVertices(c, Wire.IsClosed(wires[i])))

        contours.append(vertices[-1])
        finalWires.append(wires[-1])
        ridges = []
        if outputType == "grid" or outputType == "zigzag":
            for i in range(len(contours)-1):
                verticesA = contours[i]
                verticesB = contours[i+1]
                if outputType == "grid":
                    for j in range(len(verticesA)):
                        ridges.append(Edge.ByVertices([verticesA[j], verticesB[j]], tolerance=tolerance))
                elif outputType == "zigzag":
                    if i%2 == 0:
                        sv = verticesA[-1]
                        ev = verticesB[-1]
                        ridges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))
                    else:
                        sv = verticesA[0]
                        ev = verticesB[0]
                        ridges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))

        return Topology.SelfMerge(Cluster.ByTopologies(finalWires+ridges), tolerance=tolerance)
    
    @staticmethod
    def Invert(wire):
        """
        Creates a wire that is an inverse (mirror) of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        topologic_core.Wire
            The inverted wire.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None
        vertices = Wire.Vertices(wire)
        reversed_vertices = vertices[::-1]
        return Wire.ByVertices(reversed_vertices)

    @staticmethod
    def IsClosed(wire) -> bool:
        """
        Returns True if the input wire is closed. Returns False otherwise.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        bool
            True if the input wire is closed. False otherwise.

        """
        status = None
        if wire:
            if Topology.IsInstance(wire, "Wire"):
                status = wire.IsClosed()
        return status
    
    @staticmethod
    def IsManifold(wire) -> bool:
        """
        Returns True if the input wire is manifold. Returns False otherwise. A manifold wire is one where its vertices have a degree of 1 or 2.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        bool
            True if the input wire is manifold. False otherwise.
        """

        from topologicpy.Vertex import Vertex
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.IsManifold - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        
        vertices = Wire.Vertices(wire)
        for v in vertices:
            if Vertex.Degree(v, hostTopology=wire) > 2:
                return False
        return True

    @staticmethod
    def IsSimilar(wireA, wireB, angTolerance: float = 0.1, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input wires are similar. Returns False otherwise. The wires must be closed.

        Parameters
        ----------
        wireA : topologic_core.Wire
            The first input wire.
        wireB : topologic_core.Wire
            The second input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the two input wires are similar. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        
        def isCyclicallyEquivalent(u, v, lengthTolerance, angleTolerance):
            n, i, j = len(u), 0, 0
            if n != len(v):
                return False
            while i < n and j < n:
                if (i % 2) == 0:
                    tol = lengthTolerance
                else:
                    tol = angleTolerance
                k = 1
                while k <= n and math.fabs(u[(i + k) % n]- v[(j + k) % n]) <= tol:
                    k += 1
                if k > n:
                    return True
                if math.fabs(u[(i + k) % n]- v[(j + k) % n]) > tol:
                    i += k
                else:
                    j += k
            return False

        def angleBetweenEdges(e1, e2, tolerance):
            a = e1.EndVertex().X() - e1.StartVertex().X()
            b = e1.EndVertex().Y() - e1.StartVertex().Y()
            c = e1.EndVertex().Z() - e1.StartVertex().Z()
            d = Vertex.Distance(e1.EndVertex(), e2.StartVertex())
            if d <= tolerance:
                d = e2.StartVertex().X() - e2.EndVertex().X()
                e = e2.StartVertex().Y() - e2.EndVertex().Y()
                f = e2.StartVertex().Z() - e2.EndVertex().Z()
            else:
                d = e2.EndVertex().X() - e2.StartVertex().X()
                e = e2.EndVertex().Y() - e2.StartVertex().Y()
                f = e2.EndVertex().Z() - e2.StartVertex().Z()
            dotProduct = a*d + b*e + c*f
            modOfVector1 = math.sqrt( a*a + b*b + c*c)*math.sqrt(d*d + e*e + f*f) 
            angle = dotProduct/modOfVector1
            angleInDegrees = math.degrees(math.acos(angle))
            return angleInDegrees

        def getInteriorAngles(edges, tolerance):
            angles = []
            for i in range(len(edges)-1):
                e1 = edges[i]
                e2 = edges[i+1]
                angles.append(angleBetweenEdges(e1, e2, tolerance))
            return angles

        def getRep(edges, tolerance):
            angles = getInteriorAngles(edges, tolerance)
            lengths = []
            for anEdge in edges:
                lengths.append(Edge.Length(anEdge))
            minLength = min(lengths)
            normalisedLengths = []
            for aLength in lengths:
                normalisedLengths.append(aLength/minLength)
            return [x for x in itertools.chain(*itertools.zip_longest(normalisedLengths, angles)) if x is not None]
        
        if (wireA.IsClosed() == False):
            return None
        if (wireB.IsClosed() == False):
            return None
        edgesA = []
        _ = wireA.Edges(None, edgesA)
        edgesB = []
        _ = wireB.Edges(None, edgesB)
        if len(edgesA) != len(edgesB):
            return False
        repA = getRep(list(edgesA), tolerance)
        repB = getRep(list(edgesB), tolerance)
        if isCyclicallyEquivalent(repA, repB, tolerance, angTolerance):
            return True
        if isCyclicallyEquivalent(repA, repB[::-1], tolerance, angTolerance):
            return True
        return False

    @staticmethod
    def Length(wire, mantissa: int = 6) -> float:
        """
        Returns the length of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        float
            The length of the input wire. Test

        """
        from topologicpy.Edge import Edge
        if not wire:
            return None
        if not Topology.IsInstance(wire, "Wire"):
            return None
        totalLength = None
        try:
            edges = []
            _ = wire.Edges(None, edges)
            totalLength = 0
            for anEdge in edges:
                totalLength = totalLength + Edge.Length(anEdge)
            totalLength = round(totalLength, mantissa)
        except:
            totalLength = None
        return totalLength

    @staticmethod
    def Line(origin= None, length: float = 1, direction: list = [1, 0, 0], sides: int = 2, placement: str ="center"):
        """
        Creates a straight line wire using the input parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin location of the box. The default is None which results in the edge being placed at (0, 0, 0).
        length : float , optional
            The desired length of the edge. The default is 1.0.
        direction : list , optional
            The desired direction (vector) of the edge. The default is [1, 0, 0] (along the X-axis).
        sides : int , optional
            The desired number of sides/segments. The minimum number of sides is 2. The default is 2.
        placement : str , optional
            The desired placement of the edge. The options are:
            1. "center" which places the center of the edge at the origin.
            2. "start" which places the start of the edge at the origin.
            3. "end" which places the end of the edge at the origin.
            The default is "center".

        Returns
        -------
        topologic_core.Edge
            The created edge
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Line - Error: The input origin is not a valid vertex. Returning None.")
            return None
        if length <= 0:
            print("Wire.Line - Error: The input length is less than or equal to zero. Returning None.")
            return None
        if not isinstance(direction, list):
            print("Wire.Line - Error: The input direction is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            print("Wire.Line - Error: The length of the input direction is not equal to three. Returning None.")
            return None
        if sides < 2:
            print("Wire.Line - Error: The number of sides cannot be less than two. Consider using Edge.Line() instead. Returning None.")
            return None
        edge = Edge.Line(origin=origin, length=length, direction=direction, placement=placement)
        vertices = [Edge.StartVertex(edge)]
        unitDistance = float(1)/float(sides)
        for i in range(1, sides):
            vertices.append(Edge.VertexByParameter(edge, i*unitDistance))
        vertices.append(Edge.EndVertex(edge))
        return Wire.ByVertices(vertices)
    
    @staticmethod
    def Miter(wire, offset: float = 0, offsetKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input wire given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        offset : float
            The desired offset length of the miter along each edge.
        offsetKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired offset length. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to False, error and warning messages are printed. Otherwise, they are not. The default is False.

        Returns
        -------
        topologic_core.Wire
            The filleted wire.

        """
        def start_from(edge, v):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            if Vertex.Distance(v, ev) < Vertex.Distance(v, sv):
                return Edge.Reverse(edge)
            return edge
        
        def compute_kite_edges(alpha, r):
            # Convert angle to radians
            alpha = math.radians(alpha) *0.5
            h = r/math.cos(alpha)
            a = math.sqrt(h*h - r*r)
            return [a,h]
        
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not manifold. Returning None.")
            return None
        if not Topology.IsPlanar(wire):
            if not silent:
                print("Wire.Fillet - Error: The input wire parameter is not planar. Returning None.")
            return None

        orig_offset = offset
        f = Face.BoundingRectangle(wire, tolerance=tolerance)
        normal = Face.Normal(f)
        flat_wire = Topology.Flatten(wire, origin=Vertex.Origin(), direction=normal)
        vertices = Topology.Vertices(flat_wire)
        final_vertices = []
        miters = []
        for v in vertices:
            offset = orig_offset
            edges = Topology.SuperTopologies(v, flat_wire, topologyType="edge")
            if len(edges) == 2:
                for edge in edges:
                    ev = Edge.EndVertex(edge)
                    if Vertex.Distance(v, ev) < tolerance:
                        edge0 = edge
                    else:
                        edge1 = edge
                ang = Edge.Angle(edge0, edge1)
                e1 = start_from(edge0, v)
                e2 = start_from(edge1, v)

                dir1 = Edge.Direction(e1)
                dir2 = Edge.Direction(e2)
                if Vector.IsParallel(dir1, dir2) or Vector.IsAntiParallel(dir1, dir2):
                    pass
                else:
                    if isinstance(offsetKey, str):
                        d = Topology.Dictionary(v)
                        if Topology.IsInstance(d, "Dictionary"):
                            v_offset = Dictionary.ValueAtKey(d, offsetKey)
                            if isinstance(v_offset, float) or isinstance(v_offset, int):
                                if v_offset >= 0:
                                    offset = v_offset
                    if offset > 0 and offset <= Edge.Length(e1) and offset <=Edge.Length(e2):
                        v1 = Topology.TranslateByDirectionDistance(v, dir1, offset)
                        v2 = Topology.TranslateByDirectionDistance(v, dir2, offset)
                        final_vertices += [v1,v2]
                    else:
                        print("Wire.Fillet - Warning: The input offset parameter is greater than the length of the edge. Skipping.")
                        final_vertices.append(v)
            else:
                final_vertices.append(v)
        flat_wire = Wire.ByVertices(final_vertices, close=Wire.IsClosed(wire))
        # Unflatten the wire
        return_wire = Topology.Unflatten(flat_wire, origin=Vertex.Origin(), direction=normal)
        return return_wire
    
    @staticmethod
    def OrientEdges(wire, vertexA, tolerance=0.0001):
        """
        Returns a correctly oriented head-to-tail version of the input wire. The input wire must be manifold.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertexA : topologic_core.Vertex
            The desired start vertex of the wire.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The oriented wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.OrientEdges - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Topology.IsInstance(vertexA, "Vertex"):
            print("Wire.OrientEdges - Error: The input vertexA parameter is not a valid vertex. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.OrientEdges - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        oriented_edges = []
        remaining_edges = Topology.Edges(wire)
        current_vertex = vertexA
        while remaining_edges:
            next_edge = None
            for edge in remaining_edges:
                if Vertex.Distance(Edge.StartVertex(edge), current_vertex) < tolerance:
                    next_edge = edge
                    break
                elif Vertex.Distance(Edge.EndVertex(edge), current_vertex) < tolerance:
                    next_edge = Edge.Reverse(edge)
                    break

            if next_edge:
                oriented_edges.append(next_edge)
                remaining_edges.remove(next_edge)
                current_vertex = Edge.EndVertex(next_edge)
            else:
                # Unable to find a next edge connected to the current vertex
                break
        vertices = [Edge.StartVertex(oriented_edges[0])]
        for i, edge in enumerate(oriented_edges):
            vertices.append(Edge.EndVertex(edge))
            
        return Wire.ByVertices(vertices, close=Wire.IsClosed(wire))

    @staticmethod
    def Planarize(wire, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a planarized version of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.
        origin : topologic_core.Vertex , optional
            The desired origin of the plane unto which the planar wire will be projected. If set to None, the centroid of the input wire will be chosen. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        topologic_core.Wire
            The planarized wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Planarize - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Planarize - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        vertices = Topology.Vertices(wire)
        edges = Topology.Edges(wire)
        plane_equation = Vertex.PlaneEquation(vertices, mantissa=mantissa)
        rect = Face.RectangleByPlaneEquation(origin=origin , equation=plane_equation, tolerance=tolerance)
        new_vertices = [Vertex.Project(v, rect, mantissa=mantissa) for v in vertices]
        new_vertices = Vertex.Fuse(new_vertices, mantissa=mantissa, tolerance=tolerance)
        new_edges = []
        for edge in edges:
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            sv1 = Vertex.Project(sv, rect)
            i = Vertex.Index(sv1, new_vertices, tolerance=tolerance)
            if i:
                sv1 = new_vertices[i]
            ev1 = Vertex.Project(ev, rect)
            i = Vertex.Index(ev1, new_vertices, tolerance=tolerance)
            if i:
                ev1 = new_vertices[i]
            new_edges.append(Edge.ByVertices([sv1, ev1]))
        return Topology.SelfMerge(Cluster.ByTopologies(new_edges), tolerance=tolerance)

    @staticmethod
    def Project(wire, face, direction: list = None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a projection of the input wire unto the input face.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        face : topologic_core.Face
            The face unto which to project the input wire.
        direction : list, optional
            The vector representing the direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The projected wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not wire:
            return None
        if not Topology.IsInstance(wire, "Wire"):
            return None
        if not face:
            return None
        if not Topology.IsInstance(face, "Face"):
            return None
        if not direction:
            direction = -1*Face.NormalAtParameters(face, 0.5, 0.5, "XYZ", mantissa)
        large_face = Topology.Scale(face, face.CenterOfMass(), 500, 500, 500)
        edges = []
        _ = wire.Edges(None, edges)
        projected_edges = []

        if large_face:
            if (Topology.Type(large_face) == Topology.TypeID("Face")):
                for edge in edges:
                    if edge:
                        if (Topology.Type(edge) == Topology.TypeID("Edge")):
                            sv = edge.StartVertex()
                            ev = edge.EndVertex()

                            psv = Vertex.Project(vertex=sv, face=large_face, direction=direction)
                            pev = Vertex.Project(vertex=ev, face=large_face, direction=direction)
                            if psv and pev:
                                try:
                                    pe = Edge.ByVertices([psv, pev], tolerance=tolerance)
                                    projected_edges.append(pe)
                                except:
                                    continue
        w = Wire.ByEdges(projected_edges, tolerance=tolerance)
        return w

    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created rectangle.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Rectangle - Error: specified origin is not a topologic vertex. Retruning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Rectangle - Error: Could not find placement in the list of placements. Retruning None.")
            return None
        width = abs(width)
        length = abs(length)
        if width < tolerance or length < tolerance:
            print("Wire.Rectangle - Error: One or more of the specified dimensions is below the tolerance value. Retruning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            print("Wire.Rectangle - Error: The direction vector magnitude is below the tolerance value. Retruning None.")
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5

        vb1 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = Vertex.ByCoordinates(origin.X()+width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = Vertex.ByCoordinates(origin.X()-width*0.5+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire
    
    @staticmethod
    def RemoveCollinearEdges(wire, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Removes any collinear edges in the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created wire without any collinear edges.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def cleanup(wire, tolerance):
            vertices = Topology.Vertices(wire)
            vertices = Vertex.Fuse(vertices, tolerance=tolerance)
            edges = Topology.Edges(wire)
            new_edges = []
            for edge in edges:
                sv = Edge.StartVertex(edge)
                sv = vertices[Vertex.Index(sv, vertices, tolerance=tolerance)]
                ev = Edge.EndVertex(edge)
                ev = vertices[Vertex.Index(ev, vertices, tolerance=tolerance)]
                if Vertex.Distance(sv, ev) > tolerance:
                    new_edges.append(Edge.ByVertices([sv,ev]))
            new_wire = Topology.SelfMerge(Cluster.ByTopologies(new_edges), tolerance=tolerance)
            return new_wire
        
        def rce(wire, angTolerance=0.1):
            if not Topology.IsInstance(wire, "Wire"):
                return wire
            final_wire = None
            vertices = []
            wire_verts = []
            try:
                _ = wire.Vertices(None, vertices)
            except:
                return wire
            for aVertex in vertices:
                edges = []
                _ = aVertex.Edges(wire, edges)
                if len(edges) > 1:
                    if not Edge.IsCollinear(edges[0], edges[1], tolerance=tolerance):
                        wire_verts.append(aVertex)
                else:
                    wire_verts.append(aVertex)
            if len(wire_verts) > 2:
                if wire.IsClosed():
                    final_wire = Wire.ByVertices(wire_verts, close=True)
                else:
                    final_wire = Wire.ByVertices(wire_verts, close=False)
            elif len(wire_verts) == 2:
                final_wire = Edge.ByStartVertexEndVertex(wire_verts[0], wire_verts[1], tolerance=tolerance, silent=True)
            return final_wire
        
        new_wire = cleanup(wire, tolerance=tolerance)
        if not Wire.IsManifold(new_wire):
            wires = Wire.Split(new_wire)
        else:
            wires = [new_wire]
        returnWires = []
        for aWire in wires:
            if not Topology.IsInstance(aWire, "Wire"):
                returnWires.append(aWire)
            else:
                returnWires.append(rce(aWire, angTolerance=angTolerance))
        if len(returnWires) == 1:
            returnWire = returnWires[0]
            if Topology.IsInstance(returnWire, "Edge"):
                return Wire.ByEdges([returnWire], tolerance=tolerance)
            elif Topology.IsInstance(returnWire, "Wire"):
                return returnWire
            else:
                return wire
        elif len(returnWires) > 1:
            returnWire = Topology.SelfMerge(Cluster.ByTopologies(returnWires))
            if Topology.IsInstance(returnWire, "Edge"):
                return Wire.ByEdges([returnWire], tolerance=tolerance)
            elif Topology.IsInstance(returnWire, "Wire"):
                return returnWire
            else:
                return wire
        else:
            return wire
    
    @staticmethod
    def Reverse(wire, tolerance: float = 0.0001):
        """
        Creates a wire that has the reverse direction of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The reversed wire.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Reverse - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.Reverse - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        
        vertices = Topology.Vertices(wire)
        vertices.reverse()
        new_wire = Wire.ByVertices(vertices, close=Wire.IsClosed(wire), tolerance=tolerance)
        return new_wire

    @staticmethod
    def Roof(face, angle: float = 45, tolerance: float = 0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optioal
            The desired angle in degrees of the roof. The default is 45.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created roof. This method returns the roof as a set of edges. No faces are created.

        """
        from topologicpy import Polyskel
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import topologic_core as topologic
        import math

        def subtrees_to_edges(subtrees, polygon, slope):
            polygon_z = {}
            for x, y, z in polygon:
                polygon_z[(x, y)] = z

            edges = []
            for subtree in subtrees:
                source = subtree.source
                height = subtree.height
                z = slope * height
                source_vertex = Vertex.ByCoordinates(source.x, source.y, z)

                for sink in subtree.sinks:
                    if (sink.x, sink.y) in polygon_z:
                        z = 0
                    else:
                        z = None
                        for st in subtrees:
                            if st.source.x == sink.x and st.source.y == sink.y:
                                z = slope * st.height
                                break
                            for sk in st.sinks:
                                if sk.x == sink.x and sk.y == sink.y:
                                    z = slope * st.height
                                    break
                        if z is None:
                            height = subtree.height
                            z = slope * height
                    sink_vertex = Vertex.ByCoordinates(sink.x, sink.y, z)
                    if (source.x, source.y) == (sink.x, sink.y):
                        continue
                    e = Edge.ByStartVertexEndVertex(source_vertex, sink_vertex, tolerance=tolerance, silent=True)
                    if e not in edges and e != None:
                        edges.append(e)
            return edges
        
        def face_to_skeleton(face, angle=0):
            normal = Face.Normal(face)
            eb_wire = Face.ExternalBoundary(face)
            ib_wires = Face.InternalBoundaries(face)
            eb_vertices = Topology.Vertices(eb_wire)
            if normal[2] > 0:
                eb_vertices = list(reversed(eb_vertices))
            eb_polygon_coordinates = [(v.X(), v.Y(), v.Z()) for v in eb_vertices]
            eb_polygonxy = [(x[0], x[1]) for x in eb_polygon_coordinates]

            ib_polygonsxy = []
            zero_coordinates = eb_polygon_coordinates
            for ib_wire in ib_wires:
                ib_vertices = Topology.Vertices(ib_wire)
                if normal[2] > 0:
                    ib_vertices = list(reversed(ib_vertices))
                ib_polygon_coordinates = [(v.X(), v.Y(), v.Z()) for v in ib_vertices]
                ib_polygonxy = [(x[0], x[1]) for x in ib_polygon_coordinates]
                ib_polygonsxy.append(ib_polygonxy)
                zero_coordinates += ib_polygon_coordinates
            skeleton = Polyskel.skeletonize(eb_polygonxy, ib_polygonsxy)
            slope = math.tan(math.radians(angle))
            roofEdges = subtrees_to_edges(skeleton, zero_coordinates, slope)
            roofEdges = Helper.Flatten(roofEdges)+Topology.Edges(face)
            roofTopology = Topology.SelfMerge(Cluster.ByTopologies(roofEdges), tolerance=tolerance)
            return roofTopology
        
        if not Topology.IsInstance(face, "Face"):
            return None
        angle = abs(angle)
        if angle >= 90-tolerance:
            return None
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        d = Topology.Dictionary(flat_face)
        roof = face_to_skeleton(flat_face, angle)
        if not roof:
            return None
        roof = Topology.Unflatten(roof, origin=origin, direction=normal)
        return roof
    
    @staticmethod
    def Simplify(wire, tolerance=0.0001):
        """
            Simplifies the input wire edges based on the Douglas Peucker algorthim. See https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
            Part of this code was contributed by gaoxipeng. See https://github.com/wassimj/topologicpy/issues/35
            
            Parameters
            ----------
            wire : topologic_core.Wire
                The input wire.
            tolerance : float , optional
                The desired tolerance. The default is 0.0001. Edges shorter than this length will be removed.

            Returns
            -------
            topologic_core.Wire
                The simplified wire.
        """
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def perpendicular_distance(point, line_start, line_end):
            # Calculate the perpendicular distance from a point to a line segment
            x0 = point.X()
            y0 = point.Y()
            x1 = line_start.X()
            y1 = line_start.Y()
            x2 = line_end.X()
            y2 = line_end.Y()

            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = Vertex.Distance(line_start, line_end)

            return numerator / denominator

        def douglas_peucker(wire, tolerance):
            if isinstance(wire, list):
                points = wire
            else:
                points = Wire.Vertices(wire)
                # points.insert(0, points.pop())
            if len(points) <= 2:
                return points

            # Use the first and last points in the list as the starting and ending points
            start_point = points[0]
            end_point = points[-1]

            # Find the point with the maximum distance
            max_distance = 0
            max_index = 0

            for i in range(1, len(points) - 1):
                d = perpendicular_distance(points[i], start_point, end_point)
                if d > max_distance:
                    max_distance = d
                    max_index = i

            # If the maximum distance is less than the tolerance, no further simplification is needed
            if max_distance <= tolerance:
                return [start_point, end_point]

            # Recursively simplify
            first_segment = douglas_peucker(points[:max_index + 1], tolerance)
            second_segment = douglas_peucker(points[max_index:], tolerance)

            # Merge the two simplified segments
            return first_segment[:-1] + second_segment
        
        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.Simplify = Error: The input wire parameter is not a Wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            wires = Wire.Split(wire)
            new_wires = []
            for w in wires:
                if Topology.IsInstance(w, "Edge"):
                    if Edge.Length(w) > tolerance:
                        new_wires.append(w)
                elif Topology.IsInstance(w, "Wire"):
                    new_wires.append(Wire.Simplify(w, tolerance=tolerance))
            return_wire = Topology.SelfMerge(Cluster.ByTopologies(new_wires))
            return return_wire
        
        new_vertices = douglas_peucker(wire, tolerance=tolerance)
        new_wire = Wire.ByVertices(new_vertices, close=Wire.IsClosed(wire))
        return new_wire

    @staticmethod
    def Skeleton(face, tolerance=0.001):
        """
            Creates a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel


        Parameters
        ----------
        face : topologic_core.Face
            The input face.
       
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created straight skeleton.

        """
        if not Topology.IsInstance(face, "Face"):
            return None
        return Wire.Roof(face, angle=0, tolerance=tolerance)
    
    @staticmethod
    def Spiral(origin = None, radiusA : float = 0.05, radiusB : float = 0.5, height : float = 1, turns : int = 10, sides : int = 36, clockwise : bool = False, reverse : bool = False, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a spiral.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the spiral. The default is None which results in the spiral being placed at (0, 0, 0).
        radiusA : float , optional
            The initial radius of the spiral. The default is 0.05.
        radiusB : float , optional
            The final radius of the spiral. The default is 0.5.
        height : float , optional
            The height of the spiral. The default is 1.
        turns : int , optional
            The number of turns of the spiral. The default is 10.
        sides : int , optional
            The number of sides of one full turn in the spiral. The default is 36.
        clockwise : bool , optional
            If set to True, the spiral will be oriented in a clockwise fashion. Otherwise, it will be oriented in an anti-clockwise fashion. The default is False.
        reverse : bool , optional
            If set to True, the spiral will increase in height from the center to the circumference. Otherwise, it will increase in height from the conference to the center. The default is False.
        direction : list , optional
            The vector representing the up direction of the spiral. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the spiral. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".

        Returns
        -------
        topologic_core.Wire
            The created spiral.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        import math

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Spiral - Error: the input origin is not a valid topologic Vertex. Returning None.")
            return None
        if radiusA <= 0:
            print("Wire.Spiral - Error: the input radiusA cannot be less than or equal to zero. Returning None.")
            return None
        if radiusB <= 0:
            print("Wire.Spiral - Error: the input radiusB cannot be less than or equal to zero. Returning None.")
            return None
        if radiusA == radiusB:
            print("Wire.Spiral - Error: the inputs radiusA and radiusB cannot be equal. Returning None.")
            return None
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if turns <= 0:
            print("Wire.Spiral - Error: the input turns cannot be less than or equal to zero. Returning None.")
            return None
        if sides < 3:
            print("Wire.Spiral - Error: the input sides cannot be less than three. Returning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Spiral - Error: the input placement string is not one of center, lowerleft, upperleft, lowerright, or upperright. Returning None.")
            return None
        if (abs(direction[0]) + abs(direction[1]) + abs(direction[2])) < tolerance:
            print("Wire.Spiral - Error: the input direction vector is not a valid direction. Returning None.")
            return None
        
        vertices = []
        xList = []
        yList = []
        zList = []
        if clockwise:
            cw = -1
        else:
            cw = 1
        n_vertices = sides*turns + 1
        zOffset = height/float(n_vertices)
        if reverse == True:
            z = height
        else:
            z = 0
        ang = 0
        angOffset = float(360/float(sides))
        b = (radiusB - radiusA)/(2*math.pi*turns)
        while ang <= 360*turns:
            rad = math.radians(ang)
            x = (radiusA + b*rad)*math.cos(rad)*cw
            xList.append(x)
            y = (radiusA + b*rad)*math.sin(rad)
            yList.append(y)
            zList.append(z)
            if reverse == True:
                z = z - zOffset
            else:
                z = z + zOffset
            vertices.append(Vertex.ByCoordinates(x, y, z))
            ang = ang + angOffset
        
        minX = min(xList)
        maxX = max(xList)
        minY = min(yList)
        maxY = max(yList)
        radius = radiusA + radiusB*turns*0.5
        baseWire = Wire.ByVertices(vertices, close=False)
        if placement.lower() == "center":
            baseWire = Topology.Translate(baseWire, 0, 0, -height*0.5)
        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, -minX, -minY, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, -minX, -maxY, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -maxX, -minY, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -maxX, -maxY, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def Split(wire) -> list:
        """
        Splits the input wire into segments at its intersections (i.e. at any vertex where more than two edges meet).

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of split wire segments.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        
        def vertexDegree(v, wire):
            edges = []
            _ = v.Edges(wire, edges)
            return len(edges)
        
        def vertexOtherEdge(vertex, edge, wire):
            edges = []
            _ = vertex.Edges(wire, edges)
            if Topology.IsSame(edges[0], edge):
                return edges[-1]
            else:
                return edges[0]
        
        def edgeOtherVertex(edge, vertex):
            vertices = []
            _ = edge.Vertices(None, vertices)
            if Topology.IsSame(vertex, vertices[0]):
                return vertices[-1]
            else:
                return vertices[0]
        
        def edgeInList(edge, edgeList):
            for anEdge in edgeList:
                if Topology.IsSame(anEdge, edge):
                    return True
            return False
        
        vertices = []
        _ = wire.Vertices(None, vertices)
        hubs = []
        for aVertex in vertices:
            if vertexDegree(aVertex, wire) > 2:
                hubs.append(aVertex)
        wires = []
        global_edges = []
        for aVertex in hubs:
            hub_edges = []
            _ = aVertex.Edges(wire, hub_edges)
            wire_edges = []
            for hub_edge in hub_edges:
                if not edgeInList(hub_edge, global_edges):
                    current_edge = hub_edge
                    oe = edgeOtherVertex(current_edge, aVertex)
                    while vertexDegree(oe, wire) == 2:
                        if not edgeInList(current_edge, global_edges):
                            global_edges.append(current_edge)
                            wire_edges.append(current_edge)
                        current_edge = vertexOtherEdge(oe, current_edge, wire)
                        oe = edgeOtherVertex(current_edge, oe)
                    if not edgeInList(current_edge, global_edges):
                        global_edges.append(current_edge)
                        wire_edges.append(current_edge)
                    if len(wire_edges) > 1:
                        wires.append(Cluster.ByTopologies(wire_edges).SelfMerge())
                    else:
                        wires.append(wire_edges[0])
                    wire_edges = []
        if len(wires) < 1:
            return [wire]
        return wires
    
    @staticmethod
    def Square(origin= None, size: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. The default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created square.

        """
        return Wire.Rectangle(origin=origin, width=size, length=size, direction=direction, placement=placement, tolerance=tolerance)
    
    @staticmethod
    def Squircle(origin = None, radius: float = 0.5, sides: int = 121, a: float = 2.0, b: float = 2.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a Squircle which is a hybrid between a circle and a square. See https://en.wikipedia.org/wiki/Squircle

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the squircle. The default is None which results in the squircle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the squircle. The default is 0.5.
        sides : int , optional
            The number of sides of the squircle. The default is 121.
        a : float , optional
            The "a" factor affects the x position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        b : float , optional
            The "b" factor affects the y position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        radius : float , optional
            The desired radius of the squircle. The default is 0.5.
        sides : int , optional
            The desired number of sides for the squircle. The default is 100.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created squircle.
        """

        def get_squircle(a=1, b=1, radius=0.5, sides=100):
            import numpy as np
            t = np.linspace(0, 2*np.pi, sides)
            x = (np.abs(np.cos(t))**(1/a)) * np.sign(np.cos(t))
            y = (np.abs(np.sin(t))**(1/b)) * np.sign(np.sin(t))
            return x*radius, y*radius
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.Squircle - Error: The input origin parameter is not a valid Vertex. Retruning None.")
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            print("Wire.Squircle - Error: The input placement parameter is not a recognised string. Retruning None.")
            return None
        radius = abs(radius)
        if radius < tolerance:
            return None
        
        if a <= 0:
            print("Wire.Squircle - Error: The a input parameter must be a positive number. Returning None.")
            return None
        if b <= 0:
            print("Wire.Squircle - Error: The b input parameter must be a positive number. Returning None.")
            return None
        if a == 1 and b == 1:
            return Wire.Circle(radius=radius, sides=sides, direction=direction, placement=placement, tolerance=tolerance)
        x_list, y_list = get_squircle(a=a, b=b, radius=radius, sides=sides)
        vertices = []
        for i, x in enumerate(x_list):
            v = Vertex.ByCoordinates(x, y_list[i], 0)
            vertices.append(v)
        baseWire = Wire.ByVertices(vertices, close=True)
        baseWire = Topology.RemoveCollinearEdges(baseWire, angTolerance=angTolerance, tolerance=tolerance)
        baseWire = Wire.Simplify(baseWire, tolerance=tolerance)
        if placement.lower() == "lowerleft":
            baseWire = Topology.Translate(baseWire, radius, radius, 0)
        elif placement.lower() == "upperleft":
            baseWire = Topology.Translate(baseWire, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            baseWire = Topology.Translate(baseWire, -radius, radius, 0)
        elif placement.lower() == "upperright":
            baseWire = Topology.Translate(baseWire, -radius, -radius, 0)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def Star(origin= None, radiusA: float = 0.5, radiusB: float = 0.2, rays: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the star. The default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 8.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created star.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusA < tolerance or radiusB < tolerance:
            return None
        rays = abs(rays)
        if rays < 3:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        sides = rays*2 # Sides is double the number of rays
        baseV = []

        xList = []
        yList = []
        for i in range(sides):
            if i%2 == 0:
                radius = radiusA
            else:
                radius = radiusB
            angle = math.radians(360/sides)*i
            x = math.sin(angle)*radius + origin.X()
            y = math.cos(angle)*radius + origin.Y()
            z = origin.Z()
            xList.append(x)
            yList.append(y)
            baseV.append([x, y])

        if placement.lower() == "lowerleft":
            xmin = min(xList)
            ymin = min(yList)
            xOffset = origin.X() - xmin
            yOffset = origin.Y() - ymin
        elif placement.lower() == "upperleft":
            xmin = min(xList)
            ymax = max(yList)
            xOffset = origin.X() - xmin
            yOffset = origin.Y() - ymax
        elif placement.lower() == "lowerright":
            xmax = max(xList)
            ymin = min(yList)
            xOffset = origin.X() - xmax
            yOffset = origin.Y() - ymin
        elif placement.lower() == "upperright":
            xmax = max(xList)
            ymax = max(yList)
            xOffset = origin.X() - xmax
            yOffset = origin.Y() - ymax
        else:
            xOffset = 0
            yOffset = 0
        tranBase = []
        for coord in baseV:
            tranBase.append(Vertex.ByCoordinates(coord[0]+xOffset, coord[1]+yOffset, origin.Z()))
        
        baseWire = Wire.ByVertices(tranBase[::-1], True) #reversing the list so that the normal points up in Blender
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def StartEndVertices(wire) -> list:
        """
        Returns the start and end vertices of the input wire. The wire must be manifold and open.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Wire.IsManifold(wire):
            print("Wire.StartEndVertices - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        vertices = Topology.Vertices(wire)
        if Wire.IsClosed(wire):
            return [vertices[0], vertices[0]] # If the wire is closed, the start and end vertices are the same vertex
        endPoints = [v for v in vertices if (Vertex.Degree(v, wire) == 1)]
        if len(endPoints) < 2:
            print("Wire.StartEndVertices - Error: Could not find the end vertices if the input wire parameter. Returning None.")
            return None
        edge1 = Topology.SuperTopologies(endPoints[0], wire, topologyType="edge")[0]
        sv = Edge.StartVertex(edge1)
        if (Topology.IsSame(endPoints[0], sv)):
            wireStartVertex = endPoints[0]
            wireEndVertex = endPoints[1]
        else:
            wireStartVertex = endPoints[1]
            wireEndVertex = endPoints[0]
        return [wireStartVertex, wireEndVertex]
    
    @staticmethod
    def StartVertex(wire):
        """
        Returns the start vertex of the input wire. The wire must be manifold and open.

        """
        sv, ev = Wire.StartEndVertices(wire)
        return sv
    
    @staticmethod
    def Trapezoid(origin= None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0, 0, 0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. The default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. The default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. The default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. The default is 0.0.
        length : float , optional
            The length of the trapezoid. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Wire
            The created trapezoid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        widthA = abs(widthA)
        widthB = abs(widthB)
        length = abs(length)
        if widthA < tolerance or widthB < tolerance or length < tolerance:
            return None
        if not placement.lower() in ["center", "lowerleft", "upperleft", "lowerright", "upperright"]:
            return None
        xOffset = 0
        yOffset = 0
        if placement.lower() == "center":
            xOffset = -((-widthA*0.5 + offsetA) + (-widthB*0.5 + offsetB) + (widthA*0.5 + offsetA) + (widthB*0.5 + offsetB))/4.0
            yOffset = 0
        elif placement.lower() == "lowerleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = -(min((-widthA*0.5 + offsetA), (-widthB*0.5 + offsetB)))
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -(max((widthA*0.5 + offsetA), (widthB*0.5 + offsetB)))
            yOffset = -length*0.5

        vb1 = Vertex.ByCoordinates(origin.X()-widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb2 = Vertex.ByCoordinates(origin.X()+widthA*0.5+offsetA+xOffset,origin.Y()-length*0.5+yOffset,origin.Z())
        vb3 = Vertex.ByCoordinates(origin.X()+widthB*0.5+offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())
        vb4 = Vertex.ByCoordinates(origin.X()-widthB*0.5++offsetB+xOffset,origin.Y()+length*0.5+yOffset,origin.Z())

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], True)
        if direction != [0, 0, 1]:
            baseWire = Topology.Orient(baseWire, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return baseWire

    @staticmethod
    def VertexDistance(wire, vertex, origin= None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the distance, computed along the input wire of the input vertex from the input origin vertex.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        vertex : topologic_core.Vertex
            The input vertex
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input wire. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        float
            The distance of the input vertex from the input origin along the input wire.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Wire.VertexDistance - Error: The input vertex parameter is not a valid topologic vertex. Returning None.")
            return None
        wire_length = Wire.Length(wire)
        if wire_length < tolerance:
            print("Wire.VertexDistance: The input wire parameter is a degenerate topologic wire. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.VertexDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(vertex, wire, tolerance=tolerance):
            print("Wire.VertexDistance: The input vertex parameter is not internal to the input wire parameter. Returning None.")
            return None
        
        def distance_from_start(wire, v):
            total_distance = 0.0
            found = False
            # Iterate over the edges of the wire
            for edge in Wire.Edges(wire):
                if Vertex.IsInternal(v, edge, tolerance=tolerance):
                    total_distance += Vertex.Distance(Edge.StartVertex(edge), v)
                    found = True
                    break
                total_distance += Edge.Length(edge)
            if found == False:
                return None
            return total_distance
        
        d1 = distance_from_start(wire, vertex)
        d2 = distance_from_start(wire, origin)
        if d1 == None:
            print("Wire.VertexDistance - Error: The input vertex parameter is not internal to the input wire parameter. Returning None.")
            return None
        if d2 == None:
            print("Wire.VertexDistance - Error: The input origin parameter is not internal to the input wire parameter. Returning None.")
            return None
        return round(abs(d2-d1), mantissa)

    @staticmethod
    def VertexByDistance(wire, distance: float = 0.0, origin= None, tolerance = 0.0001):
        """
        Creates a vertex along the input wire offset by the input distance from the input origin.

        Parameters
        ----------
        edge : topologic_core.Edge
            The input edge.
        distance : float , optional
            The offset distance. The default is 0.
        origin : topologic_core.Vertex , optional
            The origin of the offset distance. If set to None, the origin will be set to the start vertex of the input edge. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        from topologicpy.Vertex import Vertex
        def compute_u(u):
            def count_decimal_places(number):
                try:
                    # Convert the number to a string to analyze decimal places
                    num_str = str(number)
                    # Split the number into integer and decimal parts
                    integer_part, decimal_part = num_str.split('.')
                    # Return the length of the decimal part
                    return len(decimal_part)
                except ValueError:
                    # If there's no decimal part, return 0
                    return 0
            dp = count_decimal_places(u)
            u = -(int(u) - u)
            return round(u,dp)

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexByDistance - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        wire_length = Wire.Length(wire)
        if wire_length < tolerance:
            print("Wire.VertexByDistance: The input wire parameter is a degenerate topologic wire. Returning None.")
            return None
        if abs(distance) < tolerance:
            return Wire.StartVertex(wire)
        if abs(distance - wire_length) < tolerance:
            return Wire.EndVertex(wire)
        if not Wire.IsManifold(wire):
            print("Wire.VertexAtParameter - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Wire.StartVertex(wire)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Wire.VertexByDistance - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        if not Vertex.IsInternal(origin, wire, tolerance=tolerance):
            print("Wire.VertexByDistance - Error: The input origin parameter is not internal to the input wire parameter. Returning None.")
            return None
        if Vertex.Distance(Wire.StartVertex(wire), origin) < tolerance:
            u = distance/wire_length
        elif Vertex.Distance(Wire.EndVertex(wire), origin) < tolerance:
            u = 1 - distance/wire_length
        else:
            d = Wire.VertexDistance(wire, origin) + distance
            u = d/wire_length

        return Wire.VertexByParameter(wire, u=compute_u(u))
    
    @staticmethod
    def VertexByParameter(wire, u: float = 0):
        """
        Creates a vertex along the input wire offset by the input *u* parameter. The wire must be manifold.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        u : float , optional
            The *u* parameter along the input topologic Wire. A parameter of 0 returns the start vertex. A parameter of 1 returns the end vertex. The default is 0.

        Returns
        -------
        topologic_core.Vertex
            The vertex at the input u parameter

        """
        from topologicpy.Edge import Edge

        if not Topology.IsInstance(wire, "Wire"):
            print("Wire.VertexAtParameter - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None
        if u < 0 or u > 1:
            print("Wire.VertexAtParameter - Error: The input u parameter is not within the valid range of [0, 1]. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Wire.VertexAtParameter - Error: The input wire parameter is non-manifold. Returning None.")
            return None
        
        if u == 0:
            return Wire.StartVertex(wire)
        if u == 1:
            return Wire.EndVertex(wire)
        
        edges = Wire.Edges(wire)
        total_length = 0.0
        edge_lengths = []
        
        # Compute the total length of the wire
        for edge in edges:
            e_length = Edge.Length(edge)
            edge_lengths.append(e_length)
            total_length += e_length

        # Initialize variables for tracking the current edge and accumulated length
        current_edge = None
        accumulated_length = 0.0

        # Iterate over the lines to find the appropriate segment
        for i, edge in enumerate(edges):
            edge_length = edge_lengths[i]

            # Check if the desired point is on this line
            if u * total_length <= accumulated_length + edge_length:
                current_edge = edge
                break
            else:
                accumulated_length += edge_length

        # Calculate the residual u value for the current line
        residual_u = (u * total_length - accumulated_length) / Edge.Length(current_edge)

        # Compute the point at the parameter on the current line
        vertex = Edge.VertexByParameter(current_edge, residual_u)

        return vertex

    @staticmethod
    def Vertices(wire) -> list:
        """
        Returns the list of vertices of the input wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.

        Returns
        -------
        list
            The list of vertices.

        """
        if not Topology.IsInstance(wire, "Wire"):
            return None
        vertices = []
        _ = wire.Vertices(None, vertices)
        return vertices

