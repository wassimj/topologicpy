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

import topologic_core as topologic
import warnings

try:
    import numpy as np
except:
    print("BVH - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("BVH - numpy library installed correctly.")
    except:
        warnings.warn("ANN - Error: Could not import numpy.")

class BVH:
    # A class for Axis-Aligned Bounding Box (AABB)
    class AABB:
        def __init__(self, min_point, max_point):
            self.min_point = np.array(min_point)
            self.max_point = np.array(max_point)
            self.centroid = (self.min_point + self.max_point) / 2.0

        def intersects(self, other):
            # Check if this AABB intersects with another AABB
            if other == None:
                return False
            return np.all(self.min_point <= other.max_point) and np.all(self.max_point >= other.min_point)

        def contains(self, point):
            # Check if a point is contained within the AABB
            return np.all(self.min_point <= point) and np.all(self.max_point >= point)

    # MeshObject class that stores a reference to the Topologic object
    class MeshObject:
        def __init__(self, vertices, topologic_object):
            self.vertices = np.array(vertices)
            self.aabb = BVH.AABB(np.min(vertices, axis=0), np.max(vertices, axis=0))
            self.centroid = np.mean(vertices, axis=0)
            self.topologic_object = topologic_object  # Store the Topologic object reference

    # BVH Node class
    class BVHNode:
        def __init__(self, aabb, left=None, right=None, objects=None):
            self.aabb = aabb
            self.left = left
            self.right = right
            self.objects = objects if objects else []

    @staticmethod
    def ByTopologies(*topologies, silent: bool = False):
        """
        Creates a BVH Tree from the input list of topologies. The input can be individual topologies each as an input argument or a list of topologies stored in one input argument.

        Parameters
        ----------
        topologies : list
            The list of topologies.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        BVH tree
            The created BVH tree.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if len(topologies) == 0:
            print("BVH.ByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
            return None
        if len(topologies) == 1:
            topologyList = topologies[0]
            if isinstance(topologyList, list):
                if len(topologyList) == 0:
                    if not silent:
                        print("BVH.ByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
                    return None
                else:
                    topologyList = [x for x in topologyList if Topology.IsInstance(x, "Topology")]
                    if len(topologyList) == 0:
                        if not silent:
                            print("BVH.ByTopologies - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
                        return None
            else:
                if not silent:
                    print("BVH.ByTopologies - Warning: The input topologies parameter contains only one topology. Returning the same topology.")
                return topologies
        else:
            topologyList = Helper.Flatten(list(topologies))
            topologyList = [x for x in topologyList if Topology.IsInstance(x, "Topology")]
        if len(topologyList) == 0:
            if not silent:
                print("BVH.ByTopologies - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None
        # Recursive BVH construction
        def build_bvh(objects, depth=0):
            if len(objects) == 1:
                return BVH.BVHNode(objects[0].aabb, objects=objects)
            
            # Split objects along the median axis based on their centroids
            axis = depth % 3
            objects.sort(key=lambda obj: obj.centroid[axis])
            
            mid = len(objects) // 2
            left_bvh = build_bvh(objects[:mid], depth + 1)
            right_bvh = build_bvh(objects[mid:], depth + 1)

            # Merge left and right bounding boxes
            combined_aabb = BVH.AABB(
                np.minimum(left_bvh.aabb.min_point, right_bvh.aabb.min_point),
                np.maximum(left_bvh.aabb.max_point, right_bvh.aabb.max_point)
            )
            
            return BVH.BVHNode(combined_aabb, left_bvh, right_bvh)
        
        mesh_objects = []
        for topology in topologyList:
            vertices = [(Vertex.X(v), Vertex.Y(v), Vertex.Z(v)) for v in Topology.Vertices(topology)]
            mesh_objects.append(BVH.MeshObject(vertices, topology))
        
        return build_bvh(mesh_objects)
    
    @staticmethod
    def QueryByTopologies(*topologies, silent: bool = False):
        """
        Creates a BVH Query from the input list of topologies. The input can be individual topologies each as an input argument or a list of topologies stored in one input argument.

        Parameters
        ----------
        topologies : list
            The list of topologies.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        BVH query
            The created BVH query.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        if len(topologies) == 0:
            print("BVH.QueryByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
            return None
        if len(topologies) == 1:
            topologyList = topologies[0]
            if isinstance(topologyList, list):
                if len(topologyList) == 0:
                    if not silent:
                        print("BVH.QueryByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
                    return None
                else:
                    topologyList = [x for x in topologyList if Topology.IsInstance(x, "Topology")]
                    if len(topologyList) == 0:
                        if not silent:
                            print("BVH.QueryByTopologies - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
                        return None
            else:
                if not silent:
                    print("BVH.QueryByTopologies - Warning: The input topologies parameter contains only one topology. Returning the same topology.")
                return topologies
        else:
            topologyList = Helper.Flatten(list(topologies))
            topologyList = [x for x in topologyList if Topology.IsInstance(x, "Topology")]
        if len(topologyList) == 0:
            if not silent:
                print("BVH.ByTopologies - Error: The input parameters do not contain any valid topologies. Returning None.")
            return None
        vertices = []
        for topology in topologyList:
            if Topology.IsInstance(topology, "Vertex"):
                vertices.append(topology)
            else:
                vertices.extend(Topology.Vertices(topology))
        cluster = Cluster.ByTopologies(vertices)
        bb = Topology.BoundingBox(cluster)
        d = Topology.Dictionary(bb)
        x_min = Dictionary.ValueAtKey(d, "xmin")
        y_min = Dictionary.ValueAtKey(d, "ymin")
        z_min = Dictionary.ValueAtKey(d, "zmin")
        x_max = Dictionary.ValueAtKey(d, "zmax")
        y_max = Dictionary.ValueAtKey(d, "ymax")
        z_max = Dictionary.ValueAtKey(d, "zmax")
        query_aabb = BVH.AABB(min_point=(x_min, y_min, z_min), max_point=(x_max, y_max, z_max))
        return query_aabb
    
    def Clashes(bvh, query):
        """
        Returns a list of topologies in the input bvh tree that clashes (broad phase) with the list of topologies in the input query.

        Parameters
        ----------
        topologies : list
            The list of topologies.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        list
            The list of clashing topologies (based on their axis-aligned bounding box (AABB))

        """
        # Function to perform clash detection (broad-phase) and return Topologic objects
        def clash_detection(bvh_node, query_aabb, clashing_objects=None):
            if clashing_objects is None:
                clashing_objects = []
            
            # Check if the query AABB intersects with the current node's AABB
            if not bvh_node.aabb.intersects(query_aabb):
                return clashing_objects
            
            # If this is a leaf node, check each object in the node
            if bvh_node.objects:
                for obj in bvh_node.objects:
                    if obj.aabb.intersects(query_aabb):
                        clashing_objects.append(obj.topologic_object)  # Return the Topologic object
                return clashing_objects
            
            # Recursively check the left and right child nodes
            clash_detection(bvh_node.left, query_aabb, clashing_objects)
            clash_detection(bvh_node.right, query_aabb, clashing_objects)
            
            return clashing_objects
        return clash_detection(bvh, query)

    # Function to recursively add nodes and edges to the TopologicPy Graph
    def Graph(bvh, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a graph from the input bvh tree.

        Parameters
        ----------
        bvh : BVH Tree
            The input BVH Tree.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Graph
            The created graph.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        import random
        def add_bvh_to_graph(bvh_node, graph, parent_vertex=None, tolerance=0.0001):
            # Create a vertex for the current node's AABB centroid
            centroid = bvh_node.aabb.centroid
            current_vertex = Vertex.ByCoordinates(x=centroid[0], y=centroid[1], z=centroid[2])

            # Add an edge from the parent to this vertex (if a parent exists)
            if parent_vertex is not None:
                d = Vertex.Distance(parent_vertex, current_vertex)
                if d <= tolerance:
                    current_vertex = Topology.Translate(current_vertex, tolerance*random.uniform(2,50), tolerance*random.uniform(2,50), tolerance*random.uniform(2,50))
                edge = Edge.ByVertices(parent_vertex, current_vertex, tolerance=tolerance, silent=silent)
                graph = Graph.AddEdge(graph, edge, silent=silent)
            
            # Recursively add child nodes
            if bvh_node.left:
                graph = add_bvh_to_graph(bvh_node.left, graph, parent_vertex=current_vertex, tolerance=tolerance)
            if bvh_node.right:
                graph = add_bvh_to_graph(bvh_node.right, graph, parent_vertex=current_vertex, tolerance=tolerance)

            return graph
        graph = Graph.ByVerticesEdges([Vertex.Origin()], [])
        return add_bvh_to_graph(bvh, graph, parent_vertex = None, tolerance=tolerance)