import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

# Edge Class unit test

#Importing libraries
import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster

# Object for test case
v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
v3 = Vertex.ByCoordinates(0, -10, 0)        # create vertex
v4 = Vertex.ByCoordinates(5, 0, 0)          # create vertex
list_v = [v0,v1]                            # create list of vertices
list_v1 = [v1,v3]                           # create list of vertices
cluster_1 = Cluster.ByTopologies(list_v)    # create cluster of vertices
cluster_2 = Cluster.ByTopologies(list_v1)   # create cluster of vertices    
eB = Edge.ByStartVertexEndVertex(v0,v3)     # create edge

# Case 1 - Create an edge ByStartVertexEndVertex
# test 1
e1 = Edge.ByStartVertexEndVertex(v0,v1)                    # without tolerance
assert isinstance(e1, topologic.Edge), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"
# test 2
e1 = Edge.ByStartVertexEndVertex(v0,v1, tolerance=0.001)   # with tolerance (optional)
assert isinstance(e1, topologic.Edge), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"
# test 3
e2 = Edge.ByStartVertexEndVertex(v1, v3)                   # without tolerance
assert isinstance(e2, topologic.Edge), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"

# Case 2 - Create an edge ByVertices
# test 1
e3 = Edge.ByVertices(list_v)                    # without tolerance
assert isinstance(e3, topologic.Edge), "Edge.ByVertices. Should be topologic.Edge"
# test 2
e3 = Edge.ByVertices(list_v, tolerance=0.001)   # with tolerance (optional) 
assert isinstance(e3, topologic.Edge), "Edge.ByVertices. Should be topologic.Edge"
# test 3
e4 = Edge.ByVertices(list_v1, tolerance=0.001)  # with tolerance (optional) 
assert isinstance(e4, topologic.Edge), "Edge.ByVertices. Should be topologic.Edge"                                                  

# Case 3 - Create an edge ByVerticesCluster
# test 1
e5 = Edge.ByVerticesCluster(cluster_1)                  # without tolerance
assert isinstance(e5, topologic.Edge), "Edge.ByVerticesCluster. Should be topologic.Edge"
# test 2
e5 = Edge.ByVerticesCluster(cluster_1, tolerance=0.001) # with tolerance (optional)
assert isinstance(e5, topologic.Edge), "Edge.ByVerticesCluster. Should be topologic.Edge"
# test 3
e6 = Edge.ByVerticesCluster(cluster_2)                  # without tolerance
assert isinstance(e6, topologic.Edge), "Edge.ByVerticesCluster. Should be topologic.Edge"

# Case 4 - Angle
e7 = Edge.ByStartVertexEndVertex(v0,v4)         #create edge
# test 1
angle = Edge.Angle(e1,e7)                                # without optional inputs
assert isinstance(angle, float), "Edge.Angle. Should be float"
# test 2
angle = Edge.Angle(e1,e7,mantissa=2, bracket=True)       # with optional inputs
assert isinstance(angle, float), "Edge.Angle. Should be float"
# test 3
angle1 = Edge.Angle(e1,e5)                               # without optional inputs
assert isinstance(angle1, float), "Edge.Angle. Should be float"

# Case 5 - Bisect
# test 1
e_bis = Edge.Bisect(e1, eB)                                                # without optional inputs
assert isinstance(e_bis, topologic.Edge), "Edge.Bisect. Should be topologic.Edge" 
# test 2
e_bis = Edge.Bisect(e1, eB, length=1.0, placement=1, tolerance=0.001)      # with optional inputs
assert isinstance(e_bis, topologic.Edge), "Edge.Bisect. Should be topologic.Edge" 

# Case 6 - Direction
# test 1
direction = Edge.Direction(e2)                   # without optional inputs
assert isinstance(direction, list), "Edge.Direction. Should be list"
# test 2
direction1 = Edge.Direction(e1)                  # without optional inputs
assert isinstance(direction1, list), "Edge.Direction. Should be list"
# test 3
direction1 = Edge.Direction(e1,mantissa=3)       # with optional inputs
assert isinstance(direction1, list), "Edge.Direction. Should be list"

# Case 7 - EndVertex
# test 1
end2 = Edge.EndVertex(e2)
assert isinstance(end2, topologic.Vertex), "Edge.EndVertex. Should be topologic.Vertex"
# test 2
end3 = Edge.EndVertex(e3)
assert isinstance(end3, topologic.Vertex), "Edge.EndVertex. Should be topologic.Vertex"

# Case 8 - Extend
# test 1
extend2 = Edge.Extend(e2)                                                               # without optional inputs
assert isinstance(extend2, topologic.Edge), "Edge.Extend. Should be topologic.Edge"
# test 2
extend3 = Edge.Extend(e3)                                                               # without optional inputs
assert isinstance(extend3, topologic.Edge), "Edge.Extend. Should be topologic.Edge"
# test 3
extend3 = Edge.Extend(e3,distance=2, bothSides=False, reverse=True, tolerance=0.001)    # with optional inputs
assert isinstance(extend3, topologic.Edge), "Edge.Extend. Should be topologic.Edge"

# Case 9 - IsCollinear (True)
e5 = Edge.ByStartVertexEndVertex(v0,v3)
# test 1
col_1 = Edge.IsCollinear(e1,e5, mantissa=3)                                         # without optional inputs
assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"
# test 2
col_2 = Edge.IsCollinear(e1,e3, mantissa=3)                                         # without optional inputs
assert isinstance(col_2, bool), "Edge.IsCollinear. Should be bool"
# test 3
col_1 = Edge.IsCollinear(e1,e5, mantissa=3, angTolerance=0.01, tolerance=0.001)     # with optional inputs
assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"

# Case 10 - IsParallel
# test 1
par_1 = Edge.IsParallel(e1,e4)                                  # without optional inputs
assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"
# test 2
par_2 = Edge.IsParallel(e1,e3)                                  # without optional inputs
assert isinstance(par_2, bool), "Edge.IsParallel. Should be bool"
# test 3
par_1 = Edge.IsParallel(e1,e4, mantissa=2, angTolerance=0.01)   # with optional inputs
assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"

# Case 11 - Length
# test 1
len_1 = Edge.Length(e1)               # without optional inputs
assert isinstance(len_1, float), "Edge.Length. Should be float"
# test 2
len_2 = Edge.Length(e2)               # without optional inputs
assert isinstance(len_2, float), "Edge.Length. Should be float"
# test 3
len_1 = Edge.Length(e1, mantissa=3)   # with optional inputs
assert isinstance(len_1, float), "Edge.Length. Should be float"

# Case 12 - Normalize
# test 1
normal_3 = Edge.Normalize(e3)                     # without optional inputs
assert isinstance(normal_3, topologic.Edge), "Edge.Normalize. Should be topologic.Edge"
# test 2
normal_4 = Edge.Normalize(e4)                     # without optional inputs
assert isinstance(normal_4, topologic.Edge), "Edge.Normalize. Should be topologic.Edge"
# test 3
normal_4 = Edge.Normalize(e4, useEndVertex=True)  # with optional inputs
assert isinstance(normal_4, topologic.Edge), "Edge.Normalize. Should be topologic.Edge"

# Case 13 - ParameterAtVertex
# test 1
param1 = Edge.ParameterAtVertex(e2,v1)              # without optional inputs
assert isinstance(param1, float), "Edge.ParameterAtVertex. Should be float"
# test 2
param2 = Edge.ParameterAtVertex(e1,v1, mantissa=3)  # with optional inputs
assert isinstance(param2, float), "Edge.ParameterAtVertex. Should be float"

# Case 14 - Reverse
# test 1
reverse3 = Edge.Reverse(e3)
assert isinstance(reverse3, topologic.Edge), "Edge.Reverse. Should be topologic.Edge"
# test 2
reverse4 = Edge.Reverse(e4)
assert isinstance(reverse4, topologic.Edge), "Edge.Reverse. Should be topologic.Edge"

# Case 15 - SetLength
# test 1
SetLen1 = Edge.SetLength(e1)                                                            # without optional inputs
assert isinstance(SetLen1, topologic.Edge), "Edge.SetLength. Should be topologic.Edge"
# test 2
SetLen2 = Edge.SetLength(e2)                                                            # without optional inputs
assert isinstance(SetLen2, topologic.Edge), "Edge.SetLength. Should be topologic.Edge"
# test 3
SetLen1 = Edge.SetLength(e1, length=2, bothSides=False, reverse=True, tolerance=0.001)  # with optional inputs
assert isinstance(SetLen1, topologic.Edge), "Edge.SetLength. Should be topologic.Edge"

# Case 16 - StartVertex
# test 1
iV = Edge.StartVertex(e1)
assert isinstance(iV, topologic.Vertex), "Edge.StartVertex. Should be topologic.Vertex"
# test 2
iV1 = Edge.StartVertex(e2)
assert isinstance(iV1, topologic.Vertex), "Edge.StartVertex. Should be topologic.Vertex"

# Case 17 - Trim
# test 1
trim3 = Edge.Trim(e3)                                                               # without optional inputs
assert isinstance(trim3, topologic.Edge), "Edge.Trim. Should be topologic.Edge"
# test 2
trim4 = Edge.Trim(e4)                                                               # without optional inputs
assert isinstance(trim4, topologic.Edge), "Edge.Trim. Should be topologic.Edge"
# test 3
trim4 = Edge.Trim(e4, distance=1, bothSides=False, reverse=True, tolerance=0.001)   # with optional inputs
assert isinstance(trim4, topologic.Edge), "Edge.Trim. Should be topologic.Edge"

# Case 18 - VertexByDistance
# test 1
dist1 = Edge.VertexByDistance(e1)                                           # without optional inputs
assert isinstance(dist1, topologic.Vertex), "Edge.VertexByDistance. Should be topologic.Vertex"
# test 2
dist2 = Edge.VertexByDistance(e2)                                           # without optional inputs
assert isinstance(dist2, topologic.Vertex), "Edge.VertexByDistance. Should be topologic.Vertex"
# test 3
dist2 = Edge.VertexByDistance(e2, distance=1, origin=v3, tolerance=0.001)   # with optional inputs
assert isinstance(dist2, topologic.Vertex), "Edge.VertexByDistance. Should be topologic.Vertex"

# Case 19 - VertexByParameter
# test 1
ByParam3 = Edge.VertexByParameter(e3)                  # without optional inputs
assert isinstance(ByParam3, topologic.Vertex), "Edge.VertexByParameter. Should be topologic.Vertex"
# test 2
ByParam4 = Edge.VertexByParameter(e4)                  # without optional inputs
assert isinstance(ByParam4, topologic.Vertex), "Edge.VertexByParameter. Should be topologic.Vertex"
# test 3
ByParam4 = Edge.VertexByParameter(e4, parameter=0.7)   # with optional inputs
assert isinstance(ByParam4, topologic.Vertex), "Edge.VertexByParameter. Should be topologic.Vertex"

#Case 20 - Vertices
# test 1
v_e5 = Edge.Vertices(e5)
assert isinstance(v_e5, list), "Edge.Vertices. Should be list"
# test 2
v_e6 = Edge.Vertices(e6)
assert isinstance(v_e6, list), "Edge.Vertices. Should be list"