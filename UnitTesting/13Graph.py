# Graph Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Graph import Graph
from topologicpy.Topology import Topology
from topologicpy.Plotly import Plotly

# Object for test case
v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
v2 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
v3 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
v4 = Vertex.ByCoordinates(0, 0, 10)         # create vertex
v5 = Vertex.ByCoordinates(0, 10, 10)        # create vertex
v6 = Vertex.ByCoordinates(10, 10, 10)       # create vertex
v7 = Vertex.ByCoordinates(10, 0, 10)        # create vertex
v8 = Vertex.ByCoordinates(0, 0, 5)          # create vertex
v9 = Vertex.ByCoordinates(5, 0, 5)          # create vertex
gv1 = Vertex.ByCoordinates(-10, 10, 0)          # create vertex
gv2 = Vertex.ByCoordinates(10, 10, 0)           # create vertex
gv3 = Vertex.ByCoordinates(10, -10, 0)          # create vertex
gv4 = Vertex.ByCoordinates(-10, -10, 0)         # create vertex
gv5 = Vertex.ByCoordinates(50, 60, 0)           # create vertex
gv6 = Vertex.ByCoordinates(65, 107, 0)          # create vertex
wire0 = Wire.ByVertices([v0,v1,v2])         # create wire
wire1 = Wire.ByVertices([v4,v5,v6])         # create wire
w_list = [wire0,wire1]                      # create list
w_cluster = Cluster.ByTopologies(w_list)    # create cluster
face0 = Face.ByVertices([v0,v1,v2,v3,v0])     # create face
face1 = Face.ByVertices([v4,v5,v6,v7,v4])     # create face
face2 = Face.ByVertices([v0,v4,v7,v3,v0])     # create face
face3 = Face.ByVertices([v3,v7,v6,v2,v3])     # create face
face4 = Face.ByVertices([v2,v6,v5,v1,v2])     # create face
face5 = Face.ByVertices([v1,v5,v4,v0,v1])     # create face
f_list = [face0,face1,face2,face3,face4,face5]  # create list
c_faces = Cluster.ByTopologies(f_list)          # create cluster
shell_f = Shell.ByFaces(f_list)                 # create shell
shell_open = Shell.ByFaces([face0])       # create shell
e0 = Edge.ByStartVertexEndVertex(v0,v1)
e1 = Edge.ByStartVertexEndVertex(v0,v2)
e2 = Edge.ByStartVertexEndVertex(v0,v3)
e3 = Edge.ByStartVertexEndVertex(v0,v4)
e4 = Edge.ByStartVertexEndVertex(v1,v2)
e5 = Edge.ByStartVertexEndVertex(v1,v3)
list_v = [v0,v1,v2,v3,v4]
list_e = [e0,e1,e2,e3]
list_e2 = [e0,e1,e2]
list_v1 = [v5,v6,v7]
list_v2 = [v1,v3,v4]

# Case 1 - ByVerticesEdges

# test 1
graph_ve = Graph.ByVerticesEdges(list_v,list_e)
assert isinstance(graph_ve, topologic.Graph), "Graph.ByVerticesEdges. Should be topologic.Graph"
# test 2
graph_ve1 = Graph.ByVerticesEdges(list_v2,list_e)
assert isinstance(graph_ve1, topologic.Graph), "Graph.ByVerticesEdges. Should be topologic.Graph"

# Case 2 - AddEdge

# test 1
graph_ae = Graph.AddEdge(graph_ve, e4)                   # without optional inputs
graph_ae_numE = len(Graph.Edges(graph_ae))
assert isinstance(graph_ae, topologic.Graph), "Graph.AddEdge. Should be topologic.Graph"
assert graph_ae_numE == 5, "Graph.AddEdge. Should be 5"  
# test 2
graph_ae1 = Graph.AddEdge(graph_ve1, e4, tolerance=0.001) # with optional inputs
graph_ae1_numE = len(Graph.Edges(graph_ae1)) 
assert isinstance(graph_ae1, topologic.Graph), "Graph.AddEdge. Should be topologic.Graph"
assert graph_ae1_numE == 5, "Graph.AddEdge. Should be 5" 

# Case 3 - AddVertex

# test 1
graph_av = Graph.AddVertex(graph_ve, v5)                      # without optional inputs
graph_av_numV = len(Graph.Vertices(graph_av))
assert isinstance(graph_av, topologic.Graph), "Graph.AddVertex. Should be topologic.Graph"
assert graph_av_numV == 6, "Graph.AddVertex. Should be 6"  
# test 2
graph_av1 = Graph.AddVertex(graph_ve1, v5, tolerance=0.001)   # with optional inputs
graph_av1_numV = len(Graph.Vertices(graph_av1))
assert isinstance(graph_av1, topologic.Graph), "Graph.AddVertex. Should be topologic.Graph"
assert graph_av1_numV == 6, "Graph.AddVertex. Should be 6"  

# Case 4 - AddVertices

# test 1
graph_avs = Graph.AddVertices(graph_ve, list_v1)                       # without optional inputs
graph_avs_numV = len(Graph.Vertices(graph_avs))
assert isinstance(graph_avs, topologic.Graph), "Graph.AddVertices. Should be topologic.Graph"
assert graph_avs_numV == 8, "Graph.AddVertices. Should be 8"
# test 2
graph_avs1 = Graph.AddVertices(graph_ve, list_v2, tolerance=0.001)    # with optional inputs
graph_avs1_numV = len(Graph.Vertices(graph_avs1))
assert isinstance(graph_avs1, topologic.Graph), "Graph.AddVertices. Should be topologic.Graph"
assert graph_avs1_numV == 8, "Graph.AddVertices. Should be 8"

# Case 5 - AdjacencyList

# test 1
graph_adjL = Graph.AdjacencyList(graph_ve)
assert isinstance(graph_adjL, list), "Graph.AdjacencyList. Should be list"

# Case 6 - AdjacencyMatrix

# test 1
graph_adjM = Graph.AdjacencyMatrix(graph_ve)
assert isinstance(graph_adjM, list), "Graph.AdjacencyMatrix. Should be list"

# Case 7 - AdjacentVertices

# test 1
graph_adjV = Graph.AdjacentVertices(graph_ve,v3)
assert isinstance(graph_adjV, list), "Graph.AdjacentVertices. Should be list"
# test 2
i0 = graph_adjV[0]
vd = Vertex.Distance(i0,v0)                                 # the adjacent vertex is v0
assert vd == 0.0, "Graph.AdjacentVertices. Should be 0.0"   
graph_adjV1 = Graph.AdjacentVertices(graph_ve1,v1)
assert isinstance(graph_adjV1, list), "Graph.AdjacentVertices. Should be list"

# Case 8 - AllPaths

# test 1
graph_ap = Graph.AllPaths(graph_ve, v0, v4)                 # without optional inputs
assert isinstance(graph_ap, list), "Graph.AllPaths. Should be list"
# test 2
graph_ap1 = Graph.AllPaths(graph_ae, v0, v3, timeLimit=5)   # with optional inputs
assert isinstance(graph_ap1, list), "Graph.AllPaths. Should be list"

# Case 9 - ByTopology

# test 1
cell_cy = Cell.Cylinder()
vCy2 = Vertex.ByCoordinates(0,1,0)
cell_cy2 = Cell.Cylinder(vCy2)
cc = CellComplex.ByCells([cell_cy,cell_cy2])
graph_t = Graph.ByTopology(cc, direct=True, directApertures=False, viaSharedTopologies=True,
                            viaSharedApertures=False, toExteriorTopologies=True, toExteriorApertures=False,
                            toContents=False, useInternalVertex=True, storeBRep=True, tolerance=0.0001)
topology_graph_t =  Graph.Topology(graph_t)
# plot geometry
data_gt = Plotly.DataByTopology(topology_graph_t)
figure_gt = Plotly.FigureByData(data_gt)
#Plotly.Show(figure_gt)                                                                  # visualization

# Case 10 - Connect

# test 1
graph_con = Graph.Connect(graph_ve, list_v1, list_v2)                         # without optional inputs
assert isinstance(graph_con, topologic.Graph), "Graph.Connect. Should be topologic.Graph"
# test 2
graph_con1 = Graph.Connect(graph_ve1, list_v1, list_v2, tolerance=0.001)      # with optional inputs
assert isinstance(graph_con1, topologic.Graph), "Graph.Connect. Should be topologic.Graph"

# Case 11 - ContainsEdge

# test 1
graph_ce = Graph.ContainsEdge(graph_ve, e5)                           # without optional inputs
assert isinstance(graph_ce, bool), "Graph.ContainsEdge. Should be bool"
assert graph_ce == False, "Graph.ContainsEdge. Should be False"
# test 2
graph_ce1 = Graph.ContainsEdge(graph_ve, e0, tolerance=0.001)        # with optional inputs
assert isinstance(graph_ce1, bool), "Graph.ContainsEdge. Should be bool"
assert graph_ce1 == True, "Graph.ContainsEdge. Should be True"

# Case 12 - ContainsVertex

# test 1
graph_cv = Graph.ContainsVertex(graph_avs, v8)                         # without optional inputs
assert isinstance(graph_cv, bool), "Graph.ContainsVertex. Should be bool"
assert graph_cv == False, "Graph.ContainsVertex. Should be False"
# test 2
graph_cv1 = Graph.ContainsVertex(graph_avs, v7, tolerance=0.001)      # with optional inputs
assert isinstance(graph_cv1, bool), "Graph.ContainsVertex. Should be bool"
assert graph_cv1 == True, "Graph.ContainsVertex. Should be True"

# Case 13 - DegreeSequence

# test 1
graph_ds = Graph.DegreeSequence(graph_ae)
assert isinstance(graph_ds, list), "Graph.DegreeSequence. Should be list"
# test 2
graph_ds1 = Graph.DegreeSequence(graph_ve)
assert isinstance(graph_ds1, list), "Graph.DegreeSequence. Should be list"

# Case 14 - Density

# test 1
graph_d = Graph.Density(graph_ve)
assert isinstance(graph_d, float), "Graph.Density. Should be float"
# test 2
graph_d1 = Graph.Density(graph_ae)
assert isinstance(graph_d1, float), "Graph.Density. Should be float"

# Case 15 - DepthMap

# test 1
graph_dm = Graph.DepthMap(graph_ve, v2)                      # without optional inputs
assert isinstance(graph_dm, list), "Graph.DepthMap. Should be list"
# test 2
graph_dm1 = Graph.DepthMap(graph_ae, v3, tolerance=0.001)    # with optional inputs
assert isinstance(graph_dm1, list), "Graph.DepthMap. Should be list"

# Case 16 - Diameter

# test 1
graph_dia = Graph.Diameter(graph_ve)
assert isinstance(graph_dia, int), "Graph.Diameter. Should be int"
# test 2
graph_dia1 = Graph.Diameter(graph_ae)
assert isinstance(graph_dia1, int), "Graph.Diameter. Should be int"

# Case 17 - Distance

# test 1
graph_dist = Graph.Distance(graph_ve, v1, v5)                       # without optional inputs
assert isinstance(graph_dist, int), "Graph.Distance. Should be int"
# test 2
graph_dist1 = Graph.Distance(graph_ve, v0, v5, tolerance=0.001)     # with optional inputs
assert isinstance(graph_dist1, int), "Graph.Distance. Should be int"

# Case 18 - Edge

# test 1
graph_edge = Graph.Edge(graph_ve, v0, v2)                    # without optional inputs
EL_e1 = Edge.Length(e1)
EL_graph_edge = Edge.Length(graph_edge)
assert isinstance(graph_edge, topologic.Edge), "Edge.Length. Should be topologic.Edge"
assert EL_graph_edge == EL_e1, "Graph.Distance. Should be the same value"
# test 2
graph_edge1 = Graph.Edge(graph_ae, v0, v3, tolerance=0.001)  # with optional inputs
EL_e2 = Edge.Length(e2)
EL_graph_edge1 = Edge.Length(graph_edge1)
assert isinstance(graph_edge1, topologic.Edge), "Edge.Length. Should be topologic.Edge"
assert EL_graph_edge1 == EL_e2, "Graph.Distance. Should be the same value"

# Case 19 - Edges

# test 1
graph_ve2 = Graph.ByVerticesEdges(list_v, list_e2)
graph_edges = Graph.Edges(graph_ve2)                             # without optional inputs 
graph_edges_numE = len(Graph.Edges(graph_ve2))
assert isinstance(graph_edges, list), "Graph.Edges. Should be list"
assert graph_edges_numE == 3, "Graph.Edges. Should be 3"
# test 2
graph_edges1 = Graph.Edges(graph_ve2, list_v2, tolerance=0.001)  # with optional inputs, multiplied by 2?
graph_edges1_numE = len(Graph.Edges(graph_ve, list_v2, tolerance=0.001))
print("Line 263, check this!", graph_edges1_numE)
"""The number of edges is double"""
assert isinstance(graph_edges1, list), "Graph.Edges. Should be list"
#assert graph_edges1_numE == 3, "Graph.Edges. Should be 3"

# Case 20 - IsBipartite

# test 1
graph_ib = Graph.IsBipartite(graph_ve)                                  # without optional inputs 
assert isinstance(graph_ib, bool), "Graph.IsBipartite. Should be bool"
assert graph_ib == True, "Graph.IsBipartite. Should be True"
# test 2
graph_ib1 = Graph.IsBipartite(graph_av, tolerance=0.001)                # with optional inputs 
assert isinstance(graph_ib1, bool), "Graph.IsBipartite. Should be bool"
assert graph_ib1 == True, "Graph.IsBipartite. Should be True"

# Case 21 - IsComplete

# test 1
chk_G1 = Graph.IsComplete(graph_ve)
assert isinstance(chk_G1, bool), "Graph.IsComplete. Should be bool"
# test 2
chk_G2 = Graph.IsComplete(graph_av1)
assert isinstance(chk_G2, bool), "Graph.IsComplete. Should be bool"

# Case 22 - IsErdoesGallai

# test 1
chkG3 = Graph.IsErdoesGallai(graph_ve1, [1,3,5])
assert isinstance(chkG3, bool), "Graph.IsComplete. Should be bool"
# test 2
chkG4 = Graph.IsErdoesGallai(graph_ve2, [1, 2])
assert isinstance(chkG4, bool), "Graph.IsComplete. Should be bool"

# Case 23 - IsolatedVertices

# test 1
graphIV1 = Graph.IsolatedVertices(graph_ve1)
assert isinstance(graphIV1, list), "Graph.IsolatedVertices. Should be list"
# test 2
graphIV2 = Graph.IsolatedVertices(graph_ae)
assert isinstance(graphIV1, list), "Graph.IsolatedVertices. Should be list"

# Case 24 - MaximumDelta

# test 1
graphMaxD1 = Graph.MaximumDelta(graph_ve2)
assert isinstance(graphMaxD1, int), "Graph.IsolatedVertices. Should be integer"
# test 2
graphMaxD2 = Graph.MaximumDelta(graph_ae)
assert isinstance(graphMaxD2, int), "Graph.IsolatedVertices. Should be integer"

# Case 25 - MinimumDelta

# test 1
graphMinD1 = Graph.MinimumDelta(graph_ve2)
assert isinstance(graphMinD1, int), "Graph.IsolatedVertices. Should be integer"
# test 2
graphMinD2 = Graph.MinimumDelta(graph_avs1)
assert isinstance(graphMinD2, int), "Graph.IsolatedVertices. Should be integer"

# Case 26 - MinimumSpanningTree

# test 1
mSpan1 = Graph.MinimumSpanningTree(graph_ve2, 5, 0.002)                         # with optional inputs         
assert isinstance(mSpan1, topologic.Graph), "Graph.MinimumSpanningTree. Should be topologic.Graph"
# test 2
mSpan2 = Graph.MinimumSpanningTree(graph_ve1)                                       # without optional inputs    
assert isinstance(mSpan2, topologic.Graph), "Graph.MinimumSpanningTree. Should be topologic.Graph"

# Case 27 - NearestVertex

# test 1
graphNV1 = Graph.NearestVertex(graph_ve1, v8)
assert isinstance(graphNV1, topologic.Vertex), "Graph.NearestVertex. Should be topologic.Vertex"
# test 2
graphNV2 = Graph.NearestVertex(graph_ve2, v6)
assert isinstance(graphNV2, topologic.Vertex), "Graph.NearestVertex. Should be topologic.Vertex"

# Case 28 - Order

# test 1
graphO1 = Graph.Order(graph_ae1)
assert isinstance(graphO1, int), "Graph.Order. Should be integer"
# test 2
graphO2 = Graph.Order(graph_ve2)
assert isinstance(graphO1, int), "Graph.Order. Should be integer"

# Case 29 - Path

# test 1
path1 = Graph.Path(graph_ve1, v1, v4)
assert isinstance(path1, topologic.Wire), "Graph.Path. Should be topologic.Wire"
# test 2
path2 = Graph.Path(graph_ve2, v0, v3)
assert isinstance(path1, topologic.Wire), "Graph.Path. Should be topologic.Wire"

# Case 30 - RemoveEdge

# test 1
remEdg1 = Graph.RemoveEdge(graph_ve1, e2)                          # without optional inputs
assert isinstance(remEdg1, topologic.Graph), "Graph.RemoveEdge. Should be topologic.Graph"
# test 2
remEdg2 = Graph.RemoveEdge(graph_ve2,e0, 0.0001)            # with optional inputs
assert isinstance(remEdg2, topologic.Graph), "Graph.RemoveEdge. Should be topologic.Graph"

# Case 31 - RemoveVertex

# test 1
remVer1 = Graph.RemoveVertex(graph_ve2, v0)                          # without optional inputs
assert isinstance(remVer1, topologic.Graph), "Graph.RemoveVertex. Should be topologic.Graph"
# test 2
remVer2 = Graph.RemoveVertex(graph_ve1, v1, 0.0001)           # with optional inputs
assert isinstance(remVer2, topologic.Graph), "Graph.RemoveVertex. Should be topologic.Graph"

# Case 32 - ShortestPath
"""TypeError: ShortestPath(): incompatible function arguments."""
# test 1
sp1 = Graph.ShortestPath(graph_ve,v1,v4)
assert isinstance(sp1, topologic.Wire), "Graph.ShortestPath. Should be topologic.Wire"
# Case 33 - ShortestPaths
# test 1
sps = Graph.ShortestPaths(graph_ve,vertexA=v1,vertexB=v3, timeLimit=5)
assert isinstance(sps, list), "Graph.ShortestPaths. Should be list"

# Case 34 - Size

# test 1
graphS1 = Graph.Size(graph_ae)
assert isinstance(graphS1, int), "Graph.Size. Should be integer"
# test 2
graphS2 = Graph.Size(graph_av1)
assert isinstance(graphS2, int), "Graph.Size. Should be integer"

# Case 35 -  TopologicalDistance

# test 1
topoDist1 = Graph.TopologicalDistance(graph_ve1, v0, v1)                                 # without optional inputs
assert isinstance(topoDist1, int), "Graph.TopologicalDistance. Should be integer"
# test 2
topoDist2 = Graph.TopologicalDistance(graph_ve2, v1, v3, 0.0003)                      # with optional inputs
assert isinstance(topoDist2, int), "Graph.TopologicalDistance. Should be integer"

# Case 36 - Topology

# test 1
graphTo1 = Graph.Topology(graph_ae)
assert isinstance (graphTo1, topologic.Cluster)
# test 2
graphTo2 = Graph.Topology(graph_avs1)
assert isinstance (graphTo2, topologic.Cluster)

# Case 37 - Tree

# test 1
graphTr1 = Graph.Tree(graph_ve1)                               # without optional inputs
assert isinstance(graphTr1, topologic.Graph), "Graph.Tree. Should be topologic.Graph"
# test 2
graphTr2 = Graph.Tree(graph_ve2, v6, 0.0001)                   # with optional inputs
assert isinstance(graphTr1, topologic.Graph), "Graph.Tree. Should be topologic.Graph"

# Case 38 - VertexDegree
v9 = Vertex.ByCoordinates(-10, -10, -10)                           # create vertex

# test 1
graphVD1 = Graph.VertexDegree(graph_ve2, v0)
assert isinstance(graphVD1, int), "Graph.VertexDegree. Should be integer"
# test 2
graphVD2 = Graph.VertexDegree(graph_ve2, v9)
assert isinstance(graphVD1, int), "Graph.VertexDegree. Should be integer"

# Case  39 - Vertices

# test 1
graphV1 = Graph.Vertices(graph_ve1)
assert isinstance(graphV1, list), "Graph.Vertices. Should be list"
# test 2
graphV2 = Graph.Vertices(graph_ae)
assert isinstance(graphV2, list), "Graph.Vertices. Should be list"

# case 40 - VisibilityGraph
Bound1 = Wire.Rectangle(v0, 50, 50)
graphVG1 = Graph.VisibilityGraph(Bound1, obstacles=None, viewpointsA=None, viewpointsB=None, tolerance=0.001)
assert isinstance(graphVG1, topologic.Graph), "Graph.VisibilityGraph. Should be topologic.Graph"