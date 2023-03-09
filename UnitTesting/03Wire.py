# Wire Classes unit test

#Importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

import topologicpy 
import topologic
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cluster import Cluster
from topologicpy.Plotly import Plotly

#Objects for test case
# Creating vertices by coordinates
v1 = Vertex.ByCoordinates(0, 0, 0)                # create vertex
v2 = Vertex.ByCoordinates(5, 5, 5)                # create vertex
v3 = Vertex.ByCoordinates(5, 10, 10)            # create vertex
v4 = Vertex.ByCoordinates(10, 15, 15)          # create vertex
v5 = Vertex.ByCoordinates(-2, 2, 0)              # create vertexByEdges
v6 = Vertex.ByCoordinates(-2, -2, 0)             # create vertex
v7 = Vertex.ByCoordinates(2, -2, 0)              # create vertex
v8= Vertex.ByCoordinates(2, 2, 0)                # create vertex
v9 = Vertex.ByCoordinates(1, 1, 0)               # create vertex
v10 = Vertex.ByCoordinates(-1, -1, 0)          # create vertex
v11 = Vertex.ByCoordinates(-4, -2, 0)          # create vertex
v12 = Vertex.ByCoordinates(2, -4, 0)           # create vertex

# Creating edges by vertices
e1 = Edge.ByVertices([v1, v2])                      # create edge
e2 = Edge.ByVertices([v2, v3])                      # create edge
e3 = Edge.ByVertices([v3, v4])                      # create edge
e4 = Edge.ByVertices([v4, v1])                      # create edge
e5 = Edge.ByVertices([v1, v2, v3, v4])           # create edge

# Case 1 - BoundingRectangle
Star = Wire.Star()                                                          # create star
cir3 = Wire.Circle()                                                       # create circle
# test 1
bRec1 = Wire.BoundingRectangle(Star)                       # without optional inputs
assert isinstance(bRec1, topologic.Wire), "Wire.BoundingRectangle. Should be topologic.Wire"
# test 2
bCir1 = Wire.BoundingRectangle(cir3, 5)                     # with optional inputs
assert isinstance(bCir1, topologic.Wire), "Wire.BoundingRectangle. Should be topologic.Wire"

# Case 2 - ByEdges
# test 1
w1 = Wire.ByEdges([e1,e2])                   
assert isinstance(w1, topologic.Wire), "Wire.ByEdges. Should be topologic.Wire"
# test 2
w2 = Wire.ByEdges([e1, e2, e3])            
assert isinstance(w2, topologic.Wire), "Wire.ByEdges. Should be topologic.Wire"

# Case 3 - ByEdgesCluster
#clE = cluster_Edges, clw = cluster_wire
clE1 = Cluster.ByTopologies([e1, e2, e3])       # create cluster
clE2 = Cluster.ByTopologies([e3, e4, e1])       # create cluster
# test 1
clw1 = Wire.ByEdgesCluster(clE1)                
assert isinstance(clw1, topologic.Wire), "Wire.ByEdgesCluster. Should be topologic.Wire"
# test 2
clw2 = Wire.ByEdgesCluster(clE2)                
assert isinstance(clw2, topologic.Wire), "Wire.ByEdgesCluster. Should be topologic.Wire"

# Case 4 - ByOffset
"""Error : Gives more output then expected if optional inputs used"""
# creating objects
rec1 = Wire.Rectangle(v1, 5.0, 5.0)                                    # create wire
Cir0 = Wire.Circle(v1, 5, 16)                                               # create wire
# test 1
offR1 = Wire.ByOffset(rec1)                                              # without optional inputs
assert isinstance(offR1, topologic.Wire),"Wire.ByOffset. Should be topologic.Wire"
# test 2        
offC1 = Wire.ByOffset(Cir0, offset=1, miter=True, miterThreshold=.5,                    # with optional inputs
                                    offsetKey='offCircleEdg', miterThresholdKey='offCircleVer', step=False)                                                                                                                     
assert isinstance(offC1, topologic.Wire),"Wire.ByOffset. Should be topologic.Wire"

# Case 5 - ByVertices
# test 1
w3 = Wire.ByVertices([v1,v2,v3,v4], False)                                              # with optional inputs
assert isinstance(w3, topologic.Wire), "Wire.ByVertices. Should be topologic.wire"
# test 2
w4 = Wire.ByVertices([v2, v3, v4])                                                          # without optional inputs
assert isinstance(w4, topologic.Wire), "Wire.ByVertices. Should be topologic.wire"
 
# Case 6 - ByVerticesCluster
# clV = Cluster_Vertices, Creating Cluster of vertices
clV1 = Cluster.ByTopologies([v1,v2,v3,v4])                  # create cluster
clV2 = Cluster.ByTopologies([v5, v6, v7, v8])               # create cluster
# test 1
clw3 = Wire.ByVerticesCluster(clV1)                            # without optional inputs
assert isinstance(clw3, topologic.Wire), "Wire.ByVerticesCluster. Should be topologic.Wire"
# test 2
clw4 = Wire.ByVerticesCluster(clV2, close=False)        # with optional inputs
assert isinstance(clw4, topologic.Wire), "Wire.ByVerticesCluster. Should be topologic.Wire"

# Case 7 - Circle
# test 1
Cir1 = Wire.Circle()                                                                                                  # without optional inputs
assert isinstance(Cir1, topologic.Wire), "Wire.CirclE. Should be topologic.Wire"
# test 2
Cir2 = Wire.Circle(origin=v1, radius=3, sides=21, fromAngle=30, toAngle=360,  # with optional inputs
                             close=False, direction=[0,1,1], placement='center', tolerance=0.0001)               
assert isinstance(Cir2, topologic.Wire), "Wire.CirclE. Should be topologic.Wire"    
 
# Case 8 - Cycles
# test 1
Cyc1 = Wire.Cycles(w4)                                      # without optional inputs
assert isinstance(Cyc1, list), "Wire.Cycles. Should be List"
# test 2
w5 = Wire.ByEdges([e1, e2, e3, e4])                    # create wire
Cyc2 = Wire.Cycles(w5, 6, 0.0001)                       # with optional inputs
assert isinstance(Cyc2, list), "Wire.Cycles. Should be List"
 
# Case 9 - Edges
# test 1
Edg1 = Wire.Edges(w3)
assert isinstance(Edg1, list), "Wire.Edges. Should be list"
# test 2
Edg2 = Wire.Edges(w5)
assert  isinstance(Edg2, list), "Wire.Edges. Should be list"
 
# Case 10 - Ellipse
# test 1
Elp1 = Wire.Ellipse()                                                                                                                                             # without optional inputs
assert isinstance(Elp1, topologic.Wire), "Wire.Ellipse. Should be topologic.Wire"
# test 2
Elp2 = Wire.Ellipse(origin=None, inputMode=1, width=2.0, length=1.0, focalLength=0.866025, 
                    eccentricity=0.866025, majorAxisLength=1.0, minorAxisLength=0.5, sides=32, fromAngle=0,
                    toAngle=360, close=True, direction=[0 ,1, 1], placement='center', tolerance=0.0001)            # with optional inputs
assert isinstance(Elp2, topologic.Wire), "Wire.Ellipse. Should be topologic.Wire"
# test 3
Elp3 = Wire.Ellipse(v2, 2, 3.5, 2.5, 0.866025, 0.866025, 1.0, 1, 16, 15, 270, [1, 0, 1], 
                    placement='lowerleft', tolerance= 0.0001)                                                                                       # with optional input
assert isinstance(Elp3, topologic.Wire), "Wire.Ellipse. Should be topologic.Wire"

# Case 11 - EllipseAll
# test 1
Elp3 = Wire.EllipseAll(v3, 3, 3, 7, 0.4, 0.6, 2.0, 1, 16)                                                                                              # with optional inputs
assert isinstance(Elp3, dict), "Wire.Ellipse. Should be dictionary"
# test 2
Elp4 = Wire.EllipseAll(origin=v3, width=2.5, length=5, sides=13, close=True, direction=[1, 0, 0])                          # with optional inputs
assert isinstance(Elp4, dict), "Wire.Ellipse. Should be dictionary"
# test 3
Elp5 = Wire.EllipseAll()                                                                                                                                         # without optional inputs
assert isinstance(Elp5, dict), "Wire.Ellipse. Should be dictionary"

# Case 12 - IsClosed
# test 1
Chk_W1 = Wire.IsClosed(w1)
assert isinstance(Chk_W1, bool), "Wire.Ellipse. Should be boolean"
# test 2
Chk_W2 = Wire.IsClosed(w5)
assert isinstance(Chk_W2, bool), "Wire.Ellipse. Should be boolean"

# Case 13 - IsSimilar
# creating Wire
Rec1 = Wire.Rectangle(v1, 2, 3)         # create wire
Rec2 = Wire.Rectangle(v2, 2, 3)         # create wire
Rec3 = Wire.Rectangle(v3, 3, 2)         # create wire
# test 1
Chk_W3 = Wire.IsSimilar(Rec1, Rec2)                                   # without optional inputs
assert isinstance(Chk_W3, bool), "Wire.IsSimilar. Should be boolean"
# test 2
Chk_W4 = Wire.IsSimilar(Rec2,Rec3, 0.0005, 0.2)                 # with optional inputs
assert isinstance(Chk_W4, bool), "Wire.IsSimilar. Should be boolean"
# test 3
Chk_W5 = Wire.IsSimilar(Rec1, Cir1, 0.01, 0.5)                      # with optional inputs
assert isinstance(Chk_W5, bool), "Wire.IsSimilar. Should be boolean"

# Case 14 - Isovist
# boundary Rectangle
bRec = Wire.Rectangle(v1, 20, 20)                           # create wire
# Vertices
obV1 = Vertex.ByCoordinates(-1, 4, 0)                    # create vertex
obV2 = Vertex.ByCoordinates(-4, 1, 0)                    # create vertex
obV3 = Vertex.ByCoordinates(-4, -1, 0)                  # create vertex
obV4 = Vertex.ByCoordinates(-1, -4, 0)                  # create vertex
obV5 = Vertex.ByCoordinates(1, -4, 0)                   # create vertex
obV6 = Vertex.ByCoordinates(4, -1, 0)                   # create vertex
obV7 = Vertex.ByCoordinates(-4, 0, 0)                   # create vertex
obV8 = Vertex.ByCoordinates(0, -4, 0)                   # create vertex
obV9 = Vertex.ByCoordinates(4, 0, 0)                    # create vertex
obV10 = Vertex.ByCoordinates(0, 4, 0)                  # create vertex
# obstacles Wire
obW1 = Wire.ByVertices([obV1, obV2, obV3])      # create vertex
obW2 = Wire.ByVertices([obV4, obV5, obV6])      # create vertex
obW3 = Wire.ByVertices([obV2, obV7, obV3])      # create wire
obW4 = Wire.ByVertices([obV4, obV8, obV5])      # create wire
obW5 = Wire.ByVertices([obV6, obV9, obV1])      # create wire
# obstaclesCluster
ObsW_C1 = Cluster.ByTopologies([obW1, obW2])                      # create cluster
ObsW_C2 = Cluster.ByTopologies([obW3, obW4, obW5])          # create cluster
# test 1
isoV1 = Wire.Isovist(v1, bRec, ObsW_C1)                                    # without optional inputs
assert isinstance(isoV1, list), "Wire.Isovist. Should be list"
# test 2
isoV2 = Wire.Isovist(v1, bRec, ObsW_C2, 0.002)                          # with optional inputs
assert isinstance(isoV2, list), "Wire.Isovist. Should be list"

# Case 15 - Length
# test 1
wLen1 = Wire.Length(w1)                 # without optional inputs
assert isinstance(wLen1, float), "Wire.Length. Should be float"
# test 2
wLen2 = Wire.Length(w4, 6)             # with optional inputs
assert isinstance(wLen2, float), "Wire.Length. Should be float"
 
# Case 16 - Planarize
# npW = Nonplanar_Wire, pW = Planar_Wire
# creating objects
npW1 = Wire.ByVertices([v1, v2, v3])                       # create wire
npW2 = Wire.ByVertices([v6, v4, v3, v1])                 # create wire
# test 1
pW1 = Wire.Planarize(npW1)
assert isinstance(pW1, topologic.Wire), "Wire.Planarize. Should be topologic.Wire"
# test 2
pW2 = Wire.Planarize(npW2)
assert isinstance(npW2, topologic.Wire), "Wire.Planarize. Should be topologic.Wire"

# Case 17 - Project
# creating objects
p1 = Vertex.ByCoordinates(1, 1, 1)                        # create vertex
p2 = Vertex.ByCoordinates(0, 0, 1)                        # create vertex
p3 = Vertex.ByCoordinates(-1, -1, 1)                     # create vertex
p4 = Vertex.ByCoordinates(-3, -2, 3)                     # create vertex
f1 = Face.ByVertices([v5, v6, v7, v8])                     # create face
f2 = Face.Rectangle(v1, 10, 10)                             # create face
w6 = Wire.ByVertices([p1, p2, p3])                       # create wire
w7 = Wire.ByVertices([p1, p3, p4])                       # create wire
# test 1
pro1 = Wire.Project(w6, f1)                                  # without optional inputs
assert isinstance(pro1, topologic.Wire),"Wire.Project. Should be topologic.Wire"
# test 2
pro2 = Wire.Project(w7, f2, [0,0,-1])                     # with optional inputs               
assert isinstance(pro2, topologic.Wire),"Wire.Project. Should be topologic.Wire"

# Case 18 - Rectangle
# test 1
rec1 = Wire.Rectangle()                                                    # without optional inputs
assert isinstance(rec1, topologic.Wire), "Wire.Rectangle. Should be topologic.Wire"
# test 2
rec2 = Wire.Rectangle(v2, 3, 7, [1, 0, 0], 'center', 0.005)     # with optional inputs
assert isinstance(rec2, topologic.Wire), "Wire.Rectangle. Should be topologic.Wire"
 
# Case 19 - RemoveCollinearEdges
# cw = CollinearWire, ce = collinearEdge
cw1 = Wire.ByVertices([v6, v1, v8])                      # create wire
cw2 = Wire.ByVertices([v5, v1, v7])                      # create wire                            
# test 1
ce1 = Wire.RemoveCollinearEdges(cw1)             # without optional inputs
assert isinstance(ce1, topologic.Wire), "Wire.RemoveCollinearEdges. Should be topologic.Wire"
# test 2
ce2 = Wire.RemoveCollinearEdges(cw2, 0.2)      # without optional inputs
assert isinstance(ce2, topologic.Wire), "Wire.RemoveCollinearEdges. Should be topologic.Wire"
 
# Case 20 -  Split
# creating Wire
w7 = Wire.ByVertices([v5, v7, v8, v6])                      # create wire
w8 = Wire.ByVertices([v8, v6, v7, v11, v12, v5])       # create wire
# test 1
Spl1 = Wire.Split(w7)
assert isinstance(Spl1, list), "Wire.Split. Should be list"
# test 2
Spl2 = Wire.Split(w8)
assert isinstance(Spl2, list), "Wire.Split. Should be list"

# Case 21 - Star
# test 1
s1 = Wire.Star(v3, 2, 5, 6, [1, 1, 1], 'lowerleft', 0.0001)                                                             # with optional inputs
assert isinstance(s1, topologic.Wire), "Wire.Star. Should be topologic.Wire"
# test 2
s2 = Wire.Star()                                                                                                                       # without optional inputs
assert isinstance(s2, topologic.Wire), "Wire.Star. Should be topologic.Wire"
 
# Case 22 - Trapezoid
# test 1
t1 = Wire.Trapezoid()                                                                                                              # without optional inputs
assert isinstance(t1, topologic.Wire), "Wire.Trapezoid. Should be topologic.Wire"
# test 2
t2 =Wire.Trapezoid(origin=v5, widthA=1.3, widthB=0.85, offsetA=0.8, offsetB=0.7, 
                   length=2.0, direction=[1, 0, 1], placement='center', tolerance=0.0001)            # with optional inputs
assert isinstance(t2, topologic.Wire), "Wire.Trapezoid. Should be topologic.Wire"

# Case 23 - Vertices
# wv = wire_vertices
# test 1
wv1 = Wire.Vertices(w3)
assert isinstance(wv1, list), "Wire.Vertices. Should be list"
# test 2
wv2 = Wire.Vertices(w6)
assert isinstance(wv2, list), "Wire.Vertices. Should be list"