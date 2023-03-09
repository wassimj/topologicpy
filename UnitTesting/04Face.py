# Face Classes unit test

# importing libraries
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
from topologicpy.Shell import Shell
from topologicpy.Plotly import Plotly

# creating objects for test 
# vertices
v0 = Vertex.ByCoordinates(0, 0, 0)                 # create vertex
v1 = Vertex.ByCoordinates(-2, 2, 0)               # create vertex
v2 = Vertex.ByCoordinates(-2, -2, 0)              # create vertex
v3 = Vertex.ByCoordinates(2, -2, 0)               # create vertex
v4 = Vertex.ByCoordinates(2, 2, 0)                 # create vertex
v5 = Vertex.ByCoordinates(-1, 1, 0)               # create vertex
v6 = Vertex.ByCoordinates(-1, -1, 0)              # create vertex
v7 = Vertex.ByCoordinates(1, -1, 0)                # create vertex
v8 = Vertex.ByCoordinates(1, 1, 0)                 # create vertex
v9 = Vertex.ByCoordinates(-1.8, 10.3, 17)           # create vertex
v10 = Vertex.ByCoordinates(-1.8, -4.33, 17)       # create vertex
v11 = Vertex.ByCoordinates(9.3, 9.4, 4.6)           # create vertex
v12 = Vertex.ByCoordinates(9.3, -5.3, 4.6)          # create vertex
v13 = Vertex.ByCoordinates(23.4, 14.3, 0)          # create vertex
v14 = Vertex.ByCoordinates(23.4, 14.3, 0)          # create vertex
v15 = Vertex.ByCoordinates(41, 10.3, 17)           # create vertex
v16 = Vertex.ByCoordinates(41, -4.3, 17)           # create vertex
# vertices to make traingle
tv1 = Vertex.ByCoordinates(-3,0, 0)              # create vertex
tv2 = Vertex.ByCoordinates(3, 0, 0)              # create vertex
tv3 = Vertex.ByCoordinates(0, 8, 0)              # create vertex
tv4 = Vertex.ByCoordinates(-6, 5, 0)             # create vertex
# edges
te1 = Edge.ByVertices([tv1, tv2])                # create edge
te2 = Edge.ByVertices([tv2, tv3])                # create edge
te3 = Edge.ByVertices([tv3, tv4])                # create edge
te4 = Edge.ByVertices([tv4, tv1])                # create edge
e1 = Edge.ByVertices([v1, v2])                   # create edge
e2 = Edge.ByVertices([v2, v3])                   # create edge
e3 = Edge.ByVertices([v3, v4])                   # create edge
e4 = Edge.ByVertices([v4, v1])                   # create edge

# Case 1 - AddInternalBoundaries
f1 = Face.ByVertices([v1, v2, v3, v4])                 # create face
w1 = Wire.ByVertices([v5, v6, v7, v8], True)      # create wire
c1 = Face.Circle(v0, 5, 16)                                 # create face
rec1 = Wire.Rectangle(v0, .5, .5)                       # create wire
rec2 = Wire.Rectangle(v8, .3, .3)                       # create wire
rec3 = Wire.Rectangle(v6, .4, .4)                       # create wire
# test 1
intB1 = Face.AddInternalBoundaries(f1, [w1])
assert isinstance(intB1, topologic.Face), "Face.AddInternalBoundaries. Should be topologic.Face"
# test 2
intB2 = Face.AddInternalBoundaries(c1, [rec1, rec2])
assert isinstance(intB2, topologic.Face), "Face.AddInternalBoundaries. Should be topologic.Face"
 
# Case 2 - AddInternalBoundariesCluster
# creating cluster
clu1 = Cluster.ByTopologies([rec1, rec2])                  # create cluster
clu2 = Cluster.ByTopologies([rec1, rec2, rec3])         # create cluster
# test 1
intB_C1 = Face.AddInternalBoundariesCluster(c1, clu1)
assert isinstance(intB_C1, topologic.Face), "Face.AddInternalBoundariesCluster. Should be topologic.Face"
# test 2
intB_C2 = Face.AddInternalBoundariesCluster(c1, clu2)
assert isinstance(intB_C2, topologic.Face), "Face.AddInternalBoundariesCluster. Should be topologic.Face"
# Topology.ExportToBRep(intB_C2,r"E:\UK_Work\Topologic_Work\Export\intB_C2.brep",True)

# Case 3 - Angle
# creating face
rec4 = Face.Rectangle(v1, 5, 4, [0, 1, 1])                                            # create face
rec5 = Face.Rectangle(v1, 5,4,[0, 0, 1])                                                # create face
c2 = Face.Circle(v1, 1, 16, direction=[45, 43, 45])                 # create face
c3 = Face.Circle(v1, 1, 16, direction=[-45, -45, -45])             # create face
# test 1
rA1 = Face.Angle(rec4, rec5)                                                            # without optional inputs                
assert isinstance(rA1, float), "Face.Angle. Should be float"
# test 2
rA2 = Face.Angle(c2, c3, 3)                                                              # with optional inputs
assert isinstance(rA2, float), "Face.Angle. Should be float"

# Case 4 - Area
# test 1
areaS = Face.Area(f1)                                                  # without optional inputs 
assert isinstance(areaS, float), "Face.Area. Should be float"
# test 2
areaC = Face.Area(c1, 6)                                               # with optional inputs 
assert isinstance(areaC, float), "Face.Area. Should be float"
 
# Case 5 - BoundingFace
# test 1
bF1 = Face.BoundingFace(f1)
assert isinstance(bF1, topologic.Face), "Face.BoundingFace. Should be topologic.Face"
# test 2
bF2 = Face.BoundingFace(c1)
assert isinstance(bF1, topologic.Face), "Face.BoundingFace. Should be topologic.Face"

# Case 6 - BoundingRectangle
starF = Face.Star(v10, 5.0, 2.0, 5)                     # create face
starF0 = Face.Star(v10, 6.0, 2.5, 6)                   # create face                                          
# test 1
bR1 = Face.BoundingRectangle(starF, 5)
assert isinstance(bR1, topologic.Face), "Face.BoundingRectanlge. Should be topologic.Face"
# test 2
bR2 = Face.BoundingRectangle(starF0, 10)
assert isinstance(bR2, topologic.Face), "Face.BoundingRectanlge. Should be topologic.Face"

# Case 7 - ByEdges
# test 1
fE1 = Face.ByEdges([e1, e2, e3, e4])
assert isinstance(fE1, topologic.Face), "Face.ByEdges. Should be topologic.Face"
# test 2
fE2 = Face.ByEdges([te1, te2, te3, te4])
assert isinstance(fE2, topologic.Face), "Face.ByEdges. Should be topologic.Face"

# Case 8 - ByEdgesCluster
# creating Cluster
Clu3 = Cluster.ByTopologies([e1, e2, e3, e4])                   # create cluster
Clu4 = Cluster.ByTopologies([te1, te2, te3, te4])              # create cluster
# test 1
fcE1 = Face.ByEdgesCluster(Clu3)                                   
assert isinstance(fcE1, topologic.Face), "Face.ByEdgesCluster. Should be topologic.Fac"
# test 2
fcE2 = Face.ByEdgesCluster(Clu4)
assert isinstance(fcE2, topologic.Face), "Face.ByEdgesCluster. Should be topologic.Fac"
# Topology.ExportToBRep(fcE1,r"E:\UK_Work\Topologic_Work\Export\fcE1.brep",True)
# Topology.ExportToBRep(fcE2,r"E:\UK_Work\Topologic_Work\Export\fcE2.brep",True)
 
# Case 9 - ByOffset
# test 1
offF1 = Face.ByOffset(f1)                                                       # without optional inputs 
assert isinstance(offF1, topologic.Face), "Face.ByOffset. Should be topologic.Face"
# test 2
offF2 = Face.ByOffset(c1, .5, True, miterThreshold= 2.0)       # without optional inputs 
assert isinstance(offF2, topologic.Face), "Face.ByOffset. Should be topologic.Face"
#  plot geometry
# geo1 = Plotly.DataByTopology(offF2)       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')
 
# Case 10 - ByShell
# creating Face by vertices
fV1 = Face.ByVertices([v9, v10, v11, v12])                   # create face
fV2 = Face.ByVertices([v11, v12, v13, v14])                 # create face
fV3 = Face.ByVertices([v13, v14, v9, v10])                   # create face
#creating Shell by faces
sh1 = Shell.ByFaces([fV1, fV2])                                   # create shell
sh2 = Shell.ByFaces([fV1, fV2, fV3])                            # create shell
# test 1
fs1 = Face.ByShell(sh1)                                               # without optional inputs
assert isinstance(fs1, topologic.Face), "Face.ByShell. Should be topologic.Face"
# test 2
fs2 = Face.ByShell(sh2, 0.02)                                      # with optional inputs
assert isinstance(fs2, topologic.Face), "Face.ByShell. Should be topologic.Face"

# Case 11 - ByVertices
#fV = Face by vertices
# test 1
fV1 = Face.ByVertices([v1, v2, v3, v4])                       # with four vertices
assert isinstance(fV1, topologic.Face), "Face.ByVertices. Should be topologic.Face"
# test 2
fV2 = Face.ByVertices([tv1, tv2, tv3])                         # with three vertices
assert isinstance(fV2, topologic.Face), "Face.ByVertices. Should be topologic.Face"
# plot geometry
# face1 = Plotly.DataByTopology(fV2)                           # create plotly data
# plotfig1 = Plotly.FigureByData(face1)
# Plotly.Show(plotfig1)
 
# Case 12 - ByWire
# fW = Face by wire
# creating Wire
w2 = Wire.ByVertices([v1, v2, v3, v4])                             # create Wire
# test 1
fW1 = Face.ByWire(w1)
assert isinstance (fW1, topologic.Face), "Face.ByWire. Should be topologic.Face"
# test 2
fW2 = Face.ByWire(w2)
assert isinstance (fW2, topologic.Face), "Face.ByWire. Should be topologic.Face"
 
# Case 13 - ByWires
# fWs = Face by wire(s)
#creating Wire
cW1 = Wire.Circle(v0, 5, 16)                                                        # create Wire
cW2 = Wire.Circle(v0, .5, 6)                                                         # create Wire
cW3 = Wire.Circle(v8, .3, 6)                                                         # create Wire
cW4 = Wire.Circle(v6, .4, 6)                                                         # create Wire
cW5 = Wire.Circle(v5, .4, 6)                                                         # create Wire
traW2 = Wire.Trapezoid(v0, 10, 20)                                             # create Wire
# test 1
fWs1 = Face.ByWires(cW1, [rec1, rec2, rec3])                             # with three wires
assert isinstance(fWs1, topologic.Face), "Face.ByWires. Should be topologic.Face"
# test 2
fWs2 = Face.ByWires(traW2, [cW2, cW3, cW4, cW5])                # with four wires
assert isinstance(fWs2, topologic.Face), "Face.ByWires. Should be topologic.Face"
 
# Case 14 - ByWiresCluster
# creating Cluster
clu5 = Cluster.ByTopologies([rec1, rec2, rec3])                          # create cluster
clu6 = Cluster.ByTopologies([cW2, cW3, cW4, cW5])                # create cluster
# test 1
fcW1 = Face.ByWiresCluster(cW1, clu5)                                    # with optional inputs
assert isinstance(fcW1, topologic.Face), "Face.ByWiresCluster, Should be topologic.Face"
# test 2
fcW2 = Face.ByWiresCluster(cW1, clu6)                                    # with optional inputs
assert isinstance(fcW2, topologic.Face), "Face.ByWiresCluster, Should be topologic.Face"

# Case 15 - Circle
# test 1
cF1 = Face.Circle()                                                                    # without optional inputs
assert isinstance(cF1, topologic.Face), "Face.Circle. Should be topologic.Face"
# test 2
cF2 = Face.Circle(v2, 4.85, 16, direction=[0, 1, 0], tolerance= 0.01)         # with optional inputs
assert isinstance(cF2, topologic.Face), "Face.Circle. Should be topologic.Face"
# test 3
cF3 = Face.Circle(v2, 2.5, 16, fromAngle=45, toAngle=270,    # with optional inputs
                            direction=[5, 1, -45], placement='lowerleft', tolerance= 0.01)
assert isinstance(cF3, topologic.Face), "Face.Circle. Should be topologic.Face"
 
# Case 16 - Compactness
# test 1
fC1 = Face.Compactness(f1)                                                # without optional inputs
assert isinstance(fC1, float), "Face.Compactness. Should be float"
# test 2
fC2 = Face.Compactness(cF1,5)                                           # with optional inputs
assert isinstance(fC2, float), "Face.Compactness. Should be float"
 
# Case 17 - CompassAngle
""" Description missing in documentation """

# Case 18 - Edges
# test 1
fEdg1 = Face.Edges(f1)
assert isinstance(fEdg1, list), "Face.Edges. Should be list"
# test 2
fEdg2 = Face.Edges(cF1)
assert isinstance(fEdg1, list), "Face.Edges. Should be list"

# Case 19 - FacingToward
# test 1
ft1 = Face.FacingToward(f1, [0, 0, 1], True, 0.005)                   # with optional inputs
assert isinstance(ft1, bool), "Face.FacingToward. Should be Boolean"
# test 2
ft2 = Face.FacingToward(cF1)                                                 # without optional inputs
assert isinstance(ft2, bool), "Face.FacingToward. Should be Boolean"

# Case 20 - Flatten
# creating Face
RecF = Face.Rectangle(v1, 5, 5, [45, 90, 15])                    # create face
CirF = Face.Circle(v5, 3, 16, direction= [1, 0, 0])                  # create face
#test 1
ff1 = Face.Flatten(CirF)
assert isinstance(ff1, topologic.Face), "Face.Flatten. Should be topologic.Face"
#test 2
ff2 = Face.Flatten(RecF)
assert isinstance(ff2, topologic.Face), "Face.Flatten. Should be topologic.Face"

# Case 20 - Harmonize
#test 1
hF1 = Face.Harmonize(RecF)
assert isinstance(hF1, topologic.Face), "Face.Harmonize. Should be topologic.Face"
#test 2
hF2 = Face.Harmonize(CirF)
assert isinstance(CirF, topologic.Face), "Face.Harmonize. Should be topologic.Face"

# Case 21 - ExternalBoundary
# test 1
ExtB1 = Face.ExternalBoundary(intB1)
assert isinstance(ExtB1, topologic.Wire), "Face.ExternalBoundary. Should be topologic.Wire"
# test 2
ExtB2 = Face.ExternalBoundary(intB2)
assert isinstance(ExtB2, topologic.Wire), "Face.ExternalBoundary. Should be topologic.Wire"

# Case 21- InternalBoundaries
# test 1
IntB1 = Face.InternalBoundaries(intB1)
assert isinstance(IntB1, list), "Face.InternalBoundaries. Should be list"
# test 2
IntB2 = Face.InternalBoundaries(intB2)
assert isinstance(IntB2, list), "Face.InternalBoundaries. Should be list"

# Case 22 - InternalVertex
# test 1
iv1 = Face.InternalVertex(f1)                                                     # without optional inputs
assert isinstance(iv1, topologic.Vertex), "Face.InternalVertex. Should be topologic.Vertex"
# test 2
iv2 = Face.InternalVertex(c1, 0.005)                                          # with optional inputs
assert isinstance(iv2, topologic.Vertex), "Face.InternalVertex. Should be topologic.Vertex"
 
# Case 23 - Invert
# test 1
IntF1 = Face.Invert(f1)
assert isinstance(IntF1, topologic.Face), "Face.Invert. Should be topologic.Face"
# test 2
IntF2 = Face.Invert(c1)
assert isinstance(IntF2, topologic.Face), "Face.Invert. Should be topologic.Face"

# Case 24 - IsCoplanar
# test 1
chkC1 = Face.IsCoplanar(fV1, fV2)               # without optional inputs
assert isinstance(chkC1, bool), "Face.IsCoplanar. Should be boolean"
# test 2
ChkC2 = Face.IsCoplanar(f1, c1, 0.005)        # with optional inputs
assert isinstance(ChkC2, bool), "Face.IsCoplanar. Should be boolean"
 
# Case 25 - IsInside
# test 1
isIn1 = Face.IsInside(f1, v8)                        # without optional inputs
assert isinstance(isIn1, bool), "Face.IsInside. Should be boolean"
# test 2
isIn2 = Face.IsInside(c1, tv4, 0.001)            # with optional inputs
assert isinstance(isIn2, bool), "Face.IsInside. Should be boolean"

# Case 26 - MedialAxis
# test 1
mAxis1 = Face.MedialAxis(f1)
assert isinstance(mAxis1, topologic.Wire), "Face.MedialAxis. Should be topologic.Wire "
# test 2
mAxis2 = Face.MedialAxis(c1, 5, True, True, True, 0.002, 0.5)
assert isinstance(mAxis2, topologic.Wire), "Face.MedialAxis. Should be topologic.Wire "

# Case 27 - NormalAtParameters
# test 1
np1 = Face.NormalAtParameters(fV1, .4, .5,'XYZ' , 5)           # XYZ output, with optional inputs
assert isinstance(np1, list), "Face.NormalAtParameters. Should be list"
# test 1.1
np1_X = Face.NormalAtParameters(fV1, .4, .5,'X' , 5)           # X output, with optional inputs
assert isinstance(np1_X, list), "Face.NormalAtParameters. Should be list"
# test 1.2
np1_Y = Face.NormalAtParameters(fV1, .4, .5,'Y' , 5)           # Y output, with optional inputs
assert isinstance(np1_Y, list), "Face.NormalAtParameters. Should be list"
# test 1.3
np1_Z = Face.NormalAtParameters(fV1, .4, .5,'Z' , 5)           # Z output, with optional inputs
assert isinstance(np1_Y, list), "Face.NormalAtParameters. Should be list"
# test 2
np2 = Face.NormalAtParameters(fV3)                             # without optional inputs
assert isinstance(np2, list), "Face.NormalAtParameters. Should be list"
# test 2.1
np2_X = Face.NormalAtParameters(fV1, outputType='X')           # with one optional input
assert isinstance(np2_X, list), "Face.NormalAtParameters. Should be list"
# test 2.2
np2_Y = Face.NormalAtParameters(fV1, outputType='Y')           # with one optional input
assert isinstance(np2_Y, list), "Face.NormalAtParameters. Should be list"
# test 2.3
np2_Z = Face.NormalAtParameters(fV1, outputType='Z')           # with one optional input
assert isinstance(np2_Y, list), "Face.NormalAtParameters. Should be list"
 
# Case 28 - Project
# creating Face
c4 = Face.Circle((Vertex.ByCoordinates(3.7,2.5,1)), .5, 6)                                # create face
c5 = Face.Circle((Vertex.ByCoordinates(2.5, 4, 1)), 1, 16)                                # create face
r1 = Face.Rectangle((Vertex.ByCoordinates(3.7,2.5,5)), 10, 10, direction= [0, 0, -1])        # create face
# test 1
pf1 = Face.Project(c4, r1)                                                              # without optional inputs
assert isinstance(pf1, topologic.Face), "Face.Project. Should be topologic.Face"
# test 2
pf2 = Face.Project(c5, r1, [0,0,-1], 2, 0.001)                                   # with optional inputs
assert isinstance(pf2, topologic.Face), "Face.Project. Should be topologic.Face"
# plot geometry
# geo1 = Plotly.DataByTopology(pf2)                                          # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1)
 
# Case 29 - Rectangle
# test 1
RecF1 = Face.Rectangle()                                                                     # without optional inputs
assert isinstance(RecF1, topologic.Face), "Face.Rectangle. Should be topologic.Face"
# test 2
RecF2 = Face.Rectangle(v8, 3.45, 6.78, [1, 1, 1], 'lowerleft', 0.01)           # with optional inputs
assert isinstance(RecF1, topologic.Face), "Face.Rectangle. Should be topologic.Face"

# Case 30 - Star
# test 1
StarF1 = Face.Star()                                                                                  # without optional inputs
assert isinstance(StarF1, topologic.Face), "Face.Star. Should be topologic.Face"
# test 2
StarF2 = Face.Star(v2, 2.5,.8,8, direction=[0, 1, 0], tolerance=0.001)             # with optional inputs
assert isinstance(StarF2, topologic.Face), "Face.Star. Should be topologic.Face"
# plot geometry
# star1 = Plotly.DataByTopology(StarF1)                                      # create plotly data
# plotfig1 = Plotly.FigureByData(star1)
# Plotly.Show(plotfig1)

# Case 31 - Trapezoid
# test 1
TraF1 = Face.Trapezoid()                                                                       # without optional inputs
assert isinstance(TraF1, topologic.Face), "Face.Trapezoid. Should be topologic.Face"
# test 2
TraF2 = Face.Trapezoid(v7, 2.7, 1.15, .75, .25, direction= [1, 0, 0])             # with optional inputs
assert isinstance(TraF2, topologic.Face), "Face.Trapezoid. Should be topologic.Face"
# plot geometry
# tra1 = Plotly.DataByTopology(TraF1)                                       # create plotly data
# plotfig1 = Plotly.FigureByData(tra1)
# Plotly.Show(plotfig1)

# Case 32 - Triangulate
# test 1
triS1 = Face.Triangulate(c1)
assert isinstance(triS1, list), "Face.Triangulate. Should be list"
# test 2
triS2 = Face.Triangulate(intB_C2)
assert isinstance(triS1, list), "Face.Triangulate. Should be list"

# Case 33 - TrimByWire
# creating Wire
tw1 = Wire.ByVertices(
                       [(Vertex.ByCoordinates(-4.7,-0.25,10)), 
                        (Vertex.ByCoordinates(20.6,9.8,10)), 
                        (Vertex.ByCoordinates(43.8,11.8,10))]
                       )                                                     
# test 1
to1 = Face.TrimByWire(fV1, tw1)                                                # without optional inputs
assert isinstance(to1, topologic.Face), "Face.TrimByWire. Should be topologic.Face"
# test 2
to2 = Face.TrimByWire(fV2, tw1, True)                                        # with optional inputs
assert isinstance(to2, topologic.Face), "Face.TrimByWire. Should be topologic.Face"
# plot geometry
# trim1 = Plotly.DataByTopology(to2)                                         # create plotly data
# plotfig1 = Plotly.FigureByData(trim1)
# Plotly.Show(plotfig1)
 
# Case 34 - VertexByParameters
# test 1
vp1 = Face.VertexByParameters(c1)                                                 # without optional inputs
assert isinstance(vp1, topologic.Vertex), "Face.VertexByParameters. Should be topologic.Vertex"
# test 2
vp2 = Face.VertexByParameters(c1, .277, .65)                                  # with optional inputs
assert isinstance(vp1, topologic.Vertex), "Face.VertexByParameters. Should be topologic.Vertex"

# Case 35 - VertexParameters
# test 1
vps1 = Face.VertexParameters(c1, v11)                                                           # without optional inputs
assert isinstance(vps1, list), "Face.VertexParameters. Should be list"
# test 2
vps2 = Face.VertexParameters(f1, v12, outputType='uv', mantissa = 2)         # with optional inputs
assert isinstance(vps2, list), "Face.VertexParameters. Should be list"

# Case 36 - Vertices
# test 1
fVer1 = Face.Vertices(intB_C2)
assert isinstance(fVer1, list), "Face.Vertices. Should be list"
# test 2
fVer2 = Face.Vertices(intB2)
assert isinstance(fVer2, list), "Face.Vertices. Should be list"

# Case 37 - Wires
# test 1
fWire1 = Face.Wires(intB_C2)
assert isinstance(fWire1, list), "Face.Wires. Should be list"
# test 2
fWire2 = Face.Wires(intB2)
assert isinstance(fWire2, list), "Face.Wires. Should be list"

# Case 38 - ByVerticesCluster
# creating Cluster
clu7 = Cluster.ByTopologies([v1, v2, v3, v4])                  # create cluster
Clu8 = Cluster.ByTopologies([tv1, tv2, tv3])                   # create cluster
# test 1
fcV1 = Face.ByVerticesCluster(clu7)
assert isinstance(fcV1, topologic.Face), "Face.ByVerticesCluster. Should be topologic.Face"
# test 2
fcV2 = Face.ByVerticesCluster(Clu8)
assert isinstance(fcV2, topologic.Face), "Face.ByVerticesCluster. Should be topologic.Face"