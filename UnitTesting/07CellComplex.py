# CellComplex Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Plotly import Plotly

# Object for test case 
v0 = Vertex.ByCoordinates(3, 3, 0)                              # create vertex
v5 = Vertex.ByCoordinates(-1.8, 10.3, 17)                   # create vertex
v6 = Vertex.ByCoordinates(-1.8, -4.33, 17)                 # create vertex
v7 = Vertex.ByCoordinates(9.3, 9.4, 4.6)                     # create vertex
v8 = Vertex.ByCoordinates(9.3, -5.3, 4.6)                    # create vertex
v9 = Vertex.ByCoordinates(23.4, 14.3, 0)                    # create vertex
v10 = Vertex.ByCoordinates(23.4, 14.3, 0)                  # create vertex
#face 1
v1 = Vertex.ByCoordinates(-2, 2, 0)                             # create vertex
v2 = Vertex.ByCoordinates(-2, -2, 0)                           # create vertex
v3 = Vertex.ByCoordinates(2, -2, 0)                            # create vertex
v4 = Vertex.ByCoordinates(2, 2, 0)                             # create vertex
# face 2
vr1 = Vertex.ByCoordinates(2, -2, 0)                      # create vertex
vr2 = Vertex.ByCoordinates(2, 2, 0)                       # create vertex
vr3 = Vertex.ByCoordinates(2, 2, 4)                       # create vertex
vr4 = Vertex.ByCoordinates(2, -2, 4)                     # create vertex
# face 3
vt1 = Vertex.ByCoordinates(2, 2, 4)                     # create vertex
vt2 = Vertex.ByCoordinates(2, -2, 4)                    # create vertex
vt3 = Vertex.ByCoordinates(-2, -2, 4)                  # create vertex
vt4 = Vertex.ByCoordinates(-2, 2, 4)                   # create vertex
#face4
vl1 = Vertex.ByCoordinates(-2, -2, 4)                   # create vertex
vl2 = Vertex.ByCoordinates(-2, 2, 4)                    # create vertex
vl3 = Vertex.ByCoordinates(-2, 2, 0)                    # create vertex
vl4 = Vertex.ByCoordinates(-2, -2, 0)                  # create vertex
# box surfaces
boxf1 = Face.ByVertices([v1, v2, v3, v4])                  # create face
boxRf2 = Face.ByVertices([vr1, vr2, vr3, vr4])          # create face
boxTf3 = Face.ByVertices([vt1, vt2, vt3, vt4])          # create face
boxLf4 = Face.ByVertices([vl1, vl2, vl3, vl4])          # create face
# cone surfaces 
cV = Vertex.ByCoordinates(0,0,4)                             # create vertex
ConeS1 = Face.ByVertices([v1, cV, v2])                     # create face
ConeS2 = Face.ByVertices([v2, cV, vr1])                   # create face
ConeS3 = Face.ByVertices([vr1, cV, vr2])                  # create face
ConeS4 = Face.ByVertices([vr2,cV, v1])                    # create face
# box by wire
boxW1 = Wire.ByVertices([v1, v2, v3, v4])                # create wire
boxW2 = Wire.ByVertices([vr1, vr2, vr3, vr4])           # create wire
boxW3 = Wire.ByVertices([vt1, vt2, vt3, vt4])           # create wire
boxW4 = Wire.ByVertices([vl1, vl2, vl3, vl4])            # create wire
w5 =  Wire.ByVertices([v5, v6, v7, v8])                     # create wire

# Case 1 - Box 
# test 1
box1 = CellComplex.Box()                                                                  # without optional inputs
assert isinstance(box1, topologic.CellComplex), "CellComplex.Box. Should be topologic.CellComplex"
# test 2
box2 = CellComplex.Box(v1, width= 3.4, length=5.6, height=7.8, 
                        uSides=3, vSides=4, wSides=6,direction=[1, 0, 0],
                        placement= 'lowerleft')                     # with optional inputs
assert isinstance(box2, topologic.CellComplex), "CellComplex.Box. Should be topologic.CellComplex"

# Case 2 - ByCells
cellBox1 = Cell.Box(v0, 6, 6, 1)             # create cell
cellCyc = Cell.Cylinder(v0, .6, 10)         # create cell
cellCyc1 = Cell.Cylinder(v10, 5, 5)        # create cell
cellCyc2= Cell.Cylinder(v9, .8, 6)          # create cell
cellCyc3= Cell.Cylinder(v7, .3, 4)          # create cell
cellCyc4= Cell.Cylinder(v4, .22, .44)      # create cel
cellBox2 = Cell.Box(v0, 8, 12, 8)           # create cell
cellPri = Cell.Prism(v0, 2, 8, 15)            # create cell
cellBox3 = Cell.Box(v0, 8, 8, 9)             # create cell
# test 1
comC1 = CellComplex.ByCells([cellCyc1, cellCyc])                                        # without optional inputs
assert isinstance(comC1, topologic.CellComplex), "CellComplex.ByCells. Should be topologic.CellComplex"
# test 2
comC2 = CellComplex.ByCells([cellCyc2, cellCyc], 0.002)                             # with optional inputs
assert isinstance(comC2, topologic.CellComplex), "CellComplex.ByCells. Should be topologic.CellComplex"

# Case 3 - ByCellsCluster
cluC1 = Cluster.ByTopologies([cellCyc1, cellCyc, cellCyc3])                       # create cluster
cluC2 = Cluster.ByTopologies([cellCyc1, cellCyc, cellCyc4, cellCyc3])        # create cluster
# test 1
cClu1 = CellComplex.ByCellsCluster(cluC1)                                            # without optional inputs
assert isinstance(cClu1, topologic.CellComplex),  "CellComplex.ByCellsCluster. Should be topologic.CellComplex"
# test 2
cClu2 = CellComplex.ByCellsCluster(cluC2, 0.002)                                 # with optional inputs
assert isinstance(cClu2, topologic.CellComplex),  "CellComplex.ByCellsCluster. Should be topologic.CellComplex"
 
# Case 4 - ByFaces
# test 1
cube1 = CellComplex.ByFaces([boxf1, boxRf2, boxTf3, boxLf4])                                         # without optional inputs
assert isinstance(cube1, topologic.CellComplex), "CellComplex.ByFaces, Should be topologic.CellComplex"
# test 2
cone1 = CellComplex.ByFaces([ConeS1, ConeS2, ConeS3, ConeS4], 0.001)                        # with optional inputs
assert isinstance(cone1, topologic.CellComplex), "CellComplex.ByFaces, Should be topologic.CellComplex"
 
# Case 5 - ByFacesCluster
fc1 = Cluster.ByTopologies([boxf1, boxRf2, boxTf3, boxLf4])                                              # create cluster
fc2 = Cluster.ByTopologies([ConeS1, ConeS2, ConeS3, ConeS4])                                       # create cluster
# test 1
cubeC1 = CellComplex.ByFacesCluster(fc1)                                                                        # without optional inputs
assert isinstance(cubeC1, topologic.CellComplex), "CellComplex.ByFacesCluster. Should be topologic.CellComplex"
# test 2
coneC1 = CellComplex.ByFacesCluster(fc2, 0.01)                                                                # with optional inputs
assert isinstance(coneC1, topologic.CellComplex), "CellComplex.ByFacesCluster. Should be topologic.CellComplex"

# Case 6 - ByWires
wS1 = Wire.Rectangle((Vertex.ByCoordinates(0 ,0 ,0)), 5.0, 5.0)                                                 # create wire
wS2 = Wire.Rectangle((Vertex.ByCoordinates(0 ,0 ,5)), 5.0, 5.0)                                                 # create wire
wS3 = Wire.Rectangle((Vertex.ByCoordinates(0, 0,10)), 2.0, 2.0)                                                # create wire
wSt1 = Wire.Star((Vertex.ByCoordinates(0 ,0 ,0)), radiusA= 4, radiusB=6, rays=6)               # create wire
wSt2 = Wire.Star((Vertex.ByCoordinates(0 ,0 ,5)), radiusA= 2, radiusB=4, rays=6)               # create wire
wSt3 = Wire.Star((Vertex.ByCoordinates(0 ,0 ,10)), radiusA= .5, radiusB=1, rays=6)            # create wire
cel = Cluster.ByTopologies([wSt1, wSt2, wSt3])                                                                    # create cluster
wC1 = Wire.Circle(radius= 3, sides=8)                                                                                  # create wire
wC2 = Wire.Circle((Vertex.ByCoordinates(0 ,0 ,5)), radius= 2, sides=8)                                # create wire
wC3 = Wire.Circle((Vertex.ByCoordinates(0 ,0 ,10)), radius= 1, sides=8)                              # create wire
# test 1
LoftW1 = CellComplex.ByWires([wS1, wS2, wS3])                                                                # without optional inputs
assert isinstance(LoftW1, topologic.CellComplex), "CellComplex.ByWires. Should be topologic.CellComplex"
# test 2
LoftW2 = CellComplex.ByWires([wC1, wC2, wC3], False, 0.0002)                                        # with optional inputs
assert isinstance(LoftW2, topologic.CellComplex), "CellComplex.ByWires. Should be topologic.CellComplex"
#test 3
LoftW3 = CellComplex.ByWires([wSt1, wSt2, wSt3])                                                           # without optional inputs
assert isinstance(LoftW3, topologic.CellComplex), "CellComplex.ByWires. Should be topologic.CellComplex"

# Case 7 - ByWiresCluster
cluW1 = Cluster.ByTopologies([wS1, wS2, wS3])                                                                  # create cluster
cluW2 = Cluster.ByTopologies([wSt1, wSt2, wSt3])                                                              # create cluster
cluW3 = Cluster.ByTopologies([wC1, wC2, wC3])                                                                # create cluster
# test 1
LoftW4 = CellComplex.ByWiresCluster(cluW1, False, 0.002)                                               # with optional inputs
assert isinstance(LoftW4, topologic.CellComplex), "CellComplex.ByWiresCluster. Should be topologic.CellComplex"
# test 2
LoftW5 = CellComplex.ByWiresCluster(cluW2)                                                                   # without optional inputs
assert isinstance(LoftW5, topologic.CellComplex), "CellComplex.ByWiresCluster. Should be topologic.CellComplex"
# test 3
LoftW6 = CellComplex.ByWiresCluster(cluW3, False, 0.0003)                                             # with optional inputs
assert isinstance(LoftW6, topologic.CellComplex), "CellComplex.ByWiresCluster. Should be topologic.CellComplex"

# Case 8 - Cells
# test 1
bC1 = CellComplex.Cells(box1)
assert isinstance(bC1, list), "CellComplex.Cells. Should be list"
#test 2
bC2 = CellComplex.Cells(box2)
assert isinstance(bC2, list), "CellComplex.Cells. Should be list"

# Case 9 - Decompose
#  test 1
d1 = CellComplex.Decompose(box1)
assert isinstance(d1, dict), "CellComplex.Decompose. Should be dictionary"
#  test 2
d2 = CellComplex.Decompose(box2, 20, 0.05)
assert isinstance(d2, dict), "CellComplex.Decompose. Should be dictionary"

# Case 10 - Edges
# test 1
bE1 = CellComplex.Edges(box1)
assert isinstance(bE1, list), "CellComplex.Edges. Should be list"
# test 2
bE2 = CellComplex.Edges(box2)
assert isinstance(bE2, list), "CellComplex.Edges. Should be list"

# Case 11 - ExternalBoundary
# test 1
bEB1 = CellComplex.ExternalBoundary(box1)
assert isinstance(bEB1, topologic.Cell), "CellComplex.ExternalBoundary. Should be topologic.Cell"
# test 2
bEB2 = CellComplex.ExternalBoundary(box2)
assert isinstance(bEB2, topologic.Cell), "CellComplex.ExternalBoundary. Should be topologic.Cell"

# Case 12 -ExternalFaces
""" Gives list of faces instead of topologic.Cell """
# test 1
ExF1 = CellComplex.ExternalFaces(box1)
# assert isinstance(ExF1, topologic.Cell),  
# "Cell.Complex.ExternalFaces. Should be topologic.Cell"
# test 2
ExF2 = CellComplex.ExternalFaces(box2)
# assert isinstance(ExF2, topologic.Cell), "Cell.Complex.ExternalFaces. Should be topologic.Cell"
# print("External Faces1", ExF1, "External Faces2", ExF2 )

# Case 13 - Faces
# test 1
bF1 = CellComplex.Faces(box1)
assert isinstance(bF1, list), "CellComplex.Faces. Should be list"
#test 2
bF2 = CellComplex.Faces(box2)
assert isinstance(bF2, list), "CellComplex.Faces. Should be list"

# Case 14 -InternalFaces
# test 1
InF1 = CellComplex.InternalFaces(box1)
assert isinstance(InF1, list), "Cell.Complex.InternalFaces. Should be list"
# test 2
InF2 = CellComplex.InternalFaces(box2)
assert isinstance(InF2, list), "Cell.Complex.InternalFaces. Should be list"

# Case 15 - NonManifoldFaces
# test 1
nmF1 = CellComplex.NonManifoldFaces(box1)
assert isinstance(nmF1, list), "CellComplex.NonManifoldFaces. Should be list"
# test 2
nmF2 = CellComplex.NonManifoldFaces(box2)
assert isinstance(nmF2, list), "CellComplex.NonManifoldFaces. Should be list"

# Case 16 - Prism
# test 1
p1 = CellComplex.Prism()                                                                                                              #  without optional inputs                                                                               
assert isinstance(p1, topologic.CellComplex), "CellComplex.Prism, Should be topologic.CellComplex"
# test 2
p2 = CellComplex.Prism(v1, width=4.9, length= 8, height= 10, uSides=3, vSides=3,
                                        wSides=6, direction = [1, 1, 0], placement='lowerleft')                    #  with optional inputs
assert isinstance(p2, topologic.CellComplex), "CellComplex.Prism, Should be topologic.CellComplex"

# Case 17 - Shells
# test 1
bS1 = CellComplex.Shells(box1)
assert isinstance(bS1, list), "CellComplex.Shells. Should be list"
# test 2
bS2 = CellComplex.Shells(box2)
assert isinstance(bS2, list), "CellComplex.Shells. Should be list"

# Case 18 - Vertices
# test 1
bV1 = CellComplex.Vertices(box1)
assert isinstance(bV1, list), "CellComplex.Vertices. Should be list"
# test 2
bV2 = CellComplex.Vertices(box1)
assert isinstance(bV2, list), "CellComplex.Vertices. Should be list"

# Case 19 - Volume
# test 1
vol1 = CellComplex.Volume(box1)          # without optional input
assert isinstance(vol1, float), "CellComplex.Volume. Should be float"
# test 2
vol2 = CellComplex.Volume(box2, 2)     # with optional input
assert isinstance(vol2, float), "CellComplex.Volume. Should be float"

# Case 20 - Wires
# test 1
bW1 = CellComplex.Wires(box1)
assert isinstance(bW1, list), "CellComplex.Wires. Should be list"
# test 2
bW2 = CellComplex.Wires(box2)
assert isinstance(bW2, list), "CellComplex.Wires. Should be list"