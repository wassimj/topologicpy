# Cluster Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

import topologicpy
import topologic
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.Shell import Shell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Plotly import Plotly

# creating vertex
v0 = Vertex.ByCoordinates(0, 0, 0)                                      # create vertex
v1 = Vertex.ByCoordinates(-2, 2, 0)                                    # create vertex
v2 = Vertex.ByCoordinates(-2, -2, 0)                                  # create vertex
v3 = Vertex.ByCoordinates(2, -2, 0)                                   # create vertex
v4 = Vertex.ByCoordinates(2, 2, 0)                                    # create vertex
v9 = Vertex.ByCoordinates(-1.8, 10.3, 17)                        # create vertex
v10 = Vertex.ByCoordinates(-1.8, -4.33, 17)                    # create vertex
v11 = Vertex.ByCoordinates(9.3, 9.4, 4.6)                        # create vertex
v12 = Vertex.ByCoordinates(9.3, -5.3, 4.6)                      # create vertex
v13 = Vertex.ByCoordinates(23.4, 14.3, 0)                      # create vertex
v14 = Vertex.ByCoordinates(23.4, 14.3, 0)                     # create vertex
# creating edges
e1 = Edge.ByVertices([v1,v2])                                        # create edge
e2 = Edge.ByVertices([v2,v3])                                        # create edge
e3 = Edge.ByVertices([v3,v4])                                        # create edge
e4 = Edge.ByVertices([v4, v1])                                       # create edge
# creating Wire
w1 = Wire.ByEdges([e1, e2, e3, e4])                               # create wire
w2 = Wire.ByVertices([v0, v4, v1, v3])                            # create wire
w3 = Wire.Ellipse(v0, 3, 2, 5)                                         # create wire
w4 = Wire.Circle(v2)
w5 = Wire.Circle(v10, 5, 8)
# creating faces
f1 = Face.Rectangle(v0, 5, 10)                                        # create face
f2 = Face.Circle(v0, 2, 16)                                              # create face
f3 = Face.Vertices([v1, v2, v3])                                      # create face
fV1 = Face.ByVertices([v9, v10, v11, v12])                     # create face
fV2 = Face.ByVertices([v11, v12, v13, v14])                  # create face
fV3 = Face.ByVertices([v13, v14, v9, v10])                   # create face
# creating shell
sh1 = Shell.ByFaces([fV1, fV2, fV3])                               # create shell
sh2 = Shell.Circle(v11, 1, 16, direction=[1, 0, 0])           # create shell
sh3 = Shell.Circle(v12, 1.5, 16, direction=[1, 1, 0])        # create shell
sh4 = Shell.ByFaces([fV2, fV3])                                     # create shell
sh4 = Shell.Rectangle()                                                # create shell
# creating cell
cyl1 = Cell.Cylinder(v10, 2, 2, 5)
cyl2 = Cell.Cylinder(v3, 3.5, 3.5, 6, direction=[0,1,1])
cyl3 = Cell.Cylinder(v9, 2, 2, 10)
cyl4 = Cell.Cylinder(v4, 3, 3, 6, direction=[1,0,0])
#creating cellcomplexs
cbox1 = CellComplex.Box()
cbox2 = CellComplex.Box(v2, 2, 2, 2)
cPri1 = CellComplex.Prism()
cPri2 = CellComplex.Prism(v1, 2, 2, 3)
# Case 1 - ByTopologies
# test 1
cT1 = Cluster.ByTopologies([v0, e1, v1, e2, w1, f1, sh1, cyl1])
assert isinstance(cT1, topologic.Cluster), "Cluster.ByTopologies. Should be topologic.Cluster"
# test 2
cT2 = Cluster.ByTopologies([v10, e2, v14, e3, fV2, cyl2, sh1, cyl1])
assert isinstance(cT2, topologic.Cluster), "Cluster.ByTopologies. Should be topologic.Cluster"
# test 3
cT3 = Cluster.ByTopologies([cyl3, sh1, w1, e1, v4, cyl4])
assert isinstance(cT3, topologic.Cluster), "Cluster.ByTopologies. Should be topologic.Cluster"
# test 4
cT4 = Cluster.ByTopologies([sh2, v12, cyl4, f3])
assert isinstance(cT4, topologic.Cluster), "Cluster.ByTopologies. Should be topologic.Cluster"

# Case 2 - CellComplexes
ccB1 = Cluster.ByTopologies([cbox1, cbox2, cPri1])
ccB2 = Cluster.ByTopologies([cbox1, cbox2, cPri1, cPri2])
# test 1
cCC1 = Cluster.CellComplexes(ccB1)
assert isinstance(cCC1, list), "Cluster.CellComplexes. Should be list"
# test 2
cCC2 = Cluster.CellComplexes(ccB2)
assert isinstance(cCC2, list), "Cluster.CellComplexes. Should be list"

# Case 3 - Cells
# test 1
cCell1 = Cluster.ByTopologies([cyl1, cyl2, cyl3])                #cell cluster
cCell2 = Cluster.ByTopologies([cyl1, cyl2, cyl3, cyl4])        #cell cluster
cC1 = Cluster.Cells(cCell1)
assert isinstance(cC1, list), "Cluster.CellComplexes. Should be list"
# test 2
cC2 = Cluster.Cells(cCell2)
assert isinstance(cC2, list), "Cluster.CellComplexes. Should be list"

# Case 4 - Edges
# test 1
cE1 = Cluster.Edges(cT1)
assert isinstance(cE1, list), "Cluster.Edges. Should be list"
# test 2
cE2 = Cluster.Edges(cT2)
assert isinstance(cE2, list), "Cluster.Edges. Should be list"

# Case 5 - Faces
# test 1
cF1 = Cluster.Faces(cT3)
assert isinstance(cF1, list), "Cluster.Faces. Should be list"
# test 2
cF2 = Cluster.Faces(cT1)
assert isinstance(cF2, list), "Cluster.Faces. Should be list"

# Case 6 - FreeCells
# test 1
fC1 = Cluster.FreeCells(cCell1)
assert isinstance (fC1, list), "Cluster.FreeCells.Should be list"
#test 2
fC2 = Cluster.FreeCells(cCell2)
assert isinstance(fC2, list), "Cluster.FreeCells.Should be list"

# Case 7  FreeEdges
cE3 = Cluster.ByTopologies(cE1)             # Edge cluster
cE4 = Cluster.ByTopologies(cE2)             # Edge cluster
# test 1
fE1 = Cluster.FreeEdges(cE3)
assert isinstance(fE1, list), "Cluster.FreeEdges. Should be list"
# test 2
fE2 = Cluster.FreeEdges(cE4)
assert isinstance(fE2, list), "Cluster.FreeEdges. Should be list"

# Case 8 - FreeFaces
cF3 = Cluster.ByTopologies(cF1)
cF4 = Cluster.ByTopologies(cF2)
# test 1
fF1 = Cluster.FreeFaces(cF3)
assert isinstance(fF1, list), "Cluster.FreeFaces. Should be list"
# test 2
fF2 = Cluster.FreeFaces(cF4)
assert isinstance(fF2, list), "Cluster.FreeFaces. Should be list"

# Case 9 - FreeShells
cShell1 = Cluster.ByTopologies([sh1, sh2, sh3, v0, e1])
cShell2 = Cluster.ByTopologies([sh4, sh3, sh1, sh2])
#test 1
fS1 = Cluster.FreeShells(cShell1)
assert isinstance(fS1, list), "Cluster.FreeShells. Should be list"
#test 2
fS2 = Cluster.FreeShells(cShell2)
assert isinstance(fS2, list), "Cluster.FreeShells. Should be list"

# Case 10 - FreeVertices
cVer1 = Cluster.ByTopologies([v1, v10, v11, v2, v9, f1])
cVer2 =  Cluster.ByTopologies([v2, v12, v9, v4])
# test 1
fVer1 = Cluster.FreeVertices(cVer1)
assert isinstance(fVer1, list), "Cluster.FreeVertices. Should be list"
# test 2
fVer2 = Cluster.FreeVertices(cVer2)
assert isinstance(fVer2, list), "Cluster.FreeVertices. Should be list"

# Case 11 - FreeWires
cWire1 = Cluster.ByTopologies([w1, w2, w3, w4, w5, e1, v1, fVer1])
cWire2 = Cluster.ByTopologies([w1, v3, e2, fC1 , w2, w4, w5])
# test 1
fW1 = Cluster.FreeWires(cWire1)
assert isinstance(fW1, list), "Cluster.FreeWires. Should be list"
#test 2
fW2 = Cluster.FreeWires(cWire2)
assert isinstance(fW2, list), "Cluster.FreeWires. Should be list"

# Case 12 - HighestType
""" TypeError: isinstance expected 2 arguments, got 1"""
# test 1
# hiTyp1 = Cluster.HighestType(cWire1)
# assert isinstance(hiTyp1, int), "Cluster.HighestType. Should be integer"
# print(hiTyp1, "hiTyp1")
# # test 2
# hiTyp2 = Cluster.HighestType(cT2)
# print(hiTyp2, "hiTyp2") 
# assert isinstance(hiTyp2, int), "Cluster.HighestType. Should be integer"

# Case 13 - MysticRose
# test 1
cMys1 = Cluster.MysticRose()                                                                                # without optional inputs
assert isinstance(cMys1, topologic.Cluster), "Cluster.MysticRose. Should be topologic.Cluster"
# test 2
cMys2 = Cluster.MysticRose(w3, v10, 1.6, 8, direction=[1, 1, 0])                   # with optional inputs
assert isinstance(cMys2, topologic.Cluster), "Cluster.MysticRose. Should be topologic.Cluster"

# Case 14 - Shells
# test 1
cS1 = Cluster.Shells(cShell1)
assert isinstance(cS1, list), "Cluster.Shells. Should be list"
# test 2
cS2 =Cluster.Shells(cShell2)
assert isinstance(cS2, list), "Cluster.Shells. Should be list"

# Case 15 - Simplify
# test 1
Simplify1 = Cluster.Simplify(cT1)
assert isinstance(Simplify1, topologic.Topology or list), "Cluster.Simplify. Should be topologic.Topology or list"
# test 2
Simplify2 = Cluster.Simplify(cT3)
assert isinstance(Simplify1, topologic.Topology or list), "Cluster.Simplify. Should be topologic.Topology or list"

# Case 16 - Vertices
# test 1
cV1 = Cluster.Vertices(cVer1)
assert isinstance(cV1, list), "Cluster.Vertices. Should be list"
# test 2
cV2 = Cluster.Vertices(cVer2)
assert isinstance(cV2, list), "Cluster.Vertices. Should be list"

# Case 17 - Wires
# test 1
cW1 = Cluster.Wires(cT3)
assert isinstance(cW1, list), "Cluster.Wires. Should be list"
# test 2
cW2 = Cluster.Wires(cWire1)
assert isinstance(cW2, list), "Cluster.Wires. Should be list"