# Vertex Classes unit test

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
from topologicpy.Topology import Topology
import math
# Object for test case
f = Face.Rectangle()


# Case 1 - ByCoordinates
# test 1
v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
assert isinstance(v0, topologic.Vertex), "Vertex.ByCoordinates. Should be topologic.Vertex"

# If the above passes, we can create the rest of the vertices
v1 = Vertex.ByCoordinates(0, 10, 1)         # create vertex
v2 = Vertex.ByCoordinates(10, 10, 1)        # create vertex
v3 = Vertex.ByCoordinates(10, 0, 1)         # create vertex
v4 = Vertex.ByCoordinates(0, 0, -1)         # create vertex
v5 = Vertex.ByCoordinates(0, 10, -1)        # create vertex
v6 = Vertex.ByCoordinates(10, 10, -1)       # create vertex

v_list1 = [v1, v2, v3, v4, v5, v6]
v_list2 = [v1, v2, v3]

# Case 2 - AreIpsilateral
# test 1
status = Vertex.AreIpsilateral(v_list1, face=f)
assert status == False, "Vertex.AreIpsilateral. Should be False"
# test 2
status = Vertex.AreIpsilateral(v_list2, face=f)
assert status == True, "Vertex.AreIpsilateral. Should be False"

# Case 3 - AreIpsilateralCluster
# test 1
status = Vertex.AreIpsilateralCluster(Cluster.ByTopologies(v_list1), face=f)
assert status == False, "Vertex.AreIpsilateral. Should be False"
# test 2
status = Vertex.AreIpsilateralCluster(Cluster.ByTopologies(v_list2), face=f)
assert status == True, "Vertex.AreIpsilateral. Should be False"

# Case 4 - AreOnSameSide
# test 1
status = Vertex.AreOnSameSide(v_list1, face=f)
assert status == False, "Vertex.AreIpsilateral. Should be False"
# test 2
status = Vertex.AreOnSameSide(v_list2, face=f)
assert status == True, "Vertex.AreIpsilateral. Should be False"

# Case 5 - AreOnSameSideCluster
# test 1
status = Vertex.AreOnSameSideCluster(Cluster.ByTopologies(v_list1), face=f)
assert status == False, "Vertex.AreIpsilateral. Should be False"
# test 2
status = Vertex.AreOnSameSideCluster(Cluster.ByTopologies(v_list2), face=f)
assert status == True, "Vertex.AreIpsilateral. Should be False"

# Case 6 - Coordinates
# test 1
coordinates = Vertex.Coordinates(v1, mantissa=0)
assert coordinates == [0,10,1], "Vertex.Coordinates. Should be [0,10,1]"
# test 2
coordinates = Vertex.Coordinates(v1, outputType = "xy", mantissa=0)
assert coordinates == [0,10], "Vertex.Coordinates. Should be [0.0,10.0]"
# test 3
coordinates = Vertex.Coordinates(v1, outputType = "xz", mantissa=0)
assert coordinates == [0,1], "Vertex.Coordinates. Should be [0.0,1.0]"
# test 4
coordinates = Vertex.Coordinates(v1, outputType = "yz", mantissa=0)
assert coordinates == [10,1], "Vertex.Coordinates. Should be [10.0,1.0]"
# test 5
coordinates = Vertex.Coordinates(v1, outputType = "x", mantissa=0)
assert coordinates == [0], "Vertex.Coordinates. Should be [0.0]"
# test 6
coordinates = Vertex.Coordinates(v1, outputType = "y", mantissa=0)
assert coordinates == [10], "Vertex.Coordinates. Should be [10.0]"
# test 7
coordinates = Vertex.Coordinates(v1, outputType = "z", mantissa=0)
assert coordinates == [1], "Vertex.Coordinates. Should be [1.0]"

# Case 7 - Distance
# test 1
d = Vertex.Distance(v4, v5)
assert d == 10, "Vertex.Distance. Should be 10"
# test 2
c = Cell.Prism(origin=v5)
d = Vertex.Distance(v4, c)
assert d == 9.5, "Vertex.Distance. Should be 9.5"
# test 3
v7 = Vertex.ByCoordinates(0,4,0)
v8 = Vertex.ByCoordinates(4,0,0)
e = Edge.ByVertices([v7, v8])
d = Vertex.Distance(v0, e)
epsilon = abs(d - 2.8284)
assert epsilon < 0.001, "Vertex.Distance. Should very small"

# Case 8 - EnclosingCell
# test 1
cc = CellComplex.Prism(height=2)
v9 = Vertex.ByCoordinates(0.7,0.8,0.5)
v10 = Vertex.ByCoordinates(0.25,0.3,0.75)
# test 1
cells = Vertex.EnclosingCell(v9, cc)
assert len(cells) == 0, "EnclosingCell. Length of cells should be 0"
# test 2
cells = Vertex.EnclosingCell(v10, cc)
assert len(cells) == 1, "EnclosingCell. Length of cells should be 1"
# test 3
cell = cells[0]
centroid = Topology.Centroid(cell)
coordinates = Vertex.Coordinates(centroid)
assert coordinates == [0.25, 0.25, 0.5], "EnclosingCell. Coordinates should be [0.5, 0.5, 0.5]"

# Case 9 - Index
# test 1
i = Vertex.Index(v3, v_list1)
assert i == 2, "EnclosingCell. Index. i should be 2"

# Case 10 - IsInside
# test 1
status = Vertex.IsInside(v9, cc)
assert status == False, "IsInside. status should be False"
# test 2
status = Vertex.IsInside(v10, cc)
assert status == True, "IsInside. status should be True"

# Case 11 - NearestVertex
# test 1
cluster = Cluster.ByTopologies(v_list1)
v = Vertex.NearestVertex(v_list1[2], cluster)
i = Vertex.Index(v, v_list1)
assert i == 2, "NearestVertex. i must be 2"

# Case 12 - Origin
# test 1
origin = Vertex.Origin()
coordinates = Vertex.Coordinates(origin)
assert coordinates == [0,0,0], "Origin. coordinates should be [0,0,0]"

# Case 13 - Project
# test 1
v = Vertex.ByCoordinates(10,10,10)
f = Face.Rectangle(width=20, length=20)
v_p = Vertex.Project(v, f)
coordinates = Vertex.Coordinates(v_p)
assert coordinates == [10,10,0], "Origin. coordinates should be [10,10,0]"

# Case 14 - X
# test 1
v = Vertex.ByCoordinates(10,20,30)
assert Vertex.X(v) == 10, "Origin. x coordinate should be 10"

# Case 15 - Y
# test 1
v = Vertex.ByCoordinates(10,20,30)
assert Vertex.Y(v) == 20, "Origin. y coordinate should be 20"

# Case 16 - Z
# test 1
v = Vertex.ByCoordinates(10,20,30)
assert Vertex.Z(v) == 30, "Origin. z coordinate should be 30"


