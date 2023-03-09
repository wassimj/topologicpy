# Cell Classes unit test

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
e0 = Edge.ByStartVertexEndVertex(v0,v1)
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
shell_open = Shell.ByFaces([face0])             # create shell

# Case 1 - Box

# test 1
list_dir = [0,0,1]
cell_b0 = Cell.Box(origin = v0, width = 0.75, length = 0.75, height = 10, uSides = 1, vSides = 1,
                   wSides = 1, direction = list_dir, placement = "center")       # with optional inputs
assert isinstance(cell_b0, topologic.Cell), "Cell.Box. Should be topologic.Cell"
# test 2
cell_b = Cell.Box()                                                                # without optional inputs
assert isinstance(cell_b, topologic.Cell), "Cell.Box. Should be topologic.Cell"
# plot geometry
data_cell_b0 = Plotly.DataByTopology(cell_b0)
figure_cell_b0 = Plotly.FigureByData(data_cell_b0)
#Plotly.Show(figure_cell_b0)                                                        # visualization

# Case 02 - Prism

# test 1
cell_prism = Cell.Prism()                                                                       # without optional inputs
assert isinstance(cell_prism, topologic.Cell), "Cell.Prism. topologic.Cell"
# test 2
cell_prism = Cell.Prism(origin=v2, width=3, length=3, height=5, uSides=3, vSides=2,
                        wSides=2, direction = [0,0,1], placement='bottom')                   # with optional inputs
assert isinstance(cell_prism, topologic.Cell), "Cell.Prism. topologic.Cell"
# plot geometry
data_cell_prism = Plotly.DataByTopology(cell_prism)
figure_cell_prism = Plotly.FigureByData(data_cell_prism)
#Plotly.Show(figure_cell_prism)                                                                 # visualization

# Case 03 - ByFaces

# test 1
cell_f = Cell.ByFaces(f_list, tolerance=0.001)                                          # with optional inputs
assert isinstance(cell_f, topologic.Cell), "Cell.ByFaces. topologic.Cell"
# test 2
cell_f = Cell.ByFaces(f_list)                                                           # without optional inputs
assert isinstance(cell_f, topologic.Cell), "Cell.ByFaces. topologic.Cell"
# plot geometry
data_cell_f = Plotly.DataByTopology(cell_f)
figure_cell_f = Plotly.FigureByData(data_cell_f)
#Plotly.Show(figure_cell_f)                                                              # visualization

# Case 04 - Area

# test 1
area_cell0 = Cell.Area(cell_f, mantissa= 2)        # with optional inputs
assert isinstance(area_cell0, float), "Cell.Area. Should be float"
assert area_cell0 == 600.00, "Cell.Area. Should be 31.12"
# test 2
area_cell = Cell.Area(cell_f)                       # without optional inputs
assert isinstance(area_cell, float), "Cell.Area. Should be float"
assert area_cell == 600.0, "Cell.Area. Should be 6.0"

# Case 05 - ByShell

# test 1
cell_s = Cell.ByShell(shell_f)
assert isinstance(cell_s, topologic.Cell), "Cell.ByShell. topologic.Cell"
# plot geometry
data_cell_s = Plotly.DataByTopology(cell_s)
figure_cell_s = Plotly.FigureByData(data_cell_s)
#Plotly.Show(figure_cell_s)                                                              # visualization

# Case 06 - ByThickenedFace

# test 1
cell_tf = Cell.ByThickenedFace(face0)                                                       # without optional inputs
assert isinstance(cell_tf, topologic.Cell), "Cell.ByThickenedFace. topologic.Cell"
# test 2
cell_tf = Cell.ByThickenedFace(face0, thickness=5, bothSides=False,
                                reverse=True, tolerance=0.001)                              # with optional inputs
# plot geometry
data_cell_tf = Plotly.DataByTopology(cell_tf)
figure_cell_tf = Plotly.FigureByData(data_cell_tf)
#Plotly.Show(figure_cell_tf)                                                                 # visualization
assert isinstance(cell_tf, topologic.Cell), "Cell.ByThickenedFace. topologic.Cell"

# Case 07 - ByThickenedShell

# test 1
cell_ts = Cell.ByThickenedShell(shell_open, direction=[0, 0, 1], thickness=2.0,
                                bothSides=False, reverse=False, tolerance=0.001)            # with optional inputs
assert isinstance(cell_ts, topologic.Cell), "Cell.ByThickenedShell. topologic.Cell"
# plot geometry
#Topology.Show(cell_ts, renderer="browser")                                                                # visualization

# Case 08 - ByWires

# test 1
cell_w = Cell.ByWires(w_list)                                                   # without optional inputs
assert isinstance(cell_w, topologic.Cell), "Cell.ByWires. topologic.Cell"
# test 2
cell_w = Cell.ByWires(w_list, close=True, triangulate=False, tolerance=0.001)   # with optional inputs
assert isinstance(cell_w, topologic.Cell), "Cell.ByWires. topologic.Cell"
# plot geometry
data_cell_w = Plotly.DataByTopology(cell_w)
figure_cell_w = Plotly.FigureByData(data_cell_w)
#Plotly.Show(figure_cell_w)                                                                 # visualization

# Case 09 - ByWiresCluster

# test 1
cell_wc = Cell.ByWiresCluster(w_cluster)                                                    # without optional inputs
assert isinstance(cell_wc, topologic.Cell), "Cell.ByWiresCluster. topologic.Cell"
# test 2
cell_wc = Cell.ByWiresCluster(w_cluster, close=True, triangulate=False, tolerance=0.001)    # with optional inputs
assert isinstance(cell_wc, topologic.Cell), "Cell.ByWiresCluster. topologic.Cell"
# plot geometry
data_cell_wc = Plotly.DataByTopology(cell_wc)
figure_cell_wc = Plotly.FigureByData(data_cell_wc)
#Plotly.Show(figure_cell_wc)                                                                 # visualization

# Case 10 - Compactness

# test 1
comp_cell = Cell.Compactness(cell_wc)                # without optional inputs
assert isinstance(comp_cell, float), "Cell.Compactness. Should be float"
# test 2
comp_cell1 = Cell.Compactness(cell_tf, mantissa=3)   # with optional inputs
assert isinstance(comp_cell1, float), "Cell.Compactness. Should be float"

# Case 11 - Cone

# test 1
cell_c = Cell.Cone()                                                                        # without optional inputs
assert isinstance(cell_c, topologic.Cell), "Cell.Cone. topologic.Cell"
# test 2
cell_c = Cell.Cone(origin=v2, baseRadius=3, topRadius=0, height=6, uSides=32, vSides=1,
                    direction = [0,0,1], placement='center', tolerance=0.001)            # with optional inputs
assert isinstance(cell_c, topologic.Cell), "Cell.Cone. topologic.Cell"
# plot geometry
data_cell_c = Plotly.DataByTopology(cell_c)
figure_cell_c = Plotly.FigureByData(data_cell_c)
#Plotly.Show(figure_cell_c)                                                                 # visualization

# Case 12 - Cylinder

# test 1
cell_cy = Cell.Cylinder()                                                                   # with optional inputs
assert isinstance(cell_cy, topologic.Cell), "Cell.Cylinder. topologic.Cell"
# test 2
cell_cy = Cell.Cylinder(origin=v3, radius=3, height=6, uSides=32, vSides=1,
                        direction = [0,0,1], placement='bottom', tolerance=0.001)                # with optional inputs
assert isinstance(cell_cy, topologic.Cell), "Cell.Cylinder. topologic.Cell"
# plot geometry
data_cell_cy = Plotly.DataByTopology(cell_cy)
figure_cell_cy = Plotly.FigureByData(data_cell_cy)
#Plotly.Show(figure_cell_cy)                                                                 # visualization

# Case 13 - Decompose

# test 1
dec_cell = Cell.Decompose(cell_cy, tiltAngle=10, tolerance=0.0001) 
assert isinstance(dec_cell, dict), "Cell.Decompose. dictionary"

# Case 14 - Edges

# test 1
e_cell = Cell.Edges(cell_c)
assert isinstance(e_cell, list), "Cell.Edges. list"
# test 2
e_cell1 = Cell.Edges(cell_cy)
assert isinstance(e_cell1, list), "Cell.Edges. list"

# Case 15 - ExternalBoundary

# test 1
eb_cell = Cell.ExternalBoundary(cell_cy)
assert isinstance(eb_cell, topologic.Shell), "ExternalBoundary. topologic.Shell"
# plot geometry
data_eb_cell = Plotly.DataByTopology(eb_cell)
figure_eb_cell = Plotly.FigureByData(data_eb_cell)
#Plotly.Show(figure_eb_cell)                                                                 # visualization

# Case 16 - Faces

# test 1
f_cell = Cell.Faces(cell_s)
assert isinstance(f_cell, list), "Cell.Faces. list"
# test 2
f_cell1 = Cell.Faces(cell_cy)
assert isinstance(f_cell1, list), "Cell.Faces. list"

# Case 17 - Hyperboloid

# test 1
cell_hp = Cell.Hyperboloid()                                                                         # without optional inputs
assert isinstance(cell_hp, topologic.Cell), "Cell.Hyperboloid. topologic.Cell"
# test 2
cell_hp = Cell.Hyperboloid(origin=v3, baseRadius=6, topRadius=2, height=4,sides=32,
                            direction = [0,0,1], twist=360, placement='bottom', tolerance=0.001)     # with optional inputs
assert isinstance(cell_hp, topologic.Cell), "Cell.Hyperboloid. topologic.Cell"
# plot geometry
data_cell_hp = Plotly.DataByTopology(cell_hp)
figure_cell_hp = Plotly.FigureByData(data_cell_hp)
#Plotly.Show(figure_cell_hp)                                                                 # visualization

# Case 18 - InternalBoundaries

# test 1
ib_cell = Cell.InternalBoundaries(cell_hp)
assert isinstance(ib_cell, list), "Cell.InternalBoundaries. list"
# test 2
ib_cell1 = Cell.InternalBoundaries(cell_cy)
assert isinstance(ib_cell1, list), "Cell.InternalBoundaries. list"

# Case 19 - InternalVertex

# test 1
iv_cell = Cell.InternalVertex(cell_s)                                                     # without optional inputs
assert isinstance(iv_cell, topologic.Vertex), "Cell.Hyperboloid. topologic.Vertex"
# test 2
iv_cell1 = Cell.InternalVertex(cell_c, tolerance=0.001)                                   # with optional inputs
assert isinstance(iv_cell1, topologic.Vertex), "Cell.Hyperboloid. topologic.Vertex"

# Case 20 - IsInside

# test 1
b_inside = Cell.IsInside(cell_c, v2)                        # without optional inputs
assert isinstance(b_inside, bool), "Cell.IsInside. boolean"
# test 2
b_inside1 = Cell.IsInside(cell_cy, v2, tolerance=0.001)     # with optional inputs
assert isinstance(b_inside1, bool), "Cell.IsInside. boolean"

# Case 21 - Pipe

# test 1
pipe_cell_dict = Cell.Pipe(e0)                                           # without optional inputs
assert isinstance(pipe_cell_dict, dict), "Cell.Pipe. dict"
pipe_cell = pipe_cell_dict['pipe']
assert isinstance(pipe_cell, topologic.Cell), "Cell.Pipe. Should be a Cell"

# test 2
pipe_cell = Cell.Pipe(e0, profile=None, radius=2, sides=32, startOffset=1,
                        endOffset=2, endcapA=None, endcapB=None)    # with optional inputs
assert isinstance(pipe_cell_dict, dict), "Cell.Pipe. dict"
pipe_cell = pipe_cell_dict['pipe']
assert isinstance(pipe_cell, topologic.Cell), "Cell.Pipe. Should be a Cell"

# Case 22 - Sets

# test 1
cell_cy2 = Cell.Cylinder()
vCy3 = Vertex.ByCoordinates(0,1,0)
cell_cy3 = Cell.Cylinder(vCy3)
cell_cy4 = Cell.Cylinder( origin = None, radius = 0.2, height = 0.2, uSides = 16, vSides = 1,
                         direction = [0, 0, 1], placement = 'center', tolerance = 0.001)
cell_cy5 = Cell.Cylinder( origin = Vertex.ByCoordinates(0.2,0,0), radius = 0.2, height = 0.2, uSides = 16, vSides = 1,
                         direction = [0, 0, 1], placement = 'center', tolerance = 0.001)
cell_cy6 = Cell.Cylinder( origin = vCy3, radius = 0.2, height = 0.2, uSides = 16, vSides = 1,
                         direction = [0, 0, 1], placement = 'center', tolerance = 0.001)
cell_cy7 = Cell.Cylinder( origin = Vertex.ByCoordinates(0,1.2,0), radius = 0.2, height = 0.2, uSides = 16, vSides = 1,
                         direction = [0, 0, 1], placement = 'center', tolerance = 0.001)
list_sub = [cell_cy4,cell_cy5,cell_cy6,cell_cy7]
list_sup = [cell_cy2,cell_cy3]
cell_sets = Cell.Sets(list_sub,list_sup)
assert isinstance(cell_sets, list), "Cell.Shells. list"

# Case 23 - Shells

# test 1
shell_cell = Cell.Shells(cell_cy)
assert isinstance(shell_cell, list), "Cell.Shells. list"
assert len(shell_cell) == 1, "Cell.Shells. List length should be 1"
# test 2
shell_cell1 = Cell.Shells(cell_hp)
assert len(shell_cell) == 1, "Cell.Shells. List length should be 1"
assert isinstance(shell_cell1, list), "Cell.Shells. list"

# Case 24 - Sphere

# test 1
cell_sphere = Cell.Sphere()                                                                       # with optional inputs
assert isinstance(cell_sphere, topologic.Cell), "Cell.Sphere. topologic.Cell"
# test 2
cell_sphere = Cell.Sphere(origin=v3, radius=3, uSides=32, vSides=16,
                            direction = [0,0,1], placement='bottom', tolerance=0.001)             # with optional inputs
assert isinstance(cell_sphere, topologic.Cell), "Cell.Sphere. topologic.Cell"
# plot geometry
data_cell_sphere = Plotly.DataByTopology(cell_sphere)
figure_cell_sphere = Plotly.FigureByData(data_cell_sphere)
#Plotly.Show(figure_cell_sphere)                                                                   # visualization

# Case 25 - SurfaceArea

# test 1
sa_cell = Cell.SurfaceArea(cell_f)                  # without optional inputs
assert isinstance(sa_cell, float), "Cell.SurfaceArea. float"
assert sa_cell == 600.0, "Cell.SurfaceArea. Should be 600.0"
# test 2
sa_cell = Cell.SurfaceArea(cell_f, mantissa=3)      # with optional inputs
assert isinstance(sa_cell, float), "Cell.SurfaceArea. float"

# Case 26 - Torus

# test 1
"""TypeError: 'NoneType' object is not iterable"""
cell_torus = Cell.Torus()                                                                    # without optional inputs
assert isinstance(cell_torus, topologic.Cell), "Cell.Torus. topologic.Cell"
# test 2
"""TypeError: 'NoneType' object is not iterable"""
#cell_torus = Cell.Torus(origin=v2, majorRadius=2, minorRadius=0.5, uSides=32, vSides=16,
#                        direction = [0,0,1], placement='bottom', tolerance=0.001)            # with optional inputs
#assert isinstance(cell_torus, topologic.Cell), "Cell.Torus. topologic.Cell"
# plot geometry
#data_cell_torus = Plotly.DataByTopology(cell_torus)
#figure_cell_torus = Plotly.FigureByData(data_cell_torus)
#Plotly.Show(figure_cell_torus)                                                               # visualization

# Case 27 - Vertices

# test 1
v_cell = Cell.Vertices(cell_hp)
assert isinstance(v_cell, list), "Cell.Vertices. list"
assert len(v_cell) == 64, "Cell.Vertices. List length should be 64"
# test 2
v_cell = Cell.Vertices(cell_f)
assert len(v_cell) == 8, "Cell.Vertices. List length should be 8"
assert isinstance(v_cell, list), "Cell.Vertices. list"

# Case 28 - Volume

# test 1
vol_cell = Cell.Volume(cell_f, mantissa=3)
assert isinstance(vol_cell, float), "Cell.Volume. float"
assert vol_cell == 1000.0, "Cell.Volume. Should be 1000.0"

# Case 29 - Wires

# test 1
w_cell = Cell.Wires(cell_c)
assert isinstance(w_cell, list), "Cell.Wires. list"
#print(len(w_cell))
# test 2
w_cell = Cell.Wires(cell_cy)
assert isinstance(w_cell, list), "Cell.Wires. list"
#print(len(w_cell))