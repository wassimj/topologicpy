# Shell Classes unit test

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
from topologicpy.Topology import Topology

# Object for test case
v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
v2 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
v3 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
v5 = Vertex.ByCoordinates(-10, 10, 0)       # create vertex
v6 = Vertex.ByCoordinates(-10, 0, 0)        # create vertex
v_list0 = [v0,v1,v2,v3,v0]                  # create list
v_list1 = [v0,v1,v5,v6,v0]                  # create list
wire0 = Wire.ByVertices(v_list0)            # create wire
wire1 = Wire.ByVertices(v_list1)            # create wire
w_list = [wire0,wire1]                      # create list
w_cluster = Cluster.ByTopologies(w_list)    # create cluster
face0 = Face.ByVertices(v_list0)            # create face
face1 = Face.ByVertices(v_list1)            # create face
f_list = [face0,face1]                      # create list
c_faces = Cluster.ByTopologies(f_list)      # create cluster

# Case 1 - ByFaces

# test 1
shell_f = Shell.ByFaces(f_list)             # without tolerance
assert isinstance(shell_f, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"
# test 2
shell_f = Shell.ByFaces(f_list,0.001)       # with tolerance
assert isinstance(shell_f, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"

# Case 2 - ByFacesCluster

# test 1
shell_fc = Shell.ByFacesCluster(c_faces)
assert isinstance(shell_fc, topologic.Shell), "Shell.ByFacesCluster. Should be topologic.Shell"

# Case 3 - ByWires

# test 1
shell_w = Shell.ByWires(w_list)                                     # without optional inputs
assert isinstance(shell_w, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"
# test 2
shell_w = Shell.ByWires(w_list, triangulate=True, tolerance=0.001)  # with optional inputs
assert isinstance(shell_w, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"
# test 3
shell_w = Shell.ByWires(w_list, triangulate=False, tolerance=0.001)  # with optional inputs
assert isinstance(shell_w, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"

# Case 4 - ByWiresCluster

# test 1
shell_wc = Shell.ByWiresCluster(w_cluster)               # without optional inputs
assert isinstance(shell_wc, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"
# test 2
shell_wc = Shell.ByWiresCluster(w_clustertriangulate=True, tolerance=0.001)   # with optional inputs
assert isinstance(shell_wc, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"
# test 3
shell_wc = Shell.ByWiresCluster(w_clustertriangulate=False, tolerance=0.001)   # with optional inputs
assert isinstance(shell_wc, topologic.Shell), "Shell.ByFaces. Should be topologic.Shell"

# Case 5 - Circle

# test 1
shell_c = Shell.Circle()                                                                 # without optional inputs
assert isinstance(shell_c, topologic.Shell), "Shell.Circle. Should be topologic.Shell"
# test 2
shell_c = Shell.Circle(v1, radius=2, sides=64, fromAngle=90, toAngle=180,
                        direction = [0,0,1], placement='lowerleft', tolerance=0.001)  # with optional inputs
assert isinstance(shell_c, topologic.Shell), "Shell.Circle. Should be topologic.Shell"

# Case 6 - Edges

# test 1
e_shell = Shell.Edges(shell_w)
assert isinstance(e_shell, list), "Shell.Edges. Should be list"
# test 2
e_shell2 = Shell.Edges(shell_fc)
assert isinstance(e_shell2, list), "Shell.Edges. Should be list"

# Case 7 - ExternalBoundary

# test 1
eb_shell = Shell.ExternalBoundary(shell_c)
assert isinstance(eb_shell, topologic.Wire), "Shell.ExternalBoundary. Should be Wire"
# test 2
eb_shell2 = Shell.ExternalBoundary(shell_wc)
assert isinstance(eb_shell2, topologic.Wire), "Shell.ExternalBoundary. Should be Wire"

# Case 8 - Faces

# test 1
f_shell = Shell.Faces(shell_wc)
assert isinstance(f_shell, list), "Shell.Faces. Should be list"
# test 2
f_shell2 = Shell.Faces(shell_c)
assert isinstance(f_shell2, list), "Shell.Faces. Should be list"

# Case 9 - HyperbolicParaboloidCircularDomain

# test 1
shell_hpcd = Shell.HyperbolicParaboloidCircularDomain()                                                 # without optional inputs
assert isinstance(shell_hpcd, topologic.Shell), "Shell.HyperbolicParaboloidCircularDomain. Should be topologic.Shell"
# test 2
shell_hpcd = Shell.HyperbolicParaboloidCircularDomain(v2, radius=3.7, sides=64, rings=21, A=3, B=-3,
                                                        direction = [0,0,1], placement='lowerleft')  # with optional inputs
assert isinstance(shell_hpcd, topologic.Shell), "Shell.HyperbolicParaboloidCircularDomain. Should be topologic.Shell"

# Case 10 - HyperbolicParaboloidRectangularDomain

# test 1
shell_hprd = Shell.HyperbolicParaboloidRectangularDomain()                                                      # without optional inputs
assert isinstance(shell_hprd, topologic.Shell), "Shell.HyperbolicParaboloidRectangularDomain. Should be topologic.Shell"
# test 2
shell_hprd = Shell.HyperbolicParaboloidRectangularDomain(v3, llVertex=None, lrVertex=None, ulVertex=None, urVertex=None, u=20,
                                                        v=20, direction = [0,0,1], placement='lowerleft')    # with optional inputs
assert isinstance(shell_hprd, topologic.Shell), "Shell.HyperbolicParaboloidRectangularDomain. Should be topologic.Shell"

# Case 11 - InternalBoundaries

# test 1
ib_shell = Shell.InternalBoundaries(shell_hpcd)
assert isinstance(ib_shell, topologic.Topology), "Shell.InternalBoundaries. Should be Topology"
# test 2
ib_shell2 = Shell.InternalBoundaries(shell_hprd)
assert isinstance(ib_shell2, topologic.Topology), "Shell.InternalBoundaries. Should be Topology"

# Case 12 - IsClosed

# test 1
bool_shell = Shell.IsClosed(shell_hprd)
assert isinstance(bool_shell, bool), "Shell.IsClosed. Should be bool"
# test 2
bool_shell2 = Shell.IsClosed(shell_hpcd)
assert isinstance(bool_shell2, bool), "Shell.IsClosed. Should be bool"

# Case 13 - Pie

# test 1
shell_p = Shell.Pie()                                                           # without optional inputs
assert isinstance(shell_p, topologic.Shell), "Shell.Pie. Should be topologic.Shell"
# test 2
shell_p = Shell.Pie(v1, radiusA=10, radiusB=5, sides=64, rings=2, fromAngle=0, toAngle=90,
                    direction = [0,0,1], placement='lowerleft', tolerance=0.001)             # with optional inputs
assert isinstance(shell_p, topologic.Shell), "Shell.Pie. Should be topologic.Shell"

# Case 14 - Rectangle

# test 1
shell_r = Shell.Rectangle()                                             # without optional inputs
assert isinstance(shell_r, topologic.Shell), "Shell.Rectangle. Should be topologic.Shell"
# test 2
shell_r = Shell.Rectangle(v2, width=2, length=4, uSides=3, vSides=3, direction = [0,0,1],
                        placement='lowerleft', tolerance=0.001)         # with optional inputs
assert isinstance(shell_r, topologic.Shell), "Shell.Rectangle. Should be topologic.Shell"

# Case 15 - SelfMerge

# test 1
f_smshell = Shell.SelfMerge(shell_f,0.1)
assert isinstance(f_smshell, topologic.Face), "Shell.SelfMerge. Should be list topologic.Face"
# test 2
f_smshell2 = Shell.SelfMerge(shell_r,0.1)
assert isinstance(f_smshell2, topologic.Face), "Shell.SelfMerge. Should be list topologic.Face"

# Case 16 - Vertices

# test 1
v_shell = Shell.Vertices(shell_r)
assert isinstance(v_shell, list), "Shell.Vertices. Should be list"
# test 2
v_shell2 = Shell.Vertices(shell_c)
assert isinstance(v_shell2, list), "Shell.Vertices. Should be list"

# Case 17 - Wires

# test 1
w_shell = Shell.Wires(shell_hprd)
assert isinstance(w_shell, list), "Shell.Wires. Should be list"
# test 2
w_shell2 = Shell.Wires(shell_c)
assert isinstance(w_shell2, list), "Shell.Wires. Should be list"