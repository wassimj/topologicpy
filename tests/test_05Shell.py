# Shell Classes unit test

#Importing libraries
import sys

import topologicpy 
import topologic_core as topologic
from topologicpy.Aperture import Aperture
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Dictionary import Dictionary
from topologicpy.Helper import Helper
from topologicpy.Vector import Vector
from topologicpy.Matrix import Matrix
from topologicpy.Plotly import Plotly
from topologicpy.Color import Color
from topologicpy.Context import Context
from topologicpy.Grid import Grid
from topologicpy.Sun import Sun
import math

def test_main():
    print("Start")
    # Object for test case
    v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
    v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
    v2 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
    v3 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
    v5 = Vertex.ByCoordinates(-10, 10, 0)       # create vertex
    v6 = Vertex.ByCoordinates(-10, 0, 0)        # create vertex
    v_list0 = [v0, v1, v2, v3]                     # create list
    v_list1 = [v0, v1, v5, v6]                     # create list
    wire0 = Wire.ByVertices(v_list0)            # create wire
    wire1 = Wire.ByVertices(v_list1)            # create wire
    w_list = [wire0,wire1]                      # create list
    w_cluster = Cluster.ByTopologies(w_list)    # create cluster
    face0 = Face.ByVertices(v_list0)            # create face
    face1 = Face.ByVertices(v_list1)            # create face
    f_list = [face0,face1]                      # create list
    c_faces = Cluster.ByTopologies(f_list)      # create cluster

    print("17 Cases")
    # Case 1 - ByFaces
    print("Case 1")
    # test 1
    shell_f = Shell.ByFaces(f_list)             # without tolerance
    assert Topology.IsInstance(shell_f, "Shell"), "Shell.ByFaces. Should be topologic.Shell"
    # test 2
    shell_f = Shell.ByFaces(f_list,0.001)       # with tolerance
    assert Topology.IsInstance(shell_f, "Shell"), "Shell.ByFaces. Should be topologic.Shell"

    # Case 2 - ByFacesCluster
    print("Case 2")
    # test 1
    shell_fc = Shell.ByFacesCluster(c_faces)
    assert Topology.IsInstance(shell_fc, "Shell"), "Shell.ByFacesCluster. Should be topologic.Shell"

    # Case 3 - ByWires
    print("Case 3")
    # test 1
    shell_w = Shell.ByWires(w_list, silent=True)                                     # without optional inputs
    assert Topology.IsInstance(shell_w, "Shell"), "Shell.ByFaces. Should be topologic.Shell"
    # test 2
    shell_w = Shell.ByWires(w_list, triangulate=True, tolerance=0.001, silent=True)  # with optional inputs
    assert Topology.IsInstance(shell_w, "Shell"), "Shell.ByFaces. Should be topologic.Shell"
    # test 3
    #print(" Test 3")
    ##shell_w = Shell.ByWires(w_list, triangulate=False, tolerance=0.001)  # with optional inputs
    #print(len(w_list), w_list[0], w_list[1])
    #Topology.Show(w_list[1], renderer="offline")
    #Topology.Show(shell_w, renderer="offline")
    #assert Topology.IsInstance(shell_w, "Shell"), "Shell.ByFaces. Should be topologic.Shell"

    # Case 4 - ByWiresCluster
    print("Case 4")
    # test 1
    shell_wc = Shell.ByWiresCluster(w_cluster, silent=True)               # without optional inputs
    assert Topology.IsInstance(shell_wc, "Shell"), "Shell.ByFaces. Should be topologic.Shell"
    # test 2
    shell_wc = Shell.ByWiresCluster(w_cluster, triangulate=True, tolerance=0.001, silent=True)   # with optional inputs
    assert Topology.IsInstance(shell_wc, "Shell"), "Shell.ByFaces. Should be topologic.Shell"
    # test 3
    #print(" Test 3")
    #shell_wc = Shell.ByWiresCluster(w_cluster, triangulate=False, tolerance=0.001)   # with optional inputs
    #assert Topology.IsInstance(shell_wc, "Shell"), "Shell.ByFaces. Should be topologic.Shell"

    # Case 5 - Circle
    print("Case 5")
    # test 1
    shell_c = Shell.Circle()                                                                 # without optional inputs
    assert Topology.IsInstance(shell_c, "Shell"), "Shell.Circle. Should be topologic.Shell"
    # test 2
    shell_c = Shell.Circle(v1, radius=2, sides=64, fromAngle=90, toAngle=180,
                            direction = [0, 0, 1], placement='lowerleft', tolerance=0.001)  # with optional inputs
    assert Topology.IsInstance(shell_c, "Shell"), "Shell.Circle. Should be topologic.Shell"

    # Case 6 - Edges
    print("Case 6")
    # test 1
    e_shell = Shell.Edges(shell_w)
    assert isinstance(e_shell, list), "Shell.Edges. Should be list"
    # test 2
    e_shell2 = Shell.Edges(shell_fc)
    assert isinstance(e_shell2, list), "Shell.Edges. Should be list"

    # Case 7 - ExternalBoundary
    print("Case 7")
    # test 1
    eb_shell = Shell.ExternalBoundary(shell_c)
    assert Topology.IsInstance(eb_shell, "Wire"), "Shell.ExternalBoundary. Should be Wire"
    # test 2
    eb_shell2 = Shell.ExternalBoundary(shell_wc)
    assert Topology.IsInstance(eb_shell2, "Wire"), "Shell.ExternalBoundary. Should be Wire"

    # Case 8 - Faces
    print("Case 8")
    # test 1
    f_shell = Shell.Faces(shell_wc)
    assert isinstance(f_shell, list), "Shell.Faces. Should be list"
    # test 2
    f_shell2 = Shell.Faces(shell_c)
    assert isinstance(f_shell2, list), "Shell.Faces. Should be list"

    # Case 9 - HyperbolicParaboloidCircularDomain
    print("Case 9")
    # test 1
    shell_hpcd = Shell.HyperbolicParaboloidCircularDomain()                                                 # without optional inputs
    assert Topology.IsInstance(shell_hpcd, "Shell"), "Shell.HyperbolicParaboloidCircularDomain. Should be topologic.Shell"
    # test 2

    shell_hpcd = Shell.HyperbolicParaboloidCircularDomain(v2, radius=3.7, sides=64, rings=21, A=3, B=-3,
                                                            direction = [0, 0, 1], placement='lowerleft')  # with optional inputs
    assert Topology.IsInstance(shell_hpcd, "Shell"), "Shell.HyperbolicParaboloidCircularDomain. Should be topologic.Shell"

    # Case 10 - HyperbolicParaboloidRectangularDomain
    print("Case 10")
    # test 1
    shell_hprd = Shell.HyperbolicParaboloidRectangularDomain()                                                      # without optional inputs
    assert Topology.IsInstance(shell_hprd, "Shell"), "Shell.HyperbolicParaboloidRectangularDomain. Should be topologic.Shell"
    # test 2
    shell_hprd = Shell.HyperbolicParaboloidRectangularDomain(v3, llVertex=None, lrVertex=None, ulVertex=None, urVertex=None, uSides=20,
                                                            vSides=20, direction = [0, 0, 1], placement='lowerleft')    # with optional inputs
    assert Topology.IsInstance(shell_hprd, "Shell"), "Shell.HyperbolicParaboloidRectangularDomain. Should be topologic.Shell"

    # Case 11 - InternalBoundaries
    print("Case 11")
    # test 1
    ib_shell = Shell.InternalBoundaries(shell_hpcd)
    assert isinstance(ib_shell, list), "Shell.InternalBoundaries. Should be a list"
    # test 2

    ib_shell2 = Shell.InternalBoundaries(shell_hprd)
    assert isinstance(ib_shell2, list), "Shell.InternalBoundaries. Should be a list"

    # Case 12 - IsClosed
    print("Case 12")
    # test 1

    bool_shell = Shell.IsClosed(shell_hprd)
    assert isinstance(bool_shell, bool), "Shell.IsClosed. Should be bool"
    # test 2

    bool_shell2 = Shell.IsClosed(shell_hpcd)
    assert isinstance(bool_shell2, bool), "Shell.IsClosed. Should be bool"

    # Case 13 - Pie
    print("Case 13")
    # test 1

    shell_p = Shell.Pie()                                                           # without optional inputs
    assert Topology.IsInstance(shell_p, "Shell"), "Shell.Pie. Should be topologic.Shell"
    # test 2

    shell_p = Shell.Pie(v1, radiusA=10, radiusB=5, sides=64, rings=2, fromAngle=0, toAngle=90,
                        direction = [0, 0, 1], placement='lowerleft', tolerance=0.001)             # with optional inputs
    assert Topology.IsInstance(shell_p, "Shell"), "Shell.Pie. Should be topologic.Shell"

    # Case 14 - Rectangle
    print("Case 14")
    # test 1

    shell_r = Shell.Rectangle()                                             # without optional inputs
    assert Topology.IsInstance(shell_r, "Shell"), "Shell.Rectangle. Should be topologic.Shell"
    # test 2

    shell_r = Shell.Rectangle(v2, width=2, length=4, uSides=3, vSides=3, direction = [0, 0, 1],
                            placement='lowerleft', tolerance=0.001)         # with optional inputs
    assert Topology.IsInstance(shell_r, "Shell"), "Shell.Rectangle. Should be topologic.Shell"

    # Case 15 - SelfMerge
    print("Case 15")
    # test 1

    f_smshell = Shell.SelfMerge(shell_f,0.1)
    assert Topology.IsInstance(f_smshell, "Face"), "Shell.SelfMerge. Should be list topologic.Face"
    # test 2

    f_smshell2 = Shell.SelfMerge(shell_r,0.1)
    assert Topology.IsInstance(f_smshell2, "Face"), "Shell.SelfMerge. Should be list topologic.Face"

    # Case 16 - Vertices
    print("Case 16")
    # test 1

    v_shell = Shell.Vertices(shell_r)
    assert isinstance(v_shell, list), "Shell.Vertices. Should be list"
    # test 2

    v_shell2 = Shell.Vertices(shell_c)
    assert isinstance(v_shell2, list), "Shell.Vertices. Should be list"

    # Case 17 - Wires
    print("Case 17")
    # test 1

    w_shell = Shell.Wires(shell_hprd)
    assert isinstance(w_shell, list), "Shell.Wires. Should be list"
    # test 2

    w_shell2 = Shell.Wires(shell_c)
    assert isinstance(w_shell2, list), "Shell.Wires. Should be list"
    print("End")
