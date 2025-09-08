# Grid Classes unit test

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
    print("3 Cases")
    # Object for test case
    v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
    v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
    v2 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
    v3 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
    v4 = Vertex.ByCoordinates(0, 0, 10)         # create vertex
    v5 = Vertex.ByCoordinates(0, 10, 10)        # create vertex
    v6 = Vertex.ByCoordinates(10, 10, 10)       # create vertex
    v7 = Vertex.ByCoordinates(10, 0, 10)        # create vertex
    list_v = [v0, v1, v2, v3, v4, v5, v6, v7]
    e0 = Edge.ByStartVertexEndVertex(v0, v1)
    wire0 = Wire.ByVertices([v0, v1, v2])         # create wire
    wire1 = Wire.ByVertices([v4, v5, v6])         # create wire
    w_list = [wire0,wire1]                      # create list
    face0 = Face.ByVertices([v0, v1, v2, v3, v0])     # create face
    face1 = Face.ByVertices([v4, v5, v6, v7, v4])     # create face
    face2 = Face.ByVertices([v0, v4, v7, v3, v0])     # create face
    face3 = Face.ByVertices([v3, v7, v6, v2, v3])     # create face
    face4 = Face.ByVertices([v2, v6, v5, v1, v2])     # create face
    face5 = Face.ByVertices([v1, v5, v4, v0, v1])     # create face
    f_list = [face0, face1, face2, face3, face4, face5]  # create list

    # Case 1 - EdgesByDistances
    print("Case 1")
    # test 1
    clus_ed = Grid.EdgesByDistances()
    assert Topology.IsInstance(clus_ed, "Cluster"), "Grid.ByDistances. topologic.Cluster"
    # test 2
    clus_ed1 = Grid.EdgesByDistances(face=face0, uOrigin=v0, vOrigin=v1, uRange=[-0.5, -0.25, 0, 0.25, 0.5],
                                vRange=[-0.5, -0.25, 0, 0.25, 0.5], clip=True, tolerance=0.001)
    assert Topology.IsInstance(clus_ed1, "Cluster"), "Grid.ByDistances. topologic.Cluster"
    # test 3
    clus_ed2 = Grid.EdgesByDistances(face=face5, uOrigin=v1, vOrigin=v2, uRange=[-0.5, -0.25, 0, 0.25, 0.5],
                                vRange=[-0.5, -0.25, 0, 0.25, 0.5], clip=False, tolerance=0.001)
    assert Topology.IsInstance(clus_ed2, "Cluster"), "Grid.ByDistances. topologic.Cluster"

    # Case 2 - EdgesByParameters
    print("Case 2")
    # test 1
    clus_ep = Grid.EdgesByParameters(face1)
    assert Topology.IsInstance(clus_ep, "Cluster"), "Grid.ByParameters. topologic.Cluster"
    # test 2
    clus_ep1 = Grid.EdgesByParameters(face2, uRange=[0, 0.25, 0.5, 0.75, 1.0],
                                vRange=[0, 0.25, 0.5, 0.75, 1.0], clip=False)
    assert Topology.IsInstance(clus_ep1, "Cluster"), "Grid.ByParameters. topologic.Cluster"

    # Case 3 - VerticesByDistances
    print("Case 3")
    # test 1
    clus_vd = Grid.VerticesByDistances()
    assert Topology.IsInstance(clus_vd, "Cluster"), "Grid.VerticesByDistances. topologic.Cluster"
    # test 2
    clus_vd1 = Grid.VerticesByDistances(face=face3, origin=v2, uRange=[-0.5, -0.25, 0, 0.25, 0.5],
                                        vRange=[-0.5, -0.25, 0, 0.25, 0.5], clip=False, tolerance=0.001)
    assert Topology.IsInstance(clus_vd1, "Cluster"), "Grid.VerticesByDistances. topologic.Cluster"
    # test 3
    clus_vd2 = Grid.VerticesByDistances(face=face4, origin=v1, uRange=[-0.5, -0.25, 0, 0.25, 0.5],
                                        vRange=[-0.5, -0.25, 0, 0.25, 0.5], clip=True, tolerance=0.001)
    assert Topology.IsInstance(clus_vd2, "Cluster"), "Grid.VerticesByDistances. topologic.Cluster"
    print("End")
