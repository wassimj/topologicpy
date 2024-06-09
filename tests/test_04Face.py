# Face Classes unit test

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
    print("43 Cases")
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
    # vertices to make triangle
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
    print("Case 1")
    f1 = Face.ByVertices([v1, v2, v3, v4])                 # create face
    w1 = Wire.ByVertices([v5, v6, v7, v8], True)      # create wire
    c1 = Face.Circle(v0, 5, 16)                                 # create face
    rec1 = Wire.Rectangle(v0, .5, .5)                       # create wire
    rec2 = Wire.Rectangle(v8, .3, .3)                       # create wire
    rec3 = Wire.Rectangle(v6, .4, .4)                       # create wire
    # test 1
    intB1 = Face.AddInternalBoundaries(f1, [w1])
    assert Topology.IsInstance(intB1, "Face"), "Face.AddInternalBoundaries. Should be topologic.Face"
    # test 2
    intB2 = Face.AddInternalBoundaries(c1, [rec1, rec2])
    assert Topology.IsInstance(intB2, "Face"), "Face.AddInternalBoundaries. Should be topologic.Face"
 
    # Case 2 - AddInternalBoundariesCluster
    print("Case 2")
    # creating cluster
    clu1 = Cluster.ByTopologies([rec1, rec2])                  # create cluster
    clu2 = Cluster.ByTopologies([rec1, rec2, rec3])         # create cluster
    # test 1
    intB_C1 = Face.AddInternalBoundariesCluster(c1, clu1)
    assert Topology.IsInstance(intB_C1, "Face"), "Face.AddInternalBoundariesCluster. Should be topologic.Face"
    # test 2
    intB_C2 = Face.AddInternalBoundariesCluster(c1, clu2)
    assert Topology.IsInstance(intB_C2, "Face"), "Face.AddInternalBoundariesCluster. Should be topologic.Face"

    # Case 3 - Angle
    print("Case 3")
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
    print("Case 4")
    # test 1
    areaS = Face.Area(f1)                                                  # without optional inputs 
    assert isinstance(areaS, float), "Face.Area. Should be float"
    # test 2
    areaC = Face.Area(c1, 6)                                               # with optional inputs 
    assert isinstance(areaC, float), "Face.Area. Should be float"
 
    # Case 5 - BoundingRectangle
    print("Case 5")
    # test 1
    bF1 = Face.BoundingRectangle(f1)
    assert Topology.IsInstance(bF1, "Face"), "Face.BoundingRectangle. Should be topologic.Face"
    # test 2
    bF2 = Face.BoundingRectangle(c1)
    assert Topology.IsInstance(bF1, "Face"), "Face.BoundingRectangle. Should be topologic.Face"

    # Case 6 - BoundingRectangle
    print("Case 6")
    starF = Face.Star(v10, 5.0, 2.0, 5)                     # create face
    starF0 = Face.Star(v10, 6.0, 2.5, 6)                   # create face                                          
    # test 1
    bR1 = Face.BoundingRectangle(starF, 5)
    assert Topology.IsInstance(bR1, "Face"), "Face.BoundingRectangle. Should be topologic.Face"
    # test 2
    bR2 = Face.BoundingRectangle(starF0, 10)
    assert Topology.IsInstance(bR2, "Face"), "Face.BoundingRectangle. Should be topologic.Face"

    # Case 7 - ByEdges
    print("Case 7")
    # test 1
    fE1 = Face.ByEdges([e1, e2, e3, e4])
    assert Topology.IsInstance(fE1, "Face"), "Face.ByEdges. Should be topologic.Face"
    # test 2
    fE2 = Face.ByEdges([te1, te2, te3, te4])
    assert Topology.IsInstance(fE2, "Face"), "Face.ByEdges. Should be topologic.Face"

    # Case 8 - ByEdgesCluster
    print("Case 8")
    # creating Cluster
    Clu3 = Cluster.ByTopologies([e1, e2, e3, e4])                   # create cluster
    Clu4 = Cluster.ByTopologies([te1, te2, te3, te4])              # create cluster
    # test 1
    fcE1 = Face.ByEdgesCluster(Clu3)                                   
    assert Topology.IsInstance(fcE1, "Face"), "Face.ByEdgesCluster. Should be topologic.Fac"
    # test 2
    fcE2 = Face.ByEdgesCluster(Clu4)
    assert Topology.IsInstance(fcE2, "Face"), "Face.ByEdgesCluster. Should be topologic.Fac"
 
    # Case 9 - ByOffset
    print("Case 9")
    # test 1
    offF1 = Face.ByOffset(f1)                                                       # without optional inputs 
    assert Topology.IsInstance(offF1, "Face"), "Face.ByOffset. Should be topologic.Face"
    # test 2
    offF2 = Face.ByOffset(c1, offset=0.5)       # without optional inputs 
    assert Topology.IsInstance(offF2, "Face"), "Face.ByOffset. Should be topologic.Face"
    #  plot geometry
    # geo1 = Plotly.DataByTopology(offF2)       # create plotly data
    # plotfig1 = Plotly.FigureByData(geo1)
    # Plotly.Show(plotfig1, renderer= 'browser')
 
    # Case 10 - ByShell
    print("Case 10")
    v_s0 = Vertex.ByCoordinates(0, 0, 0)                 # create vertex
    v_s1 = Vertex.ByCoordinates(1, 0, 0)               # create vertex
    v_s2 = Vertex.ByCoordinates(1, 1, 0)              # create vertex
    v_s3 = Vertex.ByCoordinates(0, 1, 0)               # create vertex
    v_s4 = Vertex.ByCoordinates(2, -1, 0)                 # create vertex
    v_s5 = Vertex.ByCoordinates(2, 2, 0)               # create vertex
    # creating Face by vertices
    fV1 = Face.ByVertices([v_s0, v_s1, v_s2, v_s3])        # create face
    fV2 = Face.ByVertices([v_s1, v_s4, v_s5, v_s2])                 # create face
    #creating Shell by faces
    sh1 = Shell.ByFaces([fV1, fV2])                                   # create shell
    # test 1
    fs1 = Face.ByShell(sh1)                                               # without optional inputs
    assert Topology.IsInstance(fs1, "Face"), "Face.ByShell. Should be topologic.Face"

    # Case 11 - ByVertices
    print("Case 11")
    #fV = Face by vertices
    # test 1
    fV1 = Face.ByVertices([v1, v2, v3, v4])                       # with four vertices
    assert Topology.IsInstance(fV1, "Face"), "Face.ByVertices. Should be topologic.Face"
    # test 2
    fV2 = Face.ByVertices([tv1, tv2, tv3])                         # with three vertices
    assert Topology.IsInstance(fV2, "Face"), "Face.ByVertices. Should be topologic.Face"
    # plot geometry
    # face1 = Plotly.DataByTopology(fV2)                           # create plotly data
    # plotfig1 = Plotly.FigureByData(face1)
    # Plotly.Show(plotfig1)
 
    # Case 12 - ByWire
    print("Case 12")
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
    print("Case 13")
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
    assert Topology.IsInstance(fWs1, "Face"), "Face.ByWires. Should be topologic.Face"
    # test 2
    fWs2 = Face.ByWires(traW2, [cW2, cW3, cW4, cW5])                # with four wires
    assert Topology.IsInstance(fWs2, "Face"), "Face.ByWires. Should be topologic.Face"
 
    # Case 14 - ByWiresCluster
    print("Case 14")
    # creating Cluster
    clu5 = Cluster.ByTopologies([rec1, rec2, rec3])                          # create cluster
    clu6 = Cluster.ByTopologies([cW2, cW3, cW4, cW5])                # create cluster
    # test 1
    fcW1 = Face.ByWiresCluster(cW1, clu5)                                    # with optional inputs
    assert Topology.IsInstance(fcW1, "Face"), "Face.ByWiresCluster, Should be topologic.Face"
    # test 2
    fcW2 = Face.ByWiresCluster(cW1, clu6)                                    # with optional inputs
    assert Topology.IsInstance(fcW2, "Face"), "Face.ByWiresCluster, Should be topologic.Face"

    # Case 15 - Circle
    print("Case 15")
    # test 1
    cF1 = Face.Circle()                                                                    # without optional inputs
    assert Topology.IsInstance(cF1, "Face"), "Face.Circle. Should be topologic.Face"
    # test 2
    cF2 = Face.Circle(v2, 4.85, 16, direction=[0, 1, 0], tolerance= 0.01)         # with optional inputs
    assert Topology.IsInstance(cF2, "Face"), "Face.Circle. Should be topologic.Face"
    # test 3
    cF3 = Face.Circle(v2, 2.5, 16, fromAngle=45, toAngle=270,    # with optional inputs
                                direction=[5, 1, -45], placement='lowerleft', tolerance= 0.01)
    assert Topology.IsInstance(cF3, "Face"), "Face.Circle. Should be topologic.Face"
 
    # Case 16 - Compactness
    print("Case 16")
    # test 1
    fC1 = Face.Compactness(f1)                                                # without optional inputs
    assert isinstance(fC1, float), "Face.Compactness. Should be float"
    # test 2
    fC2 = Face.Compactness(cF1,5)                                           # with optional inputs
    assert isinstance(fC2, float), "Face.Compactness. Should be float"
 
    # Case 17 - CompassAngle
    print("Case 17")
    """ Description missing in documentation """

    # Case 18 - Edges
    print("Case 18")
    # test 1
    fEdg1 = Face.Edges(f1)
    assert isinstance(fEdg1, list), "Face.Edges. Should be list"
    # test 2
    fEdg2 = Face.Edges(cF1)
    assert isinstance(fEdg1, list), "Face.Edges. Should be list"

    # Case 19 - FacingToward
    print("Case 19")
    # test 1
    ft1 = Face.FacingToward(f1, [0, 0, 1], True, tolerance=0.005)                   # with optional inputs
    assert isinstance(ft1, bool), "Face.FacingToward. Should be Boolean"
    # test 2
    ft2 = Face.FacingToward(cF1)                                                 # without optional inputs
    assert isinstance(ft2, bool), "Face.FacingToward. Should be Boolean"

    # Case 21 - Harmonize
    print("Case 21")
    #test 1
    hF1 = Face.Harmonize(rec4)
    assert Topology.IsInstance(hF1, "Face"), "Face.Harmonize. Should be topologic.Face"
    #test 2
    hF2 = Face.Harmonize(rec5)
    assert Topology.IsInstance(hF2, "Face"), "Face.Harmonize. Should be topologic.Face"

    # Case 22 - ExternalBoundary
    print("Case 22")
    # test 1
    ExtB1 = Face.ExternalBoundary(intB1)
    assert Topology.IsInstance(ExtB1, "Wire"), "Face.ExternalBoundary. Should be topologic.Wire"
    # test 2
    ExtB2 = Face.ExternalBoundary(intB2)
    assert Topology.IsInstance(ExtB2, "Wire"), "Face.ExternalBoundary. Should be topologic.Wire"

    # Case 23 - InternalBoundaries
    print("Case 23")
    # test 1
    IntB1 = Face.InternalBoundaries(intB1)
    assert isinstance(IntB1, list), "Face.InternalBoundaries. Should be list"
    # test 2
    IntB2 = Face.InternalBoundaries(intB2)
    assert isinstance(IntB2, list), "Face.InternalBoundaries. Should be list"

    # Case 24 - InternalVertex
    print("Case 24")
    # test 1
    iv1 = Face.InternalVertex(f1)                                                     # without optional inputs
    assert Topology.IsInstance(iv1, "Vertex"), "Face.InternalVertex. Should be topologic.Vertex"
    # test 2
    iv2 = Face.InternalVertex(c1, 0.005)                                          # with optional inputs
    assert Topology.IsInstance(iv2, "Vertex"), "Face.InternalVertex. Should be topologic.Vertex"
 
    # Case 25 - Invert
    print("Case 25")
    # test 1
    IntF1 = Face.Invert(f1)
    assert Topology.IsInstance(IntF1, "Face"), "Face.Invert. Should be topologic.Face"
    # test 2
    IntF2 = Face.Invert(c1)
    assert Topology.IsInstance(IntF2, "Face"), "Face.Invert. Should be topologic.Face"

    # Case 26 - IsCoplanar
    print("Case 26")
    # test 1
    chkC1 = Face.IsCoplanar(fV1, fV2)               # without optional inputs
    assert isinstance(chkC1, bool), "Face.IsCoplanar. Should be boolean"
    # test 2
    ChkC2 = Face.IsCoplanar(f1, c1, mantissa=6, tolerance=0.005)        # with optional inputs
    assert isinstance(ChkC2, bool), "Face.IsCoplanar. Should be boolean"

    # Case 27 - Skeleton
    print("Case 27")
    # test 1
    mAxis1 = Face.Skeleton(fs1)
    #mAxis1 = Face.MedialAxis(fs1, resolution=10, externalVertices=False, internalVertices=False, toLeavesOnly=False, angTolerance=2, tolerance=0.0001)
    assert Topology.IsInstance(mAxis1, "Wire"), "Face.MedialAxis. Should be topologic.Wire "
    # test 2
    #mAxis2 = Face.MedialAxis(fs1, 5, True, True, True, 0.002, 0.5)
    #assert Topology.IsInstance(mAxis2, "Wire"), "Face.MedialAxis. Should be topologic.Wire "

    # Case 28 - NormalAtParameters
    print("Case 28")
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
    np2 = Face.NormalAtParameters(fV1)                             # without optional inputs
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
 
    # Case 29 - Project
    print("Case 29")
    # creating Face
    c4 = Face.Circle((Vertex.ByCoordinates(3.7,2.5,1)), .5, 6)                                # create face
    c5 = Face.Circle((Vertex.ByCoordinates(2.5, 4, 1)), 1, 16)                                # create face
    r1 = Face.Rectangle((Vertex.ByCoordinates(3.7,2.5,5)), 10, 10, direction= [0, 0, -1])        # create face
    # test 1
    pf1 = Face.Project(c4, r1)                                                              # without optional inputs
    assert Topology.IsInstance(pf1, "Face"), "Face.Project. Should be topologic.Face"
    # test 2
    pf2 = Face.Project(c5, r1, [0,0,-1], 2)                                   # with optional inputs
    assert Topology.IsInstance(pf2, "Face"), "Face.Project. Should be topologic.Face"
    # plot geometry
    # geo1 = Plotly.DataByTopology(pf2)                                          # create plotly data
    # plotfig1 = Plotly.FigureByData(geo1)
    # Plotly.Show(plotfig1)
 
    # Case 30 - Rectangle
    print("Case 30")
    # test 1
    RecF1 = Face.Rectangle()                                                                     # without optional inputs
    assert Topology.IsInstance(RecF1, "Face"), "Face.Rectangle. Should be topologic.Face"
    # test 2
    RecF2 = Face.Rectangle(v8, 3.45, 6.78, [1, 1, 1], 'lowerleft', 0.01)           # with optional inputs
    assert Topology.IsInstance(RecF1, "Face"), "Face.Rectangle. Should be topologic.Face"

    # Case 31 - Star
    print("Case 31")
    # test 1
    StarF1 = Face.Star()                                                                                  # without optional inputs
    assert Topology.IsInstance(StarF1, "Face"), "Face.Star. Should be topologic.Face"
    # test 2
    StarF2 = Face.Star(v2, 2.5,.8,8, direction=[0, 1, 0], tolerance=0.001)             # with optional inputs
    assert Topology.IsInstance(StarF2, "Face"), "Face.Star. Should be topologic.Face"
    # plot geometry
    # star1 = Plotly.DataByTopology(StarF1)                                      # create plotly data
    # plotfig1 = Plotly.FigureByData(star1)
    # Plotly.Show(plotfig1)

    # Case 32 - Trapezoid
    print("Case 32")
    # test 1
    TraF1 = Face.Trapezoid()                                                                       # without optional inputs
    assert Topology.IsInstance(TraF1, "Face"), "Face.Trapezoid. Should be topologic.Face"
    # test 2
    TraF2 = Face.Trapezoid(v7, 2.7, 1.15, .75, .25, direction= [1, 0, 0])             # with optional inputs
    assert Topology.IsInstance(TraF2, "Face"), "Face.Trapezoid. Should be topologic.Face"
    # plot geometry
    # tra1 = Plotly.DataByTopology(TraF1)                                       # create plotly data
    # plotfig1 = Plotly.FigureByData(tra1)
    # Plotly.Show(plotfig1)

    # Case 33 - Triangulate
    print("Case 33")
    # test 1
    triS1 = Face.Triangulate(c1)
    assert isinstance(triS1, list), "Face.Triangulate. Should be list"
    # test 2
    triS2 = Face.Triangulate(intB_C2)
    assert isinstance(triS1, list), "Face.Triangulate. Should be list"

    # Case 34 - TrimByWire
    print("Case 34")
    # creating Wire
    tw1 = Wire.ByVertices(
                           [(Vertex.ByCoordinates(-4.7,-0.25,10)), 
                            (Vertex.ByCoordinates(20.6,9.8,10)), 
                            (Vertex.ByCoordinates(43.8,11.8,10))]
                           )                                                     
    # test 1
    to1 = Face.TrimByWire(fV1, tw1)                                                # without optional inputs
    assert Topology.IsInstance(to1, "Face"), "Face.TrimByWire. Should be topologic.Face"
    # test 2
    to2 = Face.TrimByWire(fV2, tw1, True)                                        # with optional inputs
    assert Topology.IsInstance(to2, "Face"), "Face.TrimByWire. Should be topologic.Face"
    # plot geometry
    # trim1 = Plotly.DataByTopology(to2)                                         # create plotly data
    # plotfig1 = Plotly.FigureByData(trim1)
    # Plotly.Show(plotfig1)
 
    # Case 35 - VertexByParameters
    print("Case 35")
    # test 1
    vp1 = Face.VertexByParameters(c1)                                                 # without optional inputs
    assert Topology.IsInstance(vp1, "Vertex"), "Face.VertexByParameters. Should be topologic.Vertex"
    # test 2
    vp2 = Face.VertexByParameters(c1, .277, .65)                                  # with optional inputs
    assert Topology.IsInstance(vp1, "Vertex"), "Face.VertexByParameters. Should be topologic.Vertex"

    # Case 36 - VertexParameters
    print("Case 36")
    # test 1
    vps1 = Face.VertexParameters(c1, v11)                                                           # without optional inputs
    assert isinstance(vps1, list), "Face.VertexParameters. Should be list"
    # test 2
    vps2 = Face.VertexParameters(f1, v12, outputType='uv', mantissa=2)         # with optional inputs
    assert isinstance(vps2, list), "Face.VertexParameters. Should be list"

    # Case 37 - Vertices
    print("Case 37")
    # test 1
    fVer1 = Face.Vertices(intB_C2)
    assert isinstance(fVer1, list), "Face.Vertices. Should be list"
    # test 2
    fVer2 = Face.Vertices(intB2)
    assert isinstance(fVer2, list), "Face.Vertices. Should be list"

    # Case 38 - Wires
    print("Case 38")
    # test 1
    fWire1 = Face.Wires(intB_C2)
    assert isinstance(fWire1, list), "Face.Wires. Should be list"
    # test 2
    fWire2 = Face.Wires(intB2)
    assert isinstance(fWire2, list), "Face.Wires. Should be list"

    # Case 39 - ByVerticesCluster
    print("Case 39")
    # creating Cluster
    clu7 = Cluster.ByTopologies([v1, v2, v3, v4])                  # create cluster
    Clu8 = Cluster.ByTopologies([tv1, tv2, tv3])                   # create cluster
    # test 1
    fcV1 = Face.ByVerticesCluster(clu7)
    assert Topology.IsInstance(fcV1, "Face"), "Face.ByVerticesCluster. Should be topologic.Face"
    # test 2
    fcV2 = Face.ByVerticesCluster(Clu8)
    assert Topology.IsInstance(fcV2, "Face"), "Face.ByVerticesCluster. Should be topologic.Face"

    # Case 40 - Einstein
    print("Case 40")
    # test 1
    ein = Face.Einstein()
    assert Topology.IsInstance(ein, "Face"), "Face.Einstein. Should be a face"

    # Case 41 - Isovist
    print("Case 41")
    # test 1
    r1 = Face.Rectangle(width=10, length=5)
    r2 = Face.Rectangle(origin=Vertex.ByCoordinates(0,2,0), width=2, length=2)
    r1 = Topology.Difference(r1, r2)
    r2 = Topology.Translate(r2, -1.5, 0.4, 0)

    r1 = Topology.Merge(r1, r2)
    r1 = Shell.ExternalBoundary(r1)

    r2 = Face.Rectangle()
    r3 = Topology.Translate(r2, -0.3,-0.4,0)
    r2 = Topology.SelfMerge(Topology.Union(r2, r3))
    r4 = Wire.Rectangle(origin=Vertex.ByCoordinates(-4.5, -2, 0), width=0.2, length=0.2)
    r2 = Face.ExternalBoundary(r2)
    r5 = Topology.Translate(r2, 3, -1.5, 0)
    r6 = Topology.Translate(r5, 0.2, 0.4, 0)
    face = Face.ByWires(r1, [r2,r4,r5])

    v = Vertex.ByCoordinates(-3,-0.5,0)
    iso = Face.Isovist(face, v, obstacles=[r6])
    assert Topology.IsInstance(iso, "Face"), "Face.Isovist. Should be a face"
    
    # test 1 with parameter 
    iso = Face.Isovist(face, v, obstacles=[r6], direction=[1,0,0], fov=90)
    assert Topology.IsInstance(iso, "Face"), "Face.Isovist. Should be a face"

    # Case 42 - Simplify
    print("Case 42")
    c = Face.Circle(sides=180)
    c2 = Face.Simplify(c, tolerance=0.01)
    assert Topology.IsInstance(c2, "Face"), "Face.Simplify. Should be a face"

    # Case 43 - Squircle
    print("Case 43")
    c = Face.Squircle()
    assert Topology.IsInstance(c, "Face"), "Face.Squircle. Should be a face"
    print("End")
