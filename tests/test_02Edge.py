import sys

# Edge Class unit test

#Importing libraries
import topologicpy
import topologic_core as topologic
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology

def test_main():
    print("Start")
    print("24 Cases")
    # Object for test case
    v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
    v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
    v3 = Vertex.ByCoordinates(0, -10, 0)        # create vertex
    v4 = Vertex.ByCoordinates(5, 0, 0)          # create vertex
    list_v = [v0, v1]                            # create list of vertices
    list_v1 = [v1, v3]                           # create list of vertices
    cluster_1 = Cluster.ByTopologies(list_v)    # create cluster of vertices
    cluster_2 = Cluster.ByTopologies(list_v1)   # create cluster of vertices    
    eB = Edge.ByStartVertexEndVertex(v0,v3)     # create edge

    # Case 1 - Create an edge ByStartVertexEndVertex
    print("Case 1")
    # test 1
    e1 = Edge.ByStartVertexEndVertex(v0,v1)                    # without tolerance
    assert Topology.IsInstance(e1, "Edge"), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"
    # test 2
    e1 = Edge.ByStartVertexEndVertex(v0,v1, tolerance=0.001)   # with tolerance (optional)
    assert Topology.IsInstance(e1, "Edge"), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"
    # test 3
    e2 = Edge.ByStartVertexEndVertex(v1, v3)                   # without tolerance
    assert Topology.IsInstance(e2, "Edge"), "Edge.ByStartVertexEndVertex. Should be topologic.Edge"

    # Case 2 - Create an edge ByVertices
    print("Case 2")
    # test 1
    e3 = Edge.ByVertices(list_v)                    # without tolerance
    assert Topology.IsInstance(e3, "Edge"), "Edge.ByVertices. Should be topologic.Edge"
    # test 2
    e3 = Edge.ByVertices(list_v, tolerance=0.001)   # with tolerance (optional) 
    assert Topology.IsInstance(e3, "Edge"), "Edge.ByVertices. Should be topologic.Edge"
    # test 3
    e4 = Edge.ByVertices(list_v1, tolerance=0.001)  # with tolerance (optional) 
    assert Topology.IsInstance(e4, "Edge"), "Edge.ByVertices. Should be topologic.Edge"                                                  

    # Case 3 - Create an edge ByVerticesCluster
    print("Case 3")
    # test 1
    e5 = Edge.ByVerticesCluster(cluster_1)                  # without tolerance
    assert Topology.IsInstance(e5, "Edge"), "Edge.ByVerticesCluster. Should be topologic.Edge"
    # test 2
    e5 = Edge.ByVerticesCluster(cluster_1, tolerance=0.001) # with tolerance (optional)
    assert Topology.IsInstance(e5, "Edge"), "Edge.ByVerticesCluster. Should be topologic.Edge"
    # test 3
    e6 = Edge.ByVerticesCluster(cluster_2)                  # without tolerance
    assert Topology.IsInstance(e6, "Edge"), "Edge.ByVerticesCluster. Should be topologic.Edge"

    # Case 4 - Angle
    print("Case 4")
    e7 = Edge.ByStartVertexEndVertex(v0,v4)         #create edge
    # test 1
    angle = Edge.Angle(e1,e7)                                # without optional inputs
    assert isinstance(angle, float), "Edge.Angle. Should be float"
    # test 2
    angle = Edge.Angle(e1,e7,mantissa=2, bracket=True)       # with optional inputs
    assert isinstance(angle, float), "Edge.Angle. Should be float"
    # test 3
    angle1 = Edge.Angle(e1,e5)                               # without optional inputs
    assert isinstance(angle1, float), "Edge.Angle. Should be float"

    # Case 5 - Bisect
    print("Case 5")
    # test 1
    e_bis = Edge.Bisect(e1, eB)                                                # without optional inputs
    assert Topology.IsInstance(e_bis, "Edge"), "Edge.Bisect. Should be topologic.Edge" 
    # test 2
    e_bis = Edge.Bisect(e1, eB, length=1.0, placement=1, tolerance=0.001)      # with optional inputs
    assert Topology.IsInstance(e_bis, "Edge"), "Edge.Bisect. Should be topologic.Edge" 

    # Case 6 - Direction
    print("Case 6")
    # test 1
    direction = Edge.Direction(e2)                   # without optional inputs
    assert isinstance(direction, list), "Edge.Direction. Should be list"
    # test 2
    direction1 = Edge.Direction(e1)                  # without optional inputs
    assert isinstance(direction1, list), "Edge.Direction. Should be list"
    # test 3
    direction1 = Edge.Direction(e1,mantissa=3)       # with optional inputs
    assert isinstance(direction1, list), "Edge.Direction. Should be list"

    # Case 7 - EndVertex
    print("Case 7")
    # test 1
    end2 = Edge.EndVertex(e2)
    assert Topology.IsInstance(end2, "Vertex"), "Edge.EndVertex. Should be topologic.Vertex"
    # test 2
    end3 = Edge.EndVertex(e3)
    assert Topology.IsInstance(end3, "Vertex"), "Edge.EndVertex. Should be topologic.Vertex"

    # Case 8 - Extend
    print("Case 8")
    # test 1
    extend2 = Edge.Extend(e2)                                                               # without optional inputs
    assert Topology.IsInstance(extend2, "Edge"), "Edge.Extend. Should be topologic.Edge"
    # test 2
    extend3 = Edge.Extend(e3)                                                               # without optional inputs
    assert Topology.IsInstance(extend3, "Edge"), "Edge.Extend. Should be topologic.Edge"
    # test 3
    extend3 = Edge.Extend(e3,distance=2, bothSides=False, reverse=True, tolerance=0.001)    # with optional inputs
    assert Topology.IsInstance(extend3, "Edge"), "Edge.Extend. Should be topologic.Edge"

    # Case 9 - IsCollinear (True)
    print("Case 9")
    e5 = Edge.ByStartVertexEndVertex(v0,v3)
    # test 1
    col_1 = Edge.IsCollinear(e1,e5, mantissa=3)                                         # without optional inputs
    assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"
    # test 2
    col_2 = Edge.IsCollinear(e1,e3, mantissa=3)                                         # without optional inputs
    assert isinstance(col_2, bool), "Edge.IsCollinear. Should be bool"
    # test 3
    col_1 = Edge.IsCollinear(e1,e5, mantissa=3, angTolerance=0.01, tolerance=0.001)     # with optional inputs
    assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"

    # Case 10 - IsParallel
    print("Case 10")
    # test 1
    par_1 = Edge.IsParallel(e1,e4)                                  # without optional inputs
    assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"
    # test 2
    par_2 = Edge.IsParallel(e1,e3)                                  # without optional inputs
    assert isinstance(par_2, bool), "Edge.IsParallel. Should be bool"
    # test 3
    par_1 = Edge.IsParallel(e1,e4, mantissa=2, angTolerance=0.01)   # with optional inputs
    assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"

    # Case 11 - Length
    print("Case 11")
    # test 1
    len_1 = Edge.Length(e1)               # without optional inputs
    assert isinstance(len_1, float), "Edge.Length. Should be float"
    # test 2
    len_2 = Edge.Length(e2)               # without optional inputs
    assert isinstance(len_2, float), "Edge.Length. Should be float"
    # test 3
    len_1 = Edge.Length(e1, mantissa=3)   # with optional inputs
    assert isinstance(len_1, float), "Edge.Length. Should be float"

    # Case 12 - Normalize
    print("Case 12")
    # test 1
    normal_3 = Edge.Normalize(e3)                     # without optional inputs
    assert Topology.IsInstance(normal_3, "Edge"), "Edge.Normalize. Should be topologic.Edge"
    # test 2
    normal_4 = Edge.Normalize(e4)                     # without optional inputs
    assert Topology.IsInstance(normal_4, "Edge"), "Edge.Normalize. Should be topologic.Edge"
    # test 3
    normal_4 = Edge.Normalize(e4, useEndVertex=True)  # with optional inputs
    assert Topology.IsInstance(normal_4, "Edge"), "Edge.Normalize. Should be topologic.Edge"

    # Case 13 - ParameterAtVertex
    print("Case 13")
    # test 1
    param1 = Edge.ParameterAtVertex(e2,v1)              # without optional inputs
    assert isinstance(param1, float), "Edge.ParameterAtVertex. Should be float"
    # test 2
    param2 = Edge.ParameterAtVertex(e1,v1, mantissa=3)  # with optional inputs
    assert isinstance(param2, float), "Edge.ParameterAtVertex. Should be float"

    # Case 14 - Reverse
    print("Case 14")
    # test 1
    reverse3 = Edge.Reverse(e3)
    assert Topology.IsInstance(reverse3, "Edge"), "Edge.Reverse. Should be topologic.Edge"
    # test 2
    reverse4 = Edge.Reverse(e4)
    assert Topology.IsInstance(reverse4, "Edge"), "Edge.Reverse. Should be topologic.Edge"

    # Case 15 - SetLength
    print("Case 15")
    # test 1
    SetLen1 = Edge.SetLength(e1)                                                            # without optional inputs
    assert Topology.IsInstance(SetLen1, "Edge"), "Edge.SetLength. Should be topologic.Edge"
    # test 2
    SetLen2 = Edge.SetLength(e2)                                                            # without optional inputs
    assert Topology.IsInstance(SetLen2, "Edge"), "Edge.SetLength. Should be topologic.Edge"
    # test 3
    SetLen1 = Edge.SetLength(e1, length=2, bothSides=False, reverse=True, tolerance=0.001)  # with optional inputs
    assert Topology.IsInstance(SetLen1, "Edge"), "Edge.SetLength. Should be topologic.Edge"

    # Case 16 - StartVertex
    print("Case 16")
    # test 1
    iV = Edge.StartVertex(e1)
    assert Topology.IsInstance(iV, "Vertex"), "Edge.StartVertex. Should be topologic.Vertex"
    # test 2
    iV1 = Edge.StartVertex(e2)
    assert Topology.IsInstance(iV1, "Vertex"), "Edge.StartVertex. Should be topologic.Vertex"

    # Case 17 - Trim
    print("Case 17")
    # test 1
    trim3 = Edge.Trim(e3)                                                               # without optional inputs
    assert Topology.IsInstance(trim3, "Edge"), "Edge.Trim. Should be topologic.Edge"
    # test 2
    trim4 = Edge.Trim(e4)                                                               # without optional inputs
    assert Topology.IsInstance(trim4, "Edge"), "Edge.Trim. Should be topologic.Edge"
    # test 3
    trim4 = Edge.Trim(e4, distance=1, bothSides=False, reverse=True, tolerance=0.001)   # with optional inputs
    assert Topology.IsInstance(trim4, "Edge"), "Edge.Trim. Should be topologic.Edge"

    # Case 18 - VertexByDistance
    print("Case 18")
    # test 1
    dist1 = Edge.VertexByDistance(e1)                                           # without optional inputs
    assert Topology.IsInstance(dist1, "Vertex"), "Edge.VertexByDistance. Should be topologic.Vertex"
    # test 2
    dist2 = Edge.VertexByDistance(e2)                                           # without optional inputs
    assert Topology.IsInstance(dist2, "Vertex"), "Edge.VertexByDistance. Should be topologic.Vertex"
    # test 3
    dist2 = Edge.VertexByDistance(e2, distance=1, origin=v3, tolerance=0.001)   # with optional inputs
    assert Topology.IsInstance(dist2, "Vertex"), "Edge.VertexByDistance. Should be topologic.Vertex"

    # Case 19 - VertexByParameter
    print("Case 19")
    # test 1
    ByParam3 = Edge.VertexByParameter(e3)                  # without optional inputs
    assert Topology.IsInstance(ByParam3, "Vertex"), "Edge.VertexByParameter. Should be topologic.Vertex"
    # test 2
    ByParam4 = Edge.VertexByParameter(e4)                  # without optional inputs
    assert Topology.IsInstance(ByParam4, "Vertex"), "Edge.VertexByParameter. Should be topologic.Vertex"
    # test 3
    ByParam4 = Edge.VertexByParameter(e4, u=0.7)   # with optional inputs
    assert Topology.IsInstance(ByParam4, "Vertex"), "Edge.VertexByParameter. Should be topologic.Vertex"

    #Case 20 - Vertices
    print("Case 20")
    # test 1
    v_e5 = Edge.Vertices(e5)
    assert isinstance(v_e5, list), "Edge.Vertices. Should be list"
    # test 2
    v_e6 = Edge.Vertices(e6)
    assert isinstance(v_e6, list), "Edge.Vertices. Should be list"

    #Case 21 - ByFaceNormal
    print("Case 21")
    # test 1
    from topologicpy.Face import Face
    face = Face.Rectangle()
    edge = Edge.ByFaceNormal(face)
    assert Topology.IsInstance(edge, "Edge"), "Edge.ByFaceNormal. Should be topologic.Edge"
    # test 2
    face = Face.Rectangle()
    edge = Edge.ByFaceNormal(face, length=3)
    assert Edge.Length(edge) == 3, "Edge.ByFaceNormal. Length should be 3"

    #Case 22 - ByOffset2D
    print("Case 12")
    # test 1
    from topologicpy.Topology import Topology
    v1 = Vertex.ByCoordinates(0, 0, 0)
    v2 = Vertex.ByCoordinates(10,0,0)
    edge = Edge.ByVertices([v1, v2])
    edge2 = Edge.ByOffset2D(edge, offset=1)
    assert Topology.IsInstance(edge2, "Edge"), "Edge.ByOffset2D. Should be topologic.Edge"
    centroid = Topology.Centroid(edge2)
    assert Vertex.X(centroid) == 5, "Edge.ByOffset2D. X Should be 5"
    assert Vertex.Y(centroid) == 1, "Edge.ByOffset2D. Y Should be 1"

    #Case 23 - ExtendToEdge2D
    print("Case 23")
    # test 1
    v1 = Vertex.ByCoordinates(0, 0, 0)
    v2 = Vertex.ByCoordinates(10,0,0)
    edge = Edge.ByVertices([v1, v2])
    v1 = Vertex.ByCoordinates(20,-10,0)
    v2 = Vertex.ByCoordinates(20,10,0)
    edge2 = Edge.ByVertices([v1, v2])
    edge3 = Edge.ExtendToEdge2D(edge, edge2)
    assert Topology.IsInstance(edge3, "Edge"), "Edge.ExtendToEdge2D. Should be topologic.Edge"
    assert Edge.Length(edge3) == 20, "Edge.ExtendToEdge2D. Length should be 3"
    centroid = Topology.Centroid(edge3)
    assert Vertex.X(centroid) == 10, "Edge.ExtendToEdge2D. X Should be 5"
    assert Vertex.Y(centroid) == 0, "Edge.ExtendToEdge2D. Y Should be 1"

    #Case 24 - Intersect2D
    print("Case 24")
    # test 1
    v1 = Vertex.ByCoordinates(0, 0, 0)
    v2 = Vertex.ByCoordinates(10,0,0)
    edge = Edge.ByVertices([v1, v2])
    v1 = Vertex.ByCoordinates(5,-10,0)
    v2 = Vertex.ByCoordinates(5,10,0)
    edge2 = Edge.ByVertices([v1, v2])
    v3 = Edge.Intersect2D(edge, edge2)
    assert Topology.IsInstance(v3, "Vertex"), "Edge.Intersect2D. Should be topologic.Edge"
    assert Vertex.X(v3) == 5, "Edge.Intersect2D. X Should be 5"
    assert Vertex.Y(v3) == 0, "Edge.Intersect2D. Y Should be 0"
    print("End")
