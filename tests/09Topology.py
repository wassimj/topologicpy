# Topology Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

from ast import FloorDiv
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
from topologicpy.Dictionary import Dictionary
from topologicpy.Matrix import Matrix
from topologicpy.Graph import Graph
from topologicpy.Plotly import  Plotly

print("Start")
print("65 Cases")
# Object for test
v0 = Vertex.ByCoordinates(0, 0, 0)                                                                   # create vertex
v1 = Vertex.ByCoordinates(0, 10, 0)                                                                 # create vertex
v2 = Vertex.ByCoordinates(10, 10, 0)                                                               # create vertex
v3 = Vertex.ByCoordinates(10, 0, 0)                                                                 # create vertex
v4 = Vertex.ByCoordinates(0, 0, 10)                                                                 # create vertex
v5 = Vertex.ByCoordinates(0, 10, 10)                                                               # create vertex
v6 = Vertex.ByCoordinates(5, 8, 0)                                                                   # create vertex
v7 =Vertex.ByCoordinates(2.5, 2.5, 2.5)                                                            # create vertex
v8 = Vertex.ByCoordinates(5, 5, 5)                                                                   # create vertex
w1 = Wire.ByVertices([v0, v2, v4])                                                                    # create wire
# Creating 3D objects
box1 = CellComplex.Box(v0, 5, 5, 5)                                                                # create box
box2 = CellComplex.Box(v3, 1.5, 3, 5)                                                             # create box
box3 = CellComplex.Box(v5, 1.2, 1.2, 1.2)                                                       # create box
box4 = CellComplex.Box(v7, 5, 5, 5)                                                               # create box
box5 = CellComplex.Box(v0, 5, 5, 5, uSides=1, vSides=1, wSides=1)
prism1 = CellComplex.Prism(v2, .5, .5, 2)                                                        # create prism
prism2 = CellComplex.Prism(v6, 1.5, 1, 3)                                                       # create prism
prism3 = CellComplex.Prism(v2, .5, .5, 2, direction = [90, 90, 90])             # create prism

# test 4.2 - symdif
symdif2 = Topology.Boolean(box1, box4, operation= 'symdif',              # with optional inputs
                                               tranDict=False, tolerance= 0.0001)

# case 1 - AddApertures
print("Case 1")
extF1 = CellComplex.Faces(box5)                                                                                  # extract faces
extF2 = CellComplex.ExternalFaces(prism1)                                                                  # extract faces
offF1 = []                                                                                                                       # create empty list for offset face
for face1 in extF1:
    offF1.append(Topology.Scale(face1, Topology.Centroid(face1), x=.5, y=.5, z=.5))   # offset face
offF2 = []                                                                                                                       # create empty list for offset face
for face2 in extF2:
    offF2.append(Topology.Scale(face2, Topology.Centroid(face2), x=.4, y=.4, z=.4))   # offset face
# test 1
addA1 = Topology.AddApertures(box5, offF1)                                                            # without optional inputs
assert isinstance(addA1, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
# test 2
addA2 = Topology.AddApertures(prism1, offF2,  exclusive=False, tolerance=0.0001) # with optional inputs
assert isinstance(addA2, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
# test 2.1 - vertex
addA2Ver = Topology.AddApertures(prism1, offF2,  exclusive=False,                          # with optional inputs
                                                           subTopologyType= 'vertex', tolerance=0.0001) 
assert isinstance(addA2Ver, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
# test 2.2 - edge
addA2Edg = Topology.AddApertures(prism1, offF2,  exclusive=False,                          # with optional inputs
                                                            subTopologyType= 'edge', tolerance=0.0001)
assert isinstance(addA2Edg, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
# test 2.3 - face
addA2Face = Topology.AddApertures(prism1, offF2,  exclusive=False,                         # with optional inputs
                                                             subTopologyType= 'face', tolerance=0.0001)
assert isinstance(addA2Face, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
# test 2.4 - cell
addA2Cell = Topology.AddApertures(prism1, offF2,  exclusive=False,                         # with optional inputs
                                                            subTopologyType= 'cell', tolerance=0.0001)
assert isinstance(addA2Cell, topologic.Topology), "Topology.AddApertures. Should be topology.Topology"
#plot geometry
# geo1 = Plotly.DataByTopology(addA2)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 2 - AddContent
print("Case 2")
contents1 = [box2, prism2]                                            # create content
contents2 = [box1, box3]                                              # create content
# test 1
addC1 = Topology.AddContent(box1, contents1)
assert isinstance(addC1, topologic.Topology), "Topology.AddContent. Should be topologic.Topologic"
# test 2
addC2 = Topology.AddContent(box2, contents2)
assert isinstance(addC2, topologic.Topology), "Topology.AddContent. Should be topologic.Topologic"
# plot geometry
# geo1 = Plotly.DataByTopology(addC1)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 3 - AddDictionary
print("Case 3")
Dict1 = Dictionary.ByKeysValues([0], ["box1"])                                      # create dictionary
Dict2 = Dictionary.ByKeysValues([1], ["prism1"])                                   # create dictionary
# test 1
AddD1 = Topology.AddDictionary(box1, Dict1)
assert isinstance(AddD1, topologic.Topology), "Topology.AddDictionary.  Should be topologic.Topology"
# test 2
AddD2 = Topology.AddDictionary(prism1, Dict2)
assert isinstance(AddD2, topologic.Topology), "Topology.AddDictionary.  Should be topologic.Topology"

# case 4 - AdjacentTopologies
print("Case 4")
box6 = Cell.Box(v0, 5, 10, 5)                           # create box
box7 = Cell.Box(v1, 4, 8, 2)                             # create box
boxE1 = Cell.Edges(box6)                             # extract box edges
boxF1 = Cell.Faces(box7)                              # extract box faces
# test 1
adT1 = Topology.AdjacentTopologies(box6, boxE1[0])
assert isinstance(adT1, list), "Topology.AdjacentTopologies. Should be list"
# test 2
adT2 = Topology.AdjacentTopologies(box7, boxF1[3], topologyType="edge")
assert isinstance(adT2, list), "Topology.AdjacentTopologies. Should be list"
# plot geometry
# cl = Cluster.ByTopologies(adT2)                          # cluster for visualisation
# geo1 = Plotly.DataByTopology(cl)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 5 - Analyze
print("Case 5")
# test 1
str1 = Topology.Analyze(box1)
assert isinstance(str1, str), "Topology.Analyze. Should be string"
# test 2
str2 = Topology.Analyze(prism2)
assert isinstance(str2, str), "Topology.Analyze. Should be string"

# case 6 - ApertureTopologies
print("Case 6")
# test 1
aperT1 = Topology.ApertureTopologies(addA1)
assert isinstance(aperT1, list), "Topology.ApertureTopologies. Should be list"
# test 2
aperT2 = Topology.ApertureTopologies(addA2)
assert isinstance(aperT2, list), "Topology.ApertureTopologies. Should be list"

# case 7 - Apertures
print("Case 7")
# test 1
aper1 = Topology.Apertures(addA1)
assert isinstance(aper1, list), "Topology.Apertures. Should be list"
# test 2
aper2 = Topology.Apertures(addA2)
assert isinstance(aper2, list), "Topology.Apertures. Should be list"

# case 8 - Boolean
print("Case 8")
# test 1 - Union
Union1 = Topology.Boolean(box1, box4)                                                      # without optional inputs                                         
# test 2 - Union
Union2 = Topology.Boolean(prism1, prism3, operation= 'union',                 # with optional inputs
                                              tranDict=False, tolerance= 0.0005)
# test 3 - Difference
diff1 = Topology.Boolean(box1, box4, operation= 'difference')
# test 4 -Difference
diff2 = Topology.Boolean(prism1, prism3, operation= 'difference',                 # with optional inputs     
                                              tranDict=False, tolerance= 0.0005)

# test 5 - Intersect
intrsct1 = Topology.Boolean(box1, box4, operation= 'intersect')

# test 6 - Intersect
intrsct2 = Topology.Boolean(box1, box4, operation= 'intersect',              # with optional inputs
                                               tranDict=False, tolerance= 0.0001)   

# test 7 - Intersect
intrsct3 = Topology.Boolean(box1, box4, operation= 'intersect',              # with optional inputs
                                               tranDict=False, tolerance= 0.0001)   

# test 8 - symdif
symdif1 = Topology.Boolean(box1, box4, operation= 'symdif')
# test 9 - symdif
symdif2 = Topology.Boolean(box1, box4, operation= 'symdif',              # with optional inputs
                                               tranDict=False, tolerance= 0.0001)
# test 10 - symdif
symdif3 = Topology.Boolean(box1, box4, operation= 'symdif',              # with optional inputs
                                               tranDict=False, tolerance= 0.0001)
# test 11 - merge
merge1 = Topology.Boolean(box1, box4,  operation= 'merge')
# test 12 - merge
merge2 = Topology.Boolean(prism1, prism3, operation= 'merge', 
                                               tranDict=False, tolerance=0.0005)
# test 13 - slice
slice1 = Topology.Boolean(box2, box1, operation= 'slice')
# test 6.2 - slice
slice2 = Topology.Boolean(prism1, prism3, operation= 'slice',
                                            tranDict=False, tolerance=0.0005)
# test 14 - impose
impo1 = Topology.Boolean(box2, box1, operation= 'impose')
# test 15 - impose
impo2 = Topology.Boolean(prism1, prism3, operation= 'impose',
                                            tranDict=False, tolerance=0.0005)
# test 16 - imprint
impr1 = Topology.Boolean(box2, box1, operation= 'imprint')
# test 17 - imprint
impr2 = Topology.Boolean(prism1, prism3, operation= 'imprint',
                                            tranDict=False, tolerance=0.0005)

# case 9 - BoundingBox
print("Case 9")
# test 1
bbox1 = Topology.BoundingBox(prism3)                                        # without optional inputs
assert isinstance(bbox1, topologic.Cell), "Topology.BoundingBox. Should be topology.Cell"
# test 2.1
bbox2x = Topology.BoundingBox(prism3, 10, "x")                          # with optional inputs           
assert isinstance(bbox2x, topologic.Cell), "Topology.BoundingBox. Should be topology.Cell"
# test 2.2
bbox2y = Topology.BoundingBox(prism3, 10, "y")                          # with optional inputs  
assert isinstance(bbox2y, topologic.Cell), "Topology.BoundingBox. Should be topology.Cell"
# test 2.3
bbox2z = Topology.BoundingBox(prism3, 10, "z")                          # with optional inputs  
assert isinstance(bbox2z, topologic.Cell), "Topology.BoundingBox. Should be topology.Cell"
#plot geometry
# geo1 = Plotly.DataByTopology(bbox1)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 10 - ByImportedBRep
print("Case 10")
# test 1
#Brep1 = Topology.ByImportedBRep('../Export/bEB2.brep')
#assert isinstance(Brep1, topologic.Topology), "Topology.ByImportedBRep. Should be topologic.Topology"
# test 2
#Brep2 = Topology.ByImportedBRep('../Export/fcE2.brep')
#assert isinstance(Brep1, topologic.Topology), "Topology.ByImportedBRep. Should be topologic.Topology"

# case 11 - ByImportedJSONMK1
""" Not tested """
# case 12 - ByImportedJSONMK2
""" Not tested """
# case 13 - ByOCCTShape
""" Not tested """

# case 14 - ByBREPString
print("Case 14")
# test 1
g1 = Topology.ByBREPString(Topology.BREPString(box1))
assert isinstance(g1,topologic.Topology), "Topology.ByBREPString. Should be topologic.Topology"
# test 2
g2 = Topology.ByBREPString(Topology.BREPString(box2))
assert isinstance(g1,topologic.Topology), "Topology.ByBREPString. Should be topologic.Topology"

# case 15 - CenterOfMass
print("Case 15")
# test 1
com1 = Topology.CenterOfMass(box1)
assert isinstance(com1, topologic.Vertex), "Topology.Centroid. Should be topologic.Vertex"
# test 2
com2 = Topology.CenterOfMass(prism3)
assert isinstance(com2, topologic.Vertex), "Topology.Centroid. Should be topologic.Vertex"

# case 16 - Centroid
print("Case 16")
# test 1
cent1 = Topology.Centroid(box1)
assert isinstance(cent1, topologic.Vertex), "Topology.Centroid. Should be topologic.Vertex"
# test 2
cent2 = Topology.Centroid(prism3)
assert isinstance(cent2, topologic.Vertex), "Topology.Centroid. Should be topologic.Vertex"

# case 17 - ClusterFaces
print("Case 17")
# test 1
cluF1 = Topology.ClusterFaces(box1, 0.0005)
assert isinstance(cluF1, list), "Topology.ClusterFaces. Should be list"
# test 2
cluF2 = Topology.ClusterFaces(prism1, 0.0005)
assert isinstance(cluF2, list), "Topology.ClusterFaces. Should be list"

# case 18 - Contents
print("Case 18")
# test 1
contnt1 = Topology.Contents(addC1)
assert isinstance(contnt1, list), "Topology.Contents. Should be list"
# test 2
contnt2 = Topology.Contents(addC2)
assert isinstance(contnt2, list), "Topology.Contents. Should be list"

# case 19 - Contexts
print("Case 19")
# test 1
contxt1 = Topology.Contexts(contnt1[0])
assert isinstance(contxt1, list), "Topology.Contexts. Should be list"
# test 2
contxt2 = Topology.Contexts(contnt2[1])
assert isinstance(contxt2, list), "Topology.Contexts. Should be list"

# case 20 - ConvexHull
print("Case 20")
# test 1
conV1 = Topology.ConvexHull(box1)
assert isinstance(conV1, topologic.Topology), "Topology.ConvexHull. Should be topologic.Topology"
# test 2
conV2 = Topology.ConvexHull(prism1)
assert isinstance(conV2, topologic.Topology), "Topology.ConvexHull. Should be topologic.Topology"

# case 21 - Copy
print("Case 21")
# test 1
copy1 = Topology.Copy(box1)
assert isinstance(copy1, topologic.Topology), "Topology.Copy. Should be topologic.Topology"
# test 2
copy2 = Topology.Copy(prism3)
assert isinstance(copy2, topologic.Topology), "Topology.Copy. Should be topologic.Topology"

# case 22 - Dictionary
print("Case 22")
# test 1
Dic1 = Topology.Dictionary(box2)
assert isinstance(Dic1, topologic.Dictionary), "Topology.Dictionary. Should be topologic.Dictionary"
# test 2
Dic2 = Topology.Dictionary(prism3)
assert isinstance(Dic2, topologic.Dictionary), "Topology.Dictionary. Should be topologic.Dictionary"

# case 23 - Dimensionality
print("Case 23")
# test 1
Dim1 = Topology.Dimensionality(box3)
assert isinstance(Dim1, int), "Topology.Dimensionality. Should be integer"
# test 2
Dim2 = Topology.Dimensionality(prism2)
assert isinstance(Dim2, int), "Topology.Dimensionality. Should be integer"

# case 24 - Divide
print("Case 24")
prism4 = Cell.Prism(v1, 10, 10, 5)                                                       # create prism
face1 = Face.Rectangle(v1, 20, 20, direction=[180, 1, 1])                                      # create face
face2 = Face.Rectangle(v1, 20, 20, direction=[1, 180, 1])                                      # create face
# test 1
div1 = Topology.Divide(prism4, face1)                                              # without optional inputs
assert isinstance(div1, topologic.Topology), "Topology.Divide. Should be topologic.Topology"
# to check childern of parent topology
# divCnt1 = Topology.Contents(div1)
# print(divCnt1," contents1")
# test 2
div2 = Topology.Divide(div1, face2, False, False)                                # wit optional inputs
assert isinstance(div2, topologic.Topology), "Topology.Divide. Should be topologic.Topology"
# to check childern of parent topology
# divCnt2 = Topology.Contents(div2)
# print(divCnt2," contents2")
#plot geometry
# geo1 = Plotly.DataByTopology(div2)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 25 - Explode
print("Case 25")
# test 1
exp1 = Topology.Explode(box1)                                                    # without optional inputs
assert isinstance(exp1, topologic.Cluster), "Topology.Explode. Should be topologic.Cluster"
# test 2
exp2 = Topology.Explode(prism1, v1, 1.5 , "face", "y")                  # with optional inputs
assert isinstance(exp2, topologic.Cluster), "Topology.Explode. Should be topologic.Cluster"
#plot geometry
# geo1 = Plotly.DataByTopology(exp2)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')

# case 26 - ExportToBRep
print("Case 26")
# test 1
#export1 = Topology.ExportToBRep(box1, r"E:\UK_Work\Topologic_Work\Export\box01.brep",True)
#assert isinstance(export1, bool), "Topology.ExportToBRep. Should be boolean"
# test 2
#export2 = Topology.ExportToBRep(prism1, r"E:\UK_Work\Topologic_Work\Export\prism01.brep",True)
#assert isinstance(export2, bool), "Topology.ExportToBRep. Should be boolean"

# case 27 - ExportToJSONMK1
""" Not tested """
# case 28 - ExportToJSONMK2
""" Not tested """

# case 29 - Filter
""" Incomplete
Works with single topology type.
Creates empty list at first then gives the output list
"""
w2 = Wire.ByVertices([v1, v3, v5])
w3 = Wire.ByVertices([v2, v4, v6, v8])
edg1 = Edge.ByVertices([v0, v1])
edg2 = Edge.ByVertices([v3, v5])
cbox1 = Cell.Box()
cPrism1 = Cell.Prism()
# test 1
# filter1 = Topology.Filter([box1, box3, box2, v1])
# test 2
# filter2 = Topology.Filter([v1])
#plot geometry
# geo1 = Plotly.DataByTopology(filter1)                       # create plotly data
# plotfig1 = Plotly.FigureByData(geo1)
# Plotly.Show(plotfig1, renderer= 'browser')
# print(filter1, "-------------------", filter2)

# case 30 - Geometry
print("Case 30")
# test 1
geom1 = Topology.Geometry(box1)
assert isinstance(geom1, dict), "Topology.Geometry. Should be dictionary"
# test 2
geom2 = Topology.Geometry(prism1)
assert isinstance(geom2, dict), "Topology.Geometry. Should be dictionary"

# case 31 - HighestType
print("Case 31")
# test 1
hT1 = Topology.HighestType(box1)
assert isinstance(hT1, int), "Topology.HighestType. Should be integer"
# test 2
hT2 = Topology.HighestType(box2)
assert isinstance(hT2, int), "Topology.HighestType. Should be integer"

# Case 32 - Impose
#print("Case 32")
#test 1
#impo1 = Topology.Impose(box2, box1)                                        # without optional inputs
#assert isinstance(impo1, topologic.Topology),  "Topology.Impose. Should be topologic.Topology"
#test 2
#impo2 = Topology.Impose(prism3, prism1, True, 0.0001)
#assert isinstance(impo2, topologic.Topology),  "Topology.Impose. Should be topologic.Topology"

# Case 33 - Imprint
#print("Case 33")
#test 1
#impr1 = Topology.Imprint(box2, box1)                                        # without optional inputs
#assert isinstance(impr1, topologic.Topology),  "Topology.Impose. Should be topologic.Topology"
# test 2
#impr2 = Topology.Imprint(prism3, prism1, True, 0.0001)
#assert isinstance(impr2, topologic.Topology),  "Topology.Impose. Should be topologic.Topology"

# Object for test case
v00 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
v01 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
v02 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
v03 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
v04 = Vertex.ByCoordinates(0, 0, 10)         # create vertex
v05 = Vertex.ByCoordinates(0, 20, 0)         # create vertex
e00 = Edge.ByStartVertexEndVertex(v00,v01)   # create edge
e01 = Edge.ByStartVertexEndVertex(v00,v02)   # create edge
e02 = Edge.ByStartVertexEndVertex(v00,v03)   # create edge

# Case 34 - InternalVertex
print("Case 34")
# test 1
cell_cy = Cell.Cylinder()
topology_iv = Topology.InternalVertex(cell_cy)
assert isinstance(topology_iv, topologic.Vertex), "Topology.InternalVertex. Should be topologic.Vertex"

# Case 35 - IsPlanar
print("Case 35")
# test 1
cluster_pts = Cluster.ByTopologies([v00,v01,v02,v03])   # all points in xy plane
topology_ip = Topology.IsPlanar(cluster_pts)        # without optional inputs
assert isinstance(topology_ip, bool), "Topology.IsPlanar. Should be bool"
assert topology_ip == True, "Topology.IsPlanar. Should be True"
# test 2
cluster_pts2 = Cluster.ByTopologies([v00,v01,v02,v04])
topology_ip2 = Topology.IsPlanar(cluster_pts2,0.01)   # with optional inputs
assert isinstance(topology_ip2, bool), "Topology.IsPlanar. Should be bool"
assert topology_ip2 == False, "Topology.IsPlanar. Should be False"

# Case 36 - IsSame
print("Case 36")
# test 1
topology_is = Topology.IsSame(e00,v00)    # vertice and edge
assert isinstance(topology_is, bool), "Topology.IsSame. Should be bool"
assert topology_is == False, "Topology.IsSame. Should be False"
# test 2
topology_is2 = Topology.IsSame(v00,v00)    # vertice and vertice
assert isinstance(topology_is2, bool), "Topology.IsSame. Should be bool"
assert topology_is2 == True, "Topology.IsSame. Should be True"

# Case 37 - MergeAll
print("Case 37")
# test 1
list_ve = [v00,v01,e00,e01]
"""AttributeError: 'NoneType' object has no attribute 'Union'"""
topology_ma = Topology.MergeAll(list_ve) #wrong

# Case 38 - OCCTShape
print("Case 38")
# test 1
topology_os = Topology.OCCTShape(cell_cy)
assert isinstance(topology_os, topologic.TopoDS_Shape), "Topology.OCCTShape. Should be topologic.TopoDS_Shape"
# test 2
topology_os2 = Topology.OCCTShape(e02)
assert isinstance(topology_os2, topologic.TopoDS_Shape), "Topology.OCCTShape. Should be topologic.TopoDS_Shape"

# Case 39 - Orient
print("Case 39")
# test 1
cell_cy = Cell.Cylinder()
topology_o = Topology.Orient(cell_cy, origin=None, dirA=[0, 0, 1], dirB=[0, 1, 0], tolerance=0.001)
assert isinstance(topology_o, topologic.Topology), "Topology.Orient. Should be topologic.Topology"
# plot geometry
data_cell_cy = Plotly.DataByTopology(cell_cy)
figure_cell_cy = Plotly.FigureByData(data_cell_cy)  
#Plotly.Show(figure_cell_cy)                         # original geometry
data_cell_cy2 = Plotly.DataByTopology(topology_o)
figure_cell_cy2 = Plotly.FigureByData(data_cell_cy2)
#Plotly.Show(figure_cell_cy2)                        # oriented geometry

# Case 40 - Place
print("Case 40")
# test 1
topology_pl = Topology.Place(cell_cy,v00,v02)
topology_pl_ctr = Topology.Centroid(topology_pl)            # get the centroid
vd = Vertex.Distance(topology_pl_ctr,v02)                    # measure distance of centroid and moved origin
assert isinstance(topology_pl, topologic.Topology), "Topology.Place. Should be topologic.Topology"
assert vd == 0.0, "Vertex.Distance. Should be 0.0"   

# Case 41 - InternalVertex
print("Case 41")
# test 1
topology_iv = Topology.InternalVertex(cell_cy, 0.01)
assert isinstance(topology_iv, topologic.Vertex), "Topology.InternalVertex. Should be topologic.Vertex"
cell_cy3 = Cell.Cylinder(topology_iv,radius = 0.25)
# plot geometry
cluster_bs = Cluster.ByTopologies([cell_cy,cell_cy3])
data_cluster_bs = Plotly.DataByTopology(cluster_bs)
figure_cluster_bs = Plotly.FigureByData(data_cluster_bs)
#Plotly.Show(figure_cluster_bs)                        # to check the returned vertex is inside the Cell.Cylinder

# Case 42 - RemoveCollinearEdges
print("Case 42")
# test 1
e4 = Edge.ByStartVertexEndVertex(v00,v01)
e5 = Edge.ByStartVertexEndVertex(v01,v05)
cluster_e = Cluster.ByTopologies([e4,e5])
topology_rce = Topology.RemoveCollinearEdges(cluster_e)
# plot geometry
data_top_rce = Plotly.DataByTopology(topology_rce)
figure_top_rce = Plotly.FigureByData(data_top_rce)
#Plotly.Show(figure_top_rce) 

# Case 43 - RemoveContent
print("Case 43")
# test 1
topology_rc = Topology.RemoveContent(cell_cy, [e00,e01])
assert isinstance(topology_rc, topologic.Topology), "Topology.RemoveContent. Should be topologic.Topology"
# plot geometry
data_topology_rc = Plotly.DataByTopology(topology_rc)
figure_topology_rc = Plotly.FigureByData(data_topology_rc)  
#Plotly.Show(figure_topology_rc)                         # original geometry

# Case 44 - RemoveCoplanarFaces
print("Case 44")
# test 1
shell_r = Shell.Rectangle(uSides=3, vSides=3)
shell_faces = Shell.Faces(shell_r)
topology_rcf = Topology.RemoveCoplanarFaces(shell_r)
assert len(shell_faces) == 9, "Shell.Faces. List length should be 9"
assert isinstance(topology_rcf, topologic.Face), "Topology.RemoveCoplanarFaces. Should be topologic.Face"
# test 2 
cell_box5 = Cell.Box(uSides= 5, vSides = 5)
topology_rcf = Topology.RemoveCoplanarFaces(cell_box5)
cell_f = Cell.Faces(cell_box5)
cell_f2 = Cell.Faces(topology_rcf)
assert len(cell_f) == 70, "Cell.Faces. List length should be 70"
assert isinstance(topology_rcf, topologic.Topology), "Topology.RemoveCoplanarFaces. Should be topologic.Topology"
assert len(cell_f2) == 6, "Topology.RemoveCoplanarFaces. List length should be 6"

# Case 45 - Rotate
print("Case 45")
# test 1
topology_rot = Topology.Rotate(cell_cy, x = 1, y = 0, z = 1, degree = 45)
assert isinstance(topology_rot, topologic.Topology), "Topology.Rotate. Should be topologic.Topology"
# plot geometry
#data_topology_rot = Plotly.DataByTopology(topology_rot)
#figure_topology_rot = Plotly.FigureByData(data_topology_rot) 
#Plotly.Show(figure_topology_rot)                         # rotated geometry

# Case 46 - Scale
print("Case 46")
# test 1
topology_scale = Topology.Scale(cell_cy)                                      # without optional inputs
assert isinstance(topology_scale, topologic.Topology), "Topology.Scale. Should be topologic.Topology"
# test 2
topology_scale02 = Topology.Scale(cell_cy,origin = v00, x = 5, y = 5, z = 1)    # with optional inputs
assert isinstance(topology_scale02, topologic.Topology), "Topology.Scale. Should be topologic.Topology"
# plot geometry
#data_topology_scale = Plotly.DataByTopology(topology_scale)
#figure_topology_scale = Plotly.FigureByData(data_topology_scale) 
#Plotly.Show(figure_topology_scale)                                            # scaled geometry
#data_topology_scale02 = Plotly.DataByTopology(topology_scale02)
#figure_topology_scale02 = Plotly.FigureByData(data_topology_scale02)             # scaled geometry
#Plotly.Show(figure_topology_scale02)                                           # scaled geometry

# Case 47 - SelectSubTopology
print("Case 47")
# test 1
"""NameError: name 'topologyType' is not defined. Did you mean: 'subTopologyType'"""
topology_sst = Topology.SelectSubTopology(cluster_bs, selector = v00, subTopologyType = "vertex")
assert isinstance(topology_sst, topologic.Vertex), "Topology.SelectSubTopology. Should be topologic.Vertex"

# Case 48 - SelfMerge
print("Case 48")
# test 1
clus_edges = Cluster.ByTopologies([e4,e5])
topology_sm = Topology.SelfMerge(clus_edges)
assert isinstance(topology_sm, topologic.Topology), "Topology.SelfMerge. Should be topologic.Topology"
assert isinstance(topology_sm, topologic.Wire), "Topology.SelfMerge. Should be topologic.Wire"

# Case 49 - SetDictionary
print("Case 49")
# test 1
k = ["a","b","c"]
val = [1,2,3]
top_dict = Dictionary.ByKeysValues(k,val)
topology_sd = Topology.SetDictionary(cell_cy,top_dict)
assert isinstance(topology_sd, topologic.Topology), "Topology.SetDictionary. Should be topologic.Topology"
#print(topology_sd)

# Case 50 - SharedTopologies
print("Case 50")
# test 1
"""Returns Empty Dictionary"""
topology_st1 = Topology.SharedTopologies(e00, e01)
#print(topology_st1)
#assert topology_st1 == True, "Topology.SharedTopologies. Should be True"
# test 2
"""Returns Empty Dictionary""" 
cell_box = Cell.Box()
vBox2 = Vertex.ByCoordinates(0,1,0)
cell_box2 = Cell.Box(vBox2)

topology_st = Topology.SharedTopologies(cell_box, cell_box2,)
# plot geometry
cc = CellComplex.ByCells([cell_box,cell_box2])
data_cc = Plotly.DataByTopology(cc)
figure_cc = Plotly.FigureByData(data_cc) 
#Plotly.Show(figure_cc)                         # visualization
#print(topology_st) #empty

# Case 51 - SortBySelectors
print("Case 51")
# test 1
cell_cy5 = Cell.Cylinder()
v_cy6 = Vertex.ByCoordinates(0,1,0)
cell_cy6 = Cell.Cylinder(origin = v_cy6)
topology_sbs = Topology.SortBySelectors([cell_cy5,cell_cy6], [v_cy6,v00], exclusive = False, tolerance = 0.001)
assert isinstance(topology_sbs, dict), "Topology.SortBySelectors. Should be dictionary"
#print(topology_sbs)                              # check the sorted and unsorted list

# Case 52 - Spin
print("Case 52")
"""Takes really long time to run"""
# test 1
topology_spin = Topology.Spin(cell_cy, origin=v00, triangulate=False,
                                direction = [0,1,0], degree=90, sides=2, tolerance=0.001)
assert isinstance(topology_spin, topologic.Topology), "Topology.Spin. Should be topologic.Topology"
# plot geometry
#data_top_spin = Plotly.DataByTopology(topology_spin)
#figure_top_spin = Plotly.FigureByData(data_top_spin) 
#Plotly.Show(figure_top_spin, renderer="browser")                         # visualization

# Case 53 - BREPString
print("Case 53")
# test 1
topology_str = Topology.BREPString(cell_cy)
topology_str2 = Topology.BREPString(cell_cy3)
assert isinstance(topology_str, str), "Topology.BREPString. Should be string"
assert isinstance(topology_str2, str), "Topology.BREPString. Should be string"
#print(topology_str)
#print(topology_str2)

# Case 54 - SubTopologies
print("Case 54")
# test 1
topology_sub = Topology.SubTopologies(cell_cy, subTopologyType='edge')
assert isinstance(topology_sub, list), "Topology.SubTopologies. Should be list"
assert len(topology_sub) == 48, "Topology.SubTopologies. List length should be 12"

# Case 55 - SuperTopologies
print("Case 55")
# test 1
v_cell = Cell.Vertices(cell_cy3)
topology_sup = Topology.SuperTopologies(v_cell[1], cell_cy3, "face")
#ssert isinstance(topology_sup, list), "Topology.SuperTopologies. Should be list"
#Topology.Show(cell_cy3, renderer="browser")
assert len(topology_sup) == 3, "Topology.SuperTopologies. List length should be 3"

# Case 56 - SymmetricDifference
print("Case 56")
# test 1
cell_cy4 = Cell.Cylinder(origin = Vertex.ByCoordinates(0,0.125,0),radius = 0.25)
topology_sym = Topology.SymmetricDifference(cell_cy, cell_cy4, tranDict = False)
assert isinstance(topology_sym, topologic.Topology), "Topology.SymmetricDifference. Should be topologic.Topology"
# plot geometry
data_top_sym = Plotly.DataByTopology(topology_sym)
figure_top_sym = Plotly.FigureByData(data_top_sym)
#Plotly.Show(figure_top_sym)                               # visualization

# Case 57 - TransferDictionaries
print("Case 57 - Skipped")
# test 1
#list_cy = [cell_cy,cell_cy3]
#list_vs = [v00,v01]
#topology_td = Topology.TransferDictionaries(list_cy,list_vs,0.001)
#assert isinstance(topology_td, dict), "Topology.TransferDictionaries. Should be Dictionary"

# Case 59 - TransferDictionariesBySelectors
print("Case 58 - Skipped")
# test 1
#cluster_c = Cluster.ByTopologies([cell_cy,cell_cy])
#topology_tdbs = Topology.TransferDictionariesBySelectors(cluster_c, selectors = [v00,v01], tranVertices = False,
                                                        #tranEdges = False, tranFaces = False, tranCells = False, tolerance = 0.001 )

# Case 59 -  Transform
print("Case 59")
# test 1
mat_rot = Matrix.ByRotation(rx=0, ry=45, rz=0, order='xyz')     # Create rotation matrix
topology_transform = Topology.Transform(cell_cy3, mat_rot)
assert isinstance(topology_transform, topologic.Topology), "Topology.Transform. Should be topologic.Topology" 
# plot geometry
data_top_transform = Plotly.DataByTopology(topology_transform)
figure_top_transform = Plotly.FigureByData(data_top_transform)
#Plotly.Show(figure_top_transform)                               # visualization

# Case 60 -  Translate
print("Case 60")
# test 1
topology_trans = Topology.Translate(cell_cy3, x=0, y=0, z=1)
center_tcy = Topology.Centroid(topology_trans)
z_tcy = Vertex.Z(center_tcy)                          # Get the z value of the translated geometry's centroid
assert isinstance(topology_trans, topologic.Topology), "Topology.Translate. Should be topologic.Topology"
assert z_tcy == 1.0, "Vertex.Z. should be 1.0"

# Case 61 -  Triangulate
print("Case 61")
# test 1
topology_tr = Topology.Triangulate(cell_cy, tolerance = 0.001)
assert isinstance(topology_tr, topologic.Topology), "Topology.Triangulate. Should be topologic.Topology"
# plot geometry
data_top_tr = Plotly.DataByTopology(topology_tr)
figure_top_tr = Plotly.FigureByData(data_top_tr) 
#Plotly.Show(figure_top_tr)                         # visualization

# Case 62 -  Type
print("Case 62")
# test 1
topology_type = Topology.Type(cell_cy)
assert isinstance(topology_type, int), "Topology.TypeID. Should be integer"
assert topology_type == 32, "Topology.SubTopologies. Should be 32"
# test 2
topology_type02 = Topology.Type(v00)
assert isinstance(topology_type02, int), "Topology.TypeID. Should be integer"
assert topology_type02 == 1, "Topology.SubTopologies. Should be 1"

# Case 63 -  TypeAsString
print("Case 63")
# test 1
topology_tas = Topology.TypeAsString(cell_cy)
assert isinstance(topology_tas, str), "Topology.TypeAsString. Should be string"
assert topology_tas == "Cell", "Topology.TypeAsString. Should be Cell"
# test 2
topology_tas2 = Topology.TypeAsString(v00)
assert isinstance(topology_tas2, str), "Topology.TypeAsString. Should be string"
assert topology_tas2 == "Vertex", "Topology.TypeAsString. Should be Vertex"

# Case 64 -  TypeID
print("Case 64")
# test 1
topology_id = Topology.TypeID("vertex")
assert isinstance(topology_id, int), "Topology.TypeID. Should be integer"
assert topology_id == 1, "Topology.TypeID. Should be 1"
# test 2
topology_id2 = Topology.TypeID("edge")
assert isinstance(topology_id2, int), "Topology.TypeID. Should be integer"
assert topology_id2 == 2, "Topology.TypeID. Should be 2"
# test 3
topology_id3 = Topology.TypeID("cell")
assert isinstance(topology_id3, int), "Topology.TypeID. Should be integer"
assert topology_id3 == 32, "Topology.TypeID. Should be 32"
print("End")