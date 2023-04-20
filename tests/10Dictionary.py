# Dictionary Classes unit test

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
from topologicpy.Dictionary import Dictionary

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
shell_open = Shell.ByFaces([face0,face2])       # create shell

# Case 1 - ByKeysValues

# test 1
k = ["a", "b", "c"]
val = [1,2,3]
dict_kv = Dictionary.ByKeysValues(k,val)
assert isinstance(dict_kv, topologic.Dictionary), "Dictionary.ByKeysValues. Should be topologic.Dictionary"
assert len(dict_kv.Keys()) == 3, "Dictionary.ByKeysValues. List length should be 3"
# test 2
dict_kv2 = Dictionary.ByKeysValues(["d","e","f"],[4,5,6])
assert isinstance(dict_kv2, topologic.Dictionary), "Dictionary.ByKeysValues. Should be topologic.Dictionary"
assert len(dict_kv2.Keys()) == 3, "Dictionary.ByKeysValues. List length should be 3"

# Case 2 - ByMergedDictionaries
# test 1
dict_bmd = Dictionary.ByMergedDictionaries([dict_kv,dict_kv2])
assert isinstance(dict_bmd, topologic.Dictionary), "Dictionary.ByKeysValues. Should be topologic.Dictionary"
assert len(dict_bmd.Keys()) == 6, "Dictionary.ByMergedDictionaries. List length should be 6"
assert len(dict_bmd.Values()) == 6, "Dictionary.ByMergedDictionaries. List length should be 6"

# Case 3 - ByPythonDictionary
# test 1
dict_py1 = {"g":7,"h":8,"i":9}
dict_bpd = Dictionary.ByPythonDictionary(dict_py1)
assert isinstance(dict_bpd, topologic.Dictionary), "Dictionary.ByPythonDictionary. Should be topologic.Dictionary"
assert len(dict_bpd.Keys()) == 3, "Dictionary.ByPythonDictionary. List length should be 3"
# test 2
dict_py2 = {"j":10,"k":11,"l":12}
dict_bpd2 = Dictionary.ByPythonDictionary(dict_py2)
assert isinstance(dict_bpd2, topologic.Dictionary), "Dictionary.ByPythonDictionary. Should be topologic.Dictionary"
assert len(dict_bpd2.Keys()) == 3, "Dictionary.ByPythonDictionary. List length should be 3"

# Case 4 - Keys

# test 1
keys_dict1 = Dictionary.Keys(dict_bpd)
assert isinstance(keys_dict1, list), "Dictionary.Keys. Should be list"
assert len(keys_dict1) == 3, "Dictionary.Keys. List length should be 3"

# test 2
keys_dict2 = Dictionary.Keys(dict_bmd)
assert isinstance(keys_dict2, list), "Dictionary.Keys. Should be list"
assert len(keys_dict2) == 6, "Dictionary.Keys. List length should be 6"

# Case 5 - ListAttributeValues
#val_dict1 = Dictionary.ListAttributeValues([])
#assert isinstance(val_dict1, list), " Dictionary.ListAttributeValues. Should be list"
#assert len(val_dict1) == 3, "Dictionary.ListAttributeValues. List length should be 3"

# Case 6 - PythonDictionary

# test 1
dict_pd1 = Dictionary.PythonDictionary(dict_bpd)
assert isinstance(dict_pd1, dict), "Dictionary.PythonDictionary. Should be Dict"
assert len(dict_pd1.keys()) == 3, "Dictionary.PythonDictionary. List length should be 3"
# test 2
dict_bpd2 = Dictionary.PythonDictionary(dict_bpd2)
assert isinstance(dict_bpd2, dict), "Dictionary.PythonDictionary. Should be Dict"
assert len(dict_bpd2.keys()) == 3, "Dictionary.PythonDictionary. List length should be 3"

# Case 7 - SetValueAtKey

# test 1
dict_svk = Dictionary.SetValueAtKey(dict_bpd2, key = "m", value= 13)
assert isinstance(dict_svk, dict), "Dictionary.SetValueAtKey. Should be Dictionary"
assert len(dict_svk.keys()) == 4, "Dictionary.SetValueAtKey. List length should be 4"
# test 2
dict_svk2 = Dictionary.SetValueAtKey(dict_svk, key = "n", value= 14)
assert isinstance(dict_svk2, dict), "Dictionary.SetValueAtKey. Should be Dictionary"
assert len(dict_svk2.keys()) == 5, "Dictionary.SetValueAtKey. List length should be 5"

# Case 8 - ValueAtKey

# test 1
vak_dict = Dictionary.ValueAtKey(dict_svk2, key= "n")
assert isinstance(vak_dict, int), "Dictionary.SetValueAtKey. Should be int,float,str,list"
assert vak_dict == 14, "Dictionary.SetValueAtKey. Should be 14"
# test 2
vak_dict2 = Dictionary.ValueAtKey(dict_svk, key= "m")
assert isinstance(vak_dict, int), "Dictionary.SetValueAtKey. Should be int,float,str,list"
assert vak_dict2 == 13, "Dictionary.SetValueAtKey. Should be 13"

# Case 9 - Values

# test 1
vals_dict = Dictionary.Values(dict_svk2)
assert isinstance(vals_dict, list), "Dictionary.Values. Should be list"
assert len(vals_dict) == 5, "Dictionary.Values. List length should be 5"
# test 2
vals_dict2 = Dictionary.Values(dict_svk)
assert isinstance(vals_dict2, list), "Dictionary.Values. Should be list"
assert len(vals_dict2) == 5, "Dictionary.Values. List length should be 5"