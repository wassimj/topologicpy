# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from topologicpy.Core import Core

class Dictionary():

    @staticmethod
    def _CoreClass(name: str):
        """
        Returns a backend class or callable namespace exposed through Core.
        """
        return Core.Namespace(name)

    @staticmethod
    def _CoreCall(obj, methodName: str, *args, **kwargs):
        """
        Calls a backend-native instance method through Core when available.
        """
        if hasattr(Core, "InstanceCall"):
            return Core.InstanceCall(obj, methodName, *args, **kwargs)
        method = getattr(obj, methodName)
        return method(*args, **kwargs)

    @staticmethod
    def _IsDictionary(dictionary) -> bool:
        """
        Returns True if the input behaves like a supported dictionary.

        Supported inputs are:
        - native Python dictionaries;
        - the active Core/topologic dictionary class;
        - backend dictionary objects exposing Keys() and ValueAtKey(key);
        - backend dictionary objects exposing PythonDictionary().
        """
        if dictionary is None:
            return False

        if isinstance(dictionary, dict):
            return True

        try:
            DictionaryClass = Dictionary._CoreClass("Dictionary")
            if isinstance(dictionary, DictionaryClass):
                return True
        except Exception:
            pass

        try:
            from topologicpy.Topology import Topology
            if Topology.IsInstance(dictionary, "Dictionary"):
                return True
        except Exception:
            pass

        try:
            if callable(getattr(dictionary, "PythonDictionary", None)):
                py_dict = dictionary.PythonDictionary()
                return isinstance(py_dict, dict)
        except Exception:
            pass

        return callable(getattr(dictionary, "Keys", None)) and callable(getattr(dictionary, "ValueAtKey", None))

    @staticmethod
    def _Keys(dictionary):
        """
        Returns dictionary keys without printing diagnostics.
        """
        if isinstance(dictionary, dict):
            return list(dictionary.keys())

        if dictionary is None:
            return None

        try:
            if callable(getattr(dictionary, "PythonDictionary", None)):
                py_dict = dictionary.PythonDictionary()
                if isinstance(py_dict, dict):
                    return list(py_dict.keys())
        except Exception:
            pass

        try:
            keys = Dictionary._CoreCall(dictionary, "Keys")
            if keys is None:
                return []
            return list(keys)
        except Exception:
            pass

        try:
            keys = dictionary.Keys()
            if keys is None:
                return []
            return list(keys)
        except Exception:
            return None

    @staticmethod
    def _RawValueAtKey(dictionary, key):
        """
        Returns a raw dictionary value or Core attribute without converting it.
        """
        if isinstance(dictionary, dict):
            return dictionary.get(key, None)

        try:
            if callable(getattr(dictionary, "PythonDictionary", None)):
                py_dict = dictionary.PythonDictionary()
                if isinstance(py_dict, dict):
                    return py_dict.get(key, None)
        except Exception:
            pass

        try:
            return Dictionary._CoreCall(dictionary, "ValueAtKey", key)
        except Exception:
            pass

        try:
            return dictionary.ValueAtKey(key)
        except Exception:
            return None

    @staticmethod
    def _ConvertAttribute(attr):
        """
        Converts a Topologic/Core attribute, Python value, or backend dictionary value into a Python value.
        """
        from topologicpy.Topology import Topology
        import json

        def is_json_string(input_string):
            if not isinstance(input_string, str):
                return False
            try:
                json.loads(input_string)
                return True
            except Exception:
                return False

        def topology_from_json_string(json_string):
            if not is_json_string(json_string):
                return None
            try:
                topologies = Topology.ByJSONString(json_string)
            except Exception:
                return None
            if isinstance(topologies, list):
                if len(topologies) == 0:
                    return None
                if len(topologies) == 1:
                    return topologies[0]
                return topologies
            if Topology.IsInstance(topologies, "Topology"):
                return topologies
            return None

        if attr is None:
            return None

        IntAttribute = Dictionary._CoreClass("IntAttribute")
        DoubleAttribute = Dictionary._CoreClass("DoubleAttribute")
        StringAttribute = Dictionary._CoreClass("StringAttribute")
        ListAttribute = Dictionary._CoreClass("ListAttribute")

        if isinstance(attr, IntAttribute):
            return Dictionary._CoreCall(attr, "IntValue")
        if isinstance(attr, DoubleAttribute):
            return Dictionary._CoreCall(attr, "DoubleValue")
        if isinstance(attr, StringAttribute):
            temp_value = Dictionary._CoreCall(attr, "StringValue")
            if temp_value == "__NONE__":
                return None
            topologies = topology_from_json_string(temp_value)
            if topologies is not None:
                return topologies
            if is_json_string(temp_value):
                return json.loads(temp_value)
            return temp_value
        if isinstance(attr, ListAttribute):
            return Dictionary.ListAttributeValues(attr)

        if isinstance(attr, bool):
            return attr
        if isinstance(attr, (float, int)):
            return attr
        if isinstance(attr, str):
            if attr == "__NONE__":
                return None
            topologies = topology_from_json_string(attr)
            if topologies is not None:
                return topologies
            if is_json_string(attr):
                return json.loads(attr)
            return attr
        if isinstance(attr, tuple):
            return [Dictionary._ConvertAttribute(x) for x in list(attr)]
        if isinstance(attr, list):
            return [Dictionary._ConvertAttribute(x) for x in attr]
        if isinstance(attr, dict):
            return attr
        if Dictionary._IsDictionary(attr):
            return Dictionary.PythonDictionary(attr, silent=True)
        return attr

    @staticmethod
    def _ConvertValue(value):
        """
        Converts a Python value to the proper Core/topologic attribute.
        """
        from topologicpy.Topology import Topology
        import json

        def dict_to_json(py_dict):
            return json.dumps(py_dict, indent=2)

        IntAttribute = Dictionary._CoreClass("IntAttribute")
        DoubleAttribute = Dictionary._CoreClass("DoubleAttribute")
        StringAttribute = Dictionary._CoreClass("StringAttribute")
        ListAttribute = Dictionary._CoreClass("ListAttribute")

        if value is None:
            return StringAttribute("__NONE__")
        if isinstance(value, bool):
            return IntAttribute(1 if value else 0)
        if isinstance(value, int):
            return IntAttribute(value)
        if isinstance(value, float):
            return DoubleAttribute(value)
        if Topology.IsInstance(value, "Topology"):
            return StringAttribute(Topology.JSONString(value))
        if Dictionary._IsDictionary(value):
            py_dict = Dictionary.PythonDictionary(value, silent=True)
            return StringAttribute(dict_to_json(py_dict if isinstance(py_dict, dict) else {}))
        if isinstance(value, str):
            return StringAttribute(value)
        if isinstance(value, tuple):
            return ListAttribute([Dictionary._ConvertValue(v) for v in list(value)])
        if isinstance(value, list):
            return ListAttribute([Dictionary._ConvertValue(v) for v in value])
        return StringAttribute("__NONE__")

    @staticmethod
    def AdjacencyDictionary(topology, subTopologyType: str = None, labelKey: str = None,  weightKey: str = None, includeWeights: bool = False, mantissa: int = 6, silent: bool = False):
        """
        Returns the adjacency dictionary of the input Shell.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        subTopologyType : str , optional
            The type of subTopology on which to base the adjacency dictionary.
        labelKey : str , optional
            The returned subTopologies are labelled according to the dictionary values stored under this key.
            If the labelKey does not exist, it will be created and the subTopologies are labelled numerically and stored in the subTopologies' dictionary under this key. Default is None.
        weightKey : str , optional
            If set, the sharedTopologies' dictionaries will be searched for this key to set their weight. If the key is set to "Area" or "Length" (case insensitive), the area of shared faces or the length of the shared edges will be used as its weight. If set to None, a weight of 1 will be used. Default is None.
        includeWeights : bool , optional
            If set to True, edge weights are included. Otherwise, they are not. Default is False.
        mantissa : int , optional
                The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            The adjacency dictionary.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(topology, "Topology") and not Topology.IsInstance(topology, "Graph"):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input topology input parameter is not a valid topology. Returning None.")
            return None
        # Special Case for Graphs

        if Topology.IsInstance(topology, "Graph"):
            return Graph.AdjacencyDictionary(topology, vertexLabelKey=labelKey, edgeKey=weightKey, includeWeights=includeWeights, mantissa=mantissa)
        if labelKey == None:
            labelKey = "__label__"
        if not isinstance(labelKey, str):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input labelKey is not a valid string. Returning None.")
            return None
        if Topology.IsInstance(topology, "cellcomplex"):
            if subTopologyType == None:
                subTopologyType = "cell"
            all_subtopologies = Topology.SubTopologies(topology, subTopologyType=subTopologyType, silent=silent)
        elif Topology.IsInstance(topology, "cell") or Topology.IsInstance(topology, "shell"):
            if subTopologyType == None:
                subTopologyType = "face"
            all_subtopologies = Topology.SubTopologies(topology, subTopologyType=subTopologyType, silent=silent)
        elif Topology.IsInstance(topology, "face") or Topology.IsInstance(topology, "wire"):
            if subTopologyType == None:
                subTopologyType = "edge"
            all_subtopologies = Topology.SubTopologies(topology, subTopologyType=subTopologyType, silent=silent)
        labels = []
        n = max(len(str(len(all_subtopologies))), 3)
        for i, subtopology in enumerate(all_subtopologies):
            d = Topology.Dictionary(subtopology)
            value = Dictionary.ValueAtKey(d, labelKey)
            if value == None:
                value = str(i+1).zfill(n)
            if d == None:
                d = Dictionary.ByKeyValue(labelKey, value)
            else:
                d = Dictionary.SetValueAtKey(d, labelKey, value, silent=silent)
            subtopology = Topology.SetDictionary(subtopology, d)
            labels.append(value)
        all_subtopologies = Helper.Sort(all_subtopologies, labels)
        labels.sort()
        order = len(all_subtopologies)
        adjDict = {}
        for i in range(order):
            subtopology = all_subtopologies[i]
            subt_label = labels[i]
            adjacent_topologies = Topology.AdjacentTopologies(subtopology, hostTopology=topology, topologyType=subTopologyType)
            temp_list = []
            for adj_topology in adjacent_topologies:
                adj_label = Dictionary.ValueAtKey(Topology.Dictionary(adj_topology), labelKey, silent=silent)
                adj_index = labels.index(adj_label)
                if includeWeights == True:
                    if weightKey == None:
                        weight = 1
                    elif "length" in weightKey.lower():
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        edges = shared_topologies.get("edges", [])
                        weight = sum([Edge.Length(edge, mantissa=mantissa) for edge in edges])
                    elif "area" in weightKey.lower():
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        faces = shared_topologies.get("faces", [])
                        weight = sum([Face.Area(face, mantissa=mantissa) for face in faces])
                    else:
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        vertices = shared_topologies.get("vertices", [])
                        edges = shared_topologies.get("edges", [])
                        wires = shared_topologies.get("wires", [])
                        faces = shared_topologies.get("faces", [])
                        everything = vertices+edges+wires+faces
                        weight = sum([Dictionary.ValueAtKey(Topology.Dictionary(x),weightKey, 0) for x in everything])
                        weight = round(weight, mantissa)
                    if not adj_index == None:
                        temp_list.append((adj_label, weight))
                else:
                    if not adj_index == None:
                        temp_list.append(adj_label)
            temp_list.sort()
            adjDict[subt_label] = temp_list
        if labelKey == "__label__": # This is label we added, so remove it
            for subtopology in all_subtopologies:
                d = Topology.Dictionary(subtopology)
                d = Dictionary.RemoveKey(d, labelKey)
                subtopology = Topology.SetDictionary(subtopology, d)
        return adjDict

    @staticmethod
    def BooleanDictionariesByKey(dictionariesA,
                                dictionariesB,
                                key: str,
                                operation: str = "union",
                                exclusive: bool = True,
                                silent: bool = False):
        """
        Booleans the keys/values of the dictionaries in the second list on the
        dictionaries in the first list based on a shared dictionary key/value and the boolean operation.
        """
        dictionariesA = [d for d in dictionariesA if Dictionary._IsDictionary(d)] if isinstance(dictionariesA, list) else []
        dictionariesB = [d for d in dictionariesB if Dictionary._IsDictionary(d)] if isinstance(dictionariesB, list) else []
        if len(dictionariesA) < 1:
            if not silent:
                print("Dictionary.BooleanDictionariesByKey - Error: The dictionariesA input parameter does not contain any valid dictionaries. Returning None.")
            return None
        if len(dictionariesB) < 1:
            if not silent:
                print("Dictionary.BooleanDictionariesByKey - Error: The dictionariesB input parameter does not contain any valid dictionaries. Returning None.")
            return None

        op = operation.lower()
        if op not in ["union", "merge", "difference", "intersection", "symmetricdifference", "symdif", "xor", "impose", "imprint"]:
            if not silent:
                print("Dictionary.BooleanDictionariesByKey - Error: Unrecognized boolean operation. Returning None.")
            return None

        def apply_op(dA, dB):
            if op in ["union", "merge"]:
                return Dictionary.Union(dA, dB, silent=silent)
            if op == "difference":
                return Dictionary.Difference(dA, dB, silent=silent)
            if op == "intersection":
                return Dictionary.Intersection(dA, dB, silent=silent)
            if op in ["symmetricdifference", "symdif", "xor"]:
                return Dictionary.SymmetricDifference(dA, dB, silent=silent)
            if op == "impose":
                return Dictionary.Impose(dA, dB, silent=silent)
            if op == "imprint":
                return Dictionary.Imprint(dA, dB, silent=silent)
            return dA

        dictionariesC = []
        if exclusive:
            lookup = {}
            for d in dictionariesB:
                lookup[Dictionary.ValueAtKey(d, key, None, silent=silent)] = d
            for dA in dictionariesA:
                vA = Dictionary.ValueAtKey(dA, key, None, silent=silent)
                dB = lookup.get(vA, None)
                dictionariesC.append(apply_op(dA, dB) if dB else Dictionary.ByKeysValues([], [], silent=silent))
        else:
            for dA in dictionariesA:
                vA = Dictionary.ValueAtKey(dA, key, None, silent=silent)
                for dB in dictionariesB:
                    if vA == Dictionary.ValueAtKey(dB, key, None, silent=silent):
                        dA = apply_op(dA, dB)
                dictionariesC.append(dA)
        return dictionariesC

    @staticmethod
    def ByKeyValue(key, value, silent: bool = False):
        """
        Creates a Dictionary from the input key and the input value.

        Parameters
        ----------
        key : str
            The string representing the key of the value in the dictionary.
        value : int, float, str, or list
            A value corresponding to the input key. A value can be an integer, a float, a string, or a list.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.

        """
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.ByKeyValue - Error: The input key is not a valid string. Returning None.")
            return None
        return Dictionary.ByKeysValues([key], [value], silent=silent)
    
    @staticmethod
    def ByKeysValues(keys, values, silent: bool = False):
        """
        Creates a Dictionary from the input list of keys and the input list of values.

        Parameters
        ----------
        keys : list
            A list of strings representing the keys of the dictionary.
        values : list
            A list of values corresponding to the list of keys. Values can be integers, floats, strings, or lists
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.

        """
        
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not isinstance(values, list):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input values parameter is not a valid list. Returning None.")
            return None
        if len(keys) != len(values):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input keys and values parameters are not of equal length. Returning None.")
            return None
        stl_keys = []
        stl_values = []
        for i in range(len(keys)):
            if isinstance(keys[i], str):
                stl_keys.append(keys[i])
            else:
                stl_keys.append(str(keys[i]))
            stl_values.append(Dictionary._ConvertValue(values[i]))
        return Core.Dictionary.ByKeysValues(stl_keys, stl_values)

    @staticmethod
    def ByMergedDictionaries(*dictionaries, silent: bool = False):
        """
        Creates a dictionary by merging the list of input dictionaries.

        Parameters
        ----------
        dictionaries : list or comma separated dictionaries
            The input list of dictionaries to be merged.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology

        if isinstance(dictionaries, tuple):
            dictionaries = Helper.Flatten(list(dictionaries))
        elif not isinstance(dictionaries, list):
            dictionaries = [dictionaries]

        valid_dictionaries = []

        for d in dictionaries:
            if Topology.IsInstance(d, "Dictionary"):
                valid_dictionaries.append(d)
            elif isinstance(d, dict):
                try:
                    valid_dictionaries.append(Dictionary.ByPythonDictionary(d))
                except Exception:
                    continue

        if len(valid_dictionaries) == 0:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: the input dictionaries parameter does not contain any valid dictionaries. Returning None.")
            return None

        if len(valid_dictionaries) == 1:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Warning: the input dictionaries parameter contains only one valid dictionary. Returning that dictionary.")
            return valid_dictionaries[0]

        first_dictionary = valid_dictionaries[0]

        try:
            sinkKeys = Dictionary.Keys(first_dictionary)
            sinkValues = Dictionary.Values(first_dictionary)
        except Exception:
            sinkKeys = []
            sinkValues = []

        if sinkKeys is None:
            sinkKeys = []
        if sinkValues is None:
            sinkValues = []

        key_to_index = {key: i for i, key in enumerate(sinkKeys)}

        for d in valid_dictionaries[1:]:
            if d is None:
                continue

            try:
                sourceKeys = Dictionary.Keys(d)
            except Exception:
                sourceKeys = []

            if not sourceKeys:
                continue

            for sourceKey in sourceKeys:
                try:
                    sourceValue = Dictionary.ValueAtKey(d, sourceKey, None)
                except Exception:
                    sourceValue = None

                if sourceValue is None:
                    continue

                index = key_to_index.get(sourceKey)

                if index is None:
                    key_to_index[sourceKey] = len(sinkKeys)
                    sinkKeys.append(sourceKey)
                    sinkValues.append(sourceValue)
                    continue

                sinkValue = sinkValues[index]

                if sinkValue is None or sinkValue == "":
                    sinkValues[index] = sourceValue
                elif isinstance(sinkValue, list):
                    if sourceValue not in sinkValue:
                        sinkValue.append(sourceValue)
                elif sourceValue != sinkValue:
                    sinkValues[index] = [sinkValue, sourceValue]

        if len(sinkKeys) > 0 and len(sinkValues) > 0:
            return Dictionary.ByKeysValues(sinkKeys, sinkValues)

        return None

    @staticmethod
    def ByMergedDictionaries_old(*dictionaries, silent: bool = False):
        """
        Creates a dictionary by merging the list of input dictionaries.
        """
        from topologicpy.Helper import Helper

        if isinstance(dictionaries, tuple):
            dictionaries = Helper.Flatten(list(dictionaries))
        elif not isinstance(dictionaries, list):
            dictionaries = [dictionaries]

        dictionaries = [d for d in dictionaries if Dictionary._IsDictionary(d)]
        if len(dictionaries) == 0:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: the input dictionaries parameter does not contain any valid dictionaries. Returning None.")
            return None
        if len(dictionaries) == 1:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Warning: the input dictionaries parameter contains only one dictionary. Returning input dictionary.")
            return dictionaries[0]

        sinkKeys = []
        sinkValues = []
        for d in dictionaries:
            sourceKeys = Dictionary.Keys(d, silent=silent) or []
            for sourceKey in sourceKeys:
                sourceValue = Dictionary.ValueAtKey(d, sourceKey, silent=silent)
                if sourceKey not in sinkKeys:
                    sinkKeys.append(sourceKey)
                    sinkValues.append(sourceValue)
                    continue
                index = sinkKeys.index(sourceKey)
                if sourceValue is None:
                    continue
                if sinkValues[index] is None or sinkValues[index] == "":
                    sinkValues[index] = sourceValue
                elif isinstance(sinkValues[index], list):
                    if sourceValue not in sinkValues[index]:
                        sinkValues[index].append(sourceValue)
                elif sourceValue != sinkValues[index]:
                    sinkValues[index] = [sinkValues[index], sourceValue]

        return Dictionary.ByKeysValues(sinkKeys, sinkValues, silent=silent)

    @staticmethod
    def ByObjectProperties(bObject, keys, importAll):
        """
        Parameters
        ----------
        bObject : TYPE
            DESCRIPTION.
        keys : TYPE
            DESCRIPTION.
        importAll : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # bObject, keys, importAll = item
        dictKeys = []
        dictValues = []

        if importAll:
            dictKeys.append("Name")
            dictValues.append(bObject.name)
            dictKeys.append("Color")
            dictValues.append([bObject.color[0], bObject.color[1], bObject.color[2], bObject.color[3]])
            dictKeys.append("Location")
            dictValues.append([bObject.location[0], bObject.location[1], bObject.location[2]])
            dictKeys.append("Scale")
            dictValues.append([bObject.scale[0], bObject.scale[1], bObject.scale[2]])
            dictKeys.append("Rotation")
            dictValues.append([bObject.rotation_euler[0], bObject.rotation_euler[1], bObject.rotation_euler[2]])
            dictKeys.append("Dimensions")
            dictValues.append([bObject.dimensions[0], bObject.dimensions[1], bObject.dimensions[2]])
            for k, v in bObject.items():
                if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    dictKeys.append(str(k))
                    dictValues.append(v)
        else:
            for k in keys:
                try:
                    v = bObject[k]
                    if v:
                        if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                            dictKeys.append(str(k))
                            dictValues.append(v)
                except:
                    if k.lower() == "name":
                        dictKeys.append("Name")
                        dictValues.append(bObject.name)
                    elif k.lower() == "color":
                        dictKeys.append("Color")
                        dictValues.append([bObject.color[0], bObject.color[1], bObject.color[2], bObject.color[3]])
                    elif k.lower() == "location":
                        dictKeys.append("Location")
                        dictValues.append([bObject.location[0], bObject.location[1], bObject.location[2]])
                    elif k.lower() == "scale":
                        dictKeys.append("Scale")
                        dictValues.append([bObject.scale[0], bObject.scale[1], bObject.scale[2]])
                    elif k.lower() == "rotation":
                        dictKeys.append("Rotation")
                        dictValues.append([bObject.rotation_euler[0], bObject.rotation_euler[1], bObject.rotation_euler[2]])
                    elif k.lower() == "dimensions":
                        dictKeys.append("Dimensions")
                        dictValues.append([bObject.dimensions[0], bObject.dimensions[1], bObject.dimensions[2]])
                    else:
                        raise Exception("Dictionary.ByObjectProperties: Key \""+k+"\" does not exist in the properties of object \""+bObject.name+"\".")

        return Dictionary.ByKeysValues(dictKeys, dictValues)

    @staticmethod
    def ByPythonDictionary(pythonDictionary, silent: bool = False):
        """
        Creates a dictionary equivalent to the input python dictionary.

        Parameters
        ----------
        pythonDictionary : dict
            The input python dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The dictionary equivalent to the input python dictionary.

        """
        if not isinstance(pythonDictionary, dict):
            if not silent:
                print("Dictionary.ByPythonDictionary - Error: The input dictionary parameter is not a valid python dictionary. Returning None.")
            return None
        keys = list(pythonDictionary.keys())
        values = []
        for key in keys:
            values.append(pythonDictionary[key])
        return Dictionary.ByKeysValues(keys, values)

    @staticmethod
    def Copy(dictionary, silent: bool = False):
        """
        Creates a copy of the input dictionary.
        """
        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.Copy - Error: The input dictionary parameter is not a valid dictionary. Returning None.")
            return None
        keys = Dictionary.Keys(dictionary, silent=silent)
        values = Dictionary.Values(dictionary, silent=silent)
        return Dictionary.ByKeysValues(keys, values, silent=silent)

    @staticmethod
    def Difference(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the difference of dictionaryA and dictionaryB (A \\ B), based on keys.
        """
        if dictionaryA is None:
            if not silent:
                print("Dictionary.Difference - Warning: The dictionaryA input parameter is None. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.Difference - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if dictionaryB is None:
            if not silent:
                print("Dictionary.Difference - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.Difference - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = Dictionary.Keys(dictionaryB, silent=silent) or []
        out_keys = [k for k in keysA if k not in keysB]
        out_vals = [Dictionary.ValueAtKey(dictionaryA, k, silent=silent) for k in out_keys]
        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

    @staticmethod
    def Filter(elements, dictionaries, searchType="any", key=None, value=None):
        """
        Filters the input list of dictionaries based on the input parameters.
        """
        filteredDictionaries = []
        otherDictionaries = []
        filteredElements = []
        otherElements = []
        filteredIndices = []
        otherIndices = []

        for i, aDictionary in enumerate(dictionaries):
            if not Dictionary._IsDictionary(aDictionary):
                continue
            if value == "" or key == "" or value is None or key is None:
                filteredDictionaries.append(aDictionary)
                filteredIndices.append(i)
                if i < len(elements):
                    filteredElements.append(elements[i])
                continue

            if isinstance(value, list):
                value = sorted(value)
            value_str = str(value).replace("*", ".+").lower()
            v = Dictionary.ValueAtKey(aDictionary, key)
            if v is None:
                otherDictionaries.append(aDictionary)
                otherIndices.append(i)
                if i < len(elements):
                    otherElements.append(elements[i])
                continue

            v_str = str(v).lower()
            st = searchType.lower()
            if st == "equal to":
                searchResult = value_str == v_str
            elif st == "contains":
                searchResult = value_str in v_str
            elif st == "starts with":
                searchResult = value_str == v_str[:len(value_str)]
            elif st == "ends with":
                searchResult = value_str == v_str[-len(value_str):]
            elif st == "not equal to":
                searchResult = value_str != v_str
            elif st == "does not contain":
                searchResult = value_str not in v_str
            else:
                searchResult = False

            if searchResult:
                filteredDictionaries.append(aDictionary)
                filteredIndices.append(i)
                if i < len(elements):
                    filteredElements.append(elements[i])
            else:
                otherDictionaries.append(aDictionary)
                otherIndices.append(i)
                if i < len(elements):
                    otherElements.append(elements[i])

        return {"filteredDictionaries": filteredDictionaries, "otherDictionaries": otherDictionaries, "filteredIndices": filteredIndices, "otherIndices": otherIndices, "filteredElements": filteredElements, "otherElements": otherElements}

    @staticmethod
    def Intersection(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the intersection of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None or dictionaryB is None:
            if not silent:
                print("Dictionary.Intersection - Warning: One or both of the input dictionaries is None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.Intersection - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.Intersection - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = Dictionary.Keys(dictionaryB, silent=silent) or []
        common_keys = [k for k in keysA if k in keysB]
        if len(common_keys) == 0:
            return Dictionary.ByKeysValues([], [], silent=silent)

        dA_1 = Dictionary.ByKeysValues(common_keys, [Dictionary.ValueAtKey(dictionaryA, k, silent=silent) for k in common_keys], silent=silent)
        dB_1 = Dictionary.ByKeysValues(common_keys, [Dictionary.ValueAtKey(dictionaryB, k, silent=silent) for k in common_keys], silent=silent)
        return Dictionary.ByMergedDictionaries(dA_1, dB_1, silent=silent)

    @staticmethod
    def Impose(dictionaryA, dictionaryB, silent: bool = False):
        """
        Imposes dictionaryB onto dictionaryA.
        """
        if dictionaryA is None:
            if not silent:
                print("Dictionary.Impose - Warning: The dictionaryA input parameter is None. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.Impose - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if dictionaryB is None:
            if not silent:
                print("Dictionary.Impose - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.Impose - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        py = Dictionary.PythonDictionary(dictionaryA, silent=True) or {}
        for k in Dictionary.Keys(dictionaryB, silent=silent) or []:
            py[k] = Dictionary.ValueAtKey(dictionaryB, k, silent=silent)
        return Dictionary.ByPythonDictionary(py, silent=silent)

    @staticmethod
    def Imprint(dictionaryA, dictionaryB, silent: bool = False):
        """
        Imprints dictionaryB onto dictionaryA.
        """
        if dictionaryA is None:
            if not silent:
                print("Dictionary.Imprint - Warning: The dictionaryA input parameter is None. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.Imprint - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if dictionaryB is None:
            if not silent:
                print("Dictionary.Imprint - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.Imprint - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = set(Dictionary.Keys(dictionaryB, silent=silent) or [])
        out_keys = list(keysA)
        out_vals = [Dictionary.ValueAtKey(dictionaryB if k in keysB else dictionaryA, k, silent=silent) for k in out_keys]
        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

    @staticmethod
    def Keys(dictionary, silent: bool = False):
        """
        Returns the keys of the input dictionary.
        """
        import inspect

        keys = Dictionary._Keys(dictionary)
        if keys is not None:
            return keys

        if not silent:
            print("Dictionary.Keys - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
        return None

    @staticmethod
    def ListAttributeValues(listAttribute):
        """
        Returns the list of values embedded in the input listAttribute.
        """
        if isinstance(listAttribute, (list, tuple)):
            return [Dictionary._ConvertAttribute(attr) for attr in listAttribute]
        listAttributes = Dictionary._CoreCall(listAttribute, "ListValue")
        return [Dictionary._ConvertAttribute(attr) for attr in listAttributes]

    @staticmethod
    def Merge(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the merge (union) of dictionaryA and dictionaryB.
        """
        return Dictionary.Union(dictionaryA, dictionaryB, silent=silent)

    @staticmethod
    def OneHotEncode(d, keys, categoriesByKey, silent=False):
        """
        One-hot encodes multiple categorical dictionary values in one pass. See https://en.wikipedia.org/wiki/One-hot
        """
        if not Dictionary._IsDictionary(d):
            if not silent:
                print("Dictionary.OneHotEncode - Error: Input is not a valid Dictionary.")
            return None
        if keys is None:
            if not silent:
                print("Dictionary.OneHotEncode - Error: keys is None.")
            return None
        keys = list(keys)
        if len(keys) == 0:
            return d
        if not isinstance(categoriesByKey, dict):
            if not silent:
                print("Dictionary.OneHotEncode - Error: categoriesByKey must be a dict mapping key -> categories.")
            return None

        orig_keys = Dictionary.Keys(d, silent=silent) or []
        orig_vals = [Dictionary.ValueAtKey(d, k, silent=silent) for k in orig_keys]
        orig_index = {k: i for i, k in enumerate(orig_keys)}
        values_by_key = {k: orig_vals[orig_index[k]] if k in orig_index else None for k in keys}

        remove_set = set(keys)
        out_keys = []
        out_vals = []
        for k, v in zip(orig_keys, orig_vals):
            if k not in remove_set:
                out_keys.append(k)
                out_vals.append(v)

        for k in keys:
            cats = categoriesByKey.get(k, None)
            if not isinstance(cats, (list, tuple)) or len(cats) == 0:
                if not silent:
                    print(f"Dictionary.OneHotEncode - Warning: No categories provided for key '{k}'. Skipping.")
                continue
            value = values_by_key.get(k, None)
            for i, cat in enumerate(cats):
                out_keys.append(f"{k}_{i}")
                out_vals.append(1 if value == cat else 0)

        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

    @staticmethod
    def PythonDictionary(dictionary, silent: bool = False):
        """
        Returns the input dictionary as a python dictionary.
        """
        if isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.PythonDictionary - Warning: The input dictionary parameter is already a python dictionary. Returning that dictionary.")
            return dictionary

        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.PythonDictionary - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            return None

        try:
            if callable(getattr(dictionary, "PythonDictionary", None)):
                py_dict = dictionary.PythonDictionary()
                if isinstance(py_dict, dict):
                    return dict(py_dict)
        except Exception:
            pass

        keys = Dictionary.Keys(dictionary, silent=silent)
        if keys is None:
            return None

        pythonDict = {}
        for key in keys:
            raw_value = Dictionary._RawValueAtKey(dictionary, key)
            pythonDict[key] = Dictionary._ConvertAttribute(raw_value)
        return pythonDict

    @staticmethod
    def RemoveKey(dictionary, key, silent: bool = False):
        """
        Removes the key and its associated value from the input dictionary.
        """
        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.RemoveKey - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.RemoveKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        py = Dictionary.PythonDictionary(dictionary, silent=True) or {}
        remove_key = None
        for k in py.keys():
            if isinstance(k, str) and k.lower() == key.lower():
                remove_key = k
                break
        if remove_key is not None:
            py.pop(remove_key, None)

        if isinstance(dictionary, dict):
            return py
        return Dictionary.ByPythonDictionary(py, silent=silent)

    @staticmethod
    def SetValueAtKey(dictionary, key, value, silent: bool = False):
        """
        Creates or updates a key/value pair in the input dictionary.
        """
        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        if isinstance(dictionary, dict):
            dictionary[key] = value
            return dictionary

        py = Dictionary.PythonDictionary(dictionary, silent=True) or {}
        py[key] = value
        return Dictionary.ByPythonDictionary(py, silent=silent)

    @staticmethod
    def SetValuesAtKeys(dictionary, keys, values, silent: bool = False):
        """
        Creates a key/value pair in the input dictionary.

        Parameters
        ----------
        keys : list
            A list of strings representing the keys of the dictionary.
        values : list
            A list of values corresponding to the list of keys. Values can be integers, floats, strings, or lists
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.

        """
        
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not isinstance(values, list):
            if not silent:
                print("Dictionary.SetValuesAtkeys - Error: The input values parameter is not a valid list. Returning None.")
            return None
        if len(keys) != len(values):
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input keys and values parameters are not of equal length. Returning None.")
            return None
        
        for i, key in enumerate(keys):
            dictionary = Dictionary.SetValueAtKey(dictionary, key, values[i], silent=silent)
        return dictionary
    
    @staticmethod
    def SymDif(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.SymDif - Warning: Both of the input dictionaries are None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)
        if dictionaryA is None:
            if not Dictionary._IsDictionary(dictionaryB):
                if not silent:
                    print("Dictionary.SymDif - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryB
        if dictionaryB is None:
            if not Dictionary._IsDictionary(dictionaryA):
                if not silent:
                    print("Dictionary.SymDif - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.SymDif - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.SymDif - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = Dictionary.Keys(dictionaryB, silent=silent) or []
        out_keys = [k for k in keysA if k not in keysB] + [k for k in keysB if k not in keysA]
        out_vals = []
        for k in out_keys:
            if k in keysA:
                out_vals.append(Dictionary.ValueAtKey(dictionaryA, k, silent=silent))
            else:
                out_vals.append(Dictionary.ValueAtKey(dictionaryB, k, silent=silent))
        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

    @staticmethod
    def SymmetricDifference(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.SymmetricDifference - Warning: Both of the input dictionaries are None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)
        if dictionaryA is None:
            if not Dictionary._IsDictionary(dictionaryB):
                if not silent:
                    print("Dictionary.SymmetricDifference - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryB
        if dictionaryB is None:
            if not Dictionary._IsDictionary(dictionaryA):
                if not silent:
                    print("Dictionary.SymmetricDifference - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.SymmetricDifference - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.SymmetricDifference - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = Dictionary.Keys(dictionaryB, silent=silent) or []
        out_keys = [k for k in keysA if k not in keysB] + [k for k in keysB if k not in keysA]
        out_vals = []
        for k in out_keys:
            if k in keysA:
                out_vals.append(Dictionary.ValueAtKey(dictionaryA, k, silent=silent))
            else:
                out_vals.append(Dictionary.ValueAtKey(dictionaryB, k, silent=silent))
        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

    @staticmethod
    def Union(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the union (merge) of dictionaryA and dictionaryB.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.Union - Warning: Both of the input dictionaries are None. Returning None.")
            return None
        if dictionaryA is None:
            if not Dictionary._IsDictionary(dictionaryB):
                if not silent:
                    print("Dictionary.Union - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryB
        if dictionaryB is None:
            if not Dictionary._IsDictionary(dictionaryA):
                if not silent:
                    print("Dictionary.Union - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.Union - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.Union - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None
        return Dictionary.ByMergedDictionaries(dictionaryA, dictionaryB, silent=silent)

    @staticmethod
    def ValueAtKey(dictionary, key, defaultValue=None, silent=False):
        """
        Returns the value at the input key in the input dictionary.
        If the dictionary or key is invalid, or the key does not exist, defaultValue is returned.
        """
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.ValueAtKey - Error: The input key parameter is not a valid str. Returning defaultValue.")
            return defaultValue

        if isinstance(dictionary, dict):
            return dictionary.get(key, defaultValue)

        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.ValueAtKey - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning defaultValue.")
            return defaultValue

        keys = Dictionary.Keys(dictionary, silent=True)
        if not isinstance(keys, list) or key not in keys:
            return defaultValue

        raw_value = Dictionary._RawValueAtKey(dictionary, key)
        if raw_value is None:
            return defaultValue

        value = Dictionary._ConvertAttribute(raw_value)
        if value is None:
            return defaultValue
        return value

    @staticmethod
    def Values(dictionary, silent: bool = False):
        """
        Returns the list of values in the input dictionary.
        """
        if isinstance(dictionary, dict):
            return list(dictionary.values())

        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.Values - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            return None

        keys = Dictionary.Keys(dictionary, silent=silent)
        if keys is None:
            return None
        return [Dictionary.ValueAtKey(dictionary, key, silent=silent) for key in keys]

    @staticmethod
    def ValuesAtKeys(dictionary, keys, defaultValue=None, silent: bool = False):
        """
        Returns the list of values of the input list of keys in the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary, dict, or backend dictionary
            The input dictionary.
        keys : list
            The input list of keys.
        defaultValue : any , optional
            The default value to return if the key or value are not found. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of values found at the input list of keys in the input dictionary.

        """
        if not Dictionary._IsDictionary(dictionary):
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input dictionary parameter is not a valid topologic, python, or backend dictionary. Returning None.")
            return None
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not all(isinstance(k, str) for k in keys):
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter contains invalid values. Returning None.")
            return None
        return [Dictionary.ValueAtKey(dictionary, key, defaultValue=defaultValue, silent=silent) for key in keys]

    @staticmethod
    def XOR(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.XOR - Warning: Both of the input dictionaries are None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)
        if dictionaryA is None:
            if not Dictionary._IsDictionary(dictionaryB):
                if not silent:
                    print("Dictionary.XOR - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryB
        if dictionaryB is None:
            if not Dictionary._IsDictionary(dictionaryA):
                if not silent:
                    print("Dictionary.XOR - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
                return None
            return dictionaryA
        if not Dictionary._IsDictionary(dictionaryA):
            if not silent:
                print("Dictionary.XOR - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if not Dictionary._IsDictionary(dictionaryB):
            if not silent:
                print("Dictionary.XOR - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        keysA = Dictionary.Keys(dictionaryA, silent=silent) or []
        keysB = Dictionary.Keys(dictionaryB, silent=silent) or []
        out_keys = [k for k in keysA if k not in keysB] + [k for k in keysB if k not in keysA]
        out_vals = []
        for k in out_keys:
            if k in keysA:
                out_vals.append(Dictionary.ValueAtKey(dictionaryA, k, silent=silent))
            else:
                out_vals.append(Dictionary.ValueAtKey(dictionaryB, k, silent=silent))
        return Dictionary.ByKeysValues(out_keys, out_vals, silent=silent)

