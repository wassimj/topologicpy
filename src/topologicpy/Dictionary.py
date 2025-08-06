# Copyright (C) 2025
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

import topologic_core as topologic
from topologic_core import IntAttribute, DoubleAttribute, StringAttribute, ListAttribute

class Dictionary():
    '''
    @staticmethod
    def ByDGLData(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        dictionaries : TYPE
            DESCRIPTION.

        """
        keys = list(item.keys())
        vList = []
        for k in keys:
            vList.append(item[k].tolist())
        dictionaries = []
        for v in range(len(vList[0])):
            values = []
            for k in range(len(keys)):
                value = vList[k][v]
                values.append(value)
            dictionaries.append(Dictionary.ByKeysValues(keys, values))
        return dictionaries
    '''
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
            If the labelKey does not exist, it will be created and the subTopologies are labelled numerically and stored in the subTopologies' dictionary under this key. The default is None.
        weightKey : str , optional
            If set, the sharedTopologies' dictionaries will be searched for this key to set their weight. If the key is set to "Area" or "Length" (case insensitive), the area of shared faces or the length of the shared edges will be used as its weight. If set to None, a weight of 1 will be used. The default is None.
        includeWeights : bool , optional
            If set to True, edge weights are included. Otherwise, they are not. The default is False.
        mantissa : int , optional
                The desired length of the mantissa. The default is 6.
        silent : bool , optional
                If set to True, error and warning messages are suppressed. The default is False.

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
            If set to True, error and warning messages are suppressed. The default is False.

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
    def _ConvertValue(value):
        """
        Converts the input value to the proper attribute
        """
        from topologicpy.Topology import Topology
        import json

        def dict_to_json(py_dict):
            """
            Convert a Python dictionary to a JSON-formatted string.
            """
            return json.dumps(py_dict, indent=2)
        
        attr = topologic.StringAttribute("__NONE__") # Hook to Core
        if value == None:
            attr = topologic.StringAttribute("__NONE__") # Hook to Core
        elif isinstance(value, bool):
            if value == False:
                attr = topologic.IntAttribute(0) # Hook to Core
            else:
                attr = topologic.IntAttribute(1) # Hook to Core
        elif isinstance(value, int):
            attr = topologic.IntAttribute(value) # Hook to Core
        elif isinstance(value, float):
            attr = topologic.DoubleAttribute(value) # Hook to Core
        elif Topology.IsInstance(value, "Topology"):
            str_value = Topology.JSONString(value)
            attr = topologic.StringAttribute(str_value) # Hook to Core
        elif isinstance(value, dict):
            str_value = dict_to_json(value)
            attr = topologic.StringAttribute(str_value) # Hook to Core
        elif isinstance(value, str):
            attr = topologic.StringAttribute(value) # Hook to Core
        elif isinstance(value, tuple):
            l = [Dictionary._ConvertValue(v) for v in list(value)]
            attr = topologic.ListAttribute(l) # Hook to Core
        elif isinstance(value, list):
            l = [Dictionary._ConvertValue(v) for v in value]
            attr = topologic.ListAttribute(l) # Hook to Core
        else:
            attr = topologic.StringAttribute("__NONE__") # Hook to Core
        return attr
    
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
            If set to True, error and warning messages are suppressed. The default is False.

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
        return topologic.Dictionary.ByKeysValues(stl_keys, stl_values) # Hook to Core
    
    @staticmethod
    def ByMergedDictionaries(*dictionaries, silent: bool = False):
        """
        Creates a dictionary by merging the list of input dictionaries.

        Parameters
        ----------
        dictionaries : list or comma separated dictionaries
            The input list of dictionaries to be merged.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if isinstance(dictionaries, tuple):
            dictionaries = Helper.Flatten(list(dictionaries))
        if isinstance(dictionaries, list):
            new_dictionaries = [d for d in dictionaries if Topology.IsInstance(d, "Dictionary")]
        if len(new_dictionaries) == 0:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: the input dictionaries parameter does not contain any valid dictionaries. Returning None.")
            return None
        if len(new_dictionaries) == 1:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: the input dictionaries parameter contains only one dictionary. Returning input dictionary.")
            return new_dictionaries[0]
        if not isinstance(new_dictionaries, list):
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: The input dictionaries parameter is not a valid list. Returning None.")
            return None
        return_dictionaries = []
        for d in new_dictionaries:
            if Topology.IsInstance(d, "Dictionary"):
                return_dictionaries.append(d)
            elif isinstance(d, dict):
                return_dictionaries.append(Dictionary.ByPythonDictionary(d))
        if len(return_dictionaries) == 0:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: The input dictionaries parameter does not contain valid dictionaries. Returning None.")
            return None
        elif len(return_dictionaries) == 1:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Warning: The input dictionaries parameter contains only one valid dictionary. Returning that dictionary.")
            return new_dictionaries[0]
        else:
            dictionaries = return_dictionaries
        
        sinkKeys = []
        sinkValues = []
        d = dictionaries[0]
        if d != None:
            stlKeys = d.Keys()
            if len(stlKeys) > 0:
                sinkKeys = Dictionary.Keys(d)
                sinkValues = Dictionary.Values(d)
            for i in range(1,len(dictionaries)):
                d = dictionaries[i]
                if d == None:
                    continue
                stlKeys = Dictionary.Keys(d)
                if len(stlKeys) > 0:
                    sourceKeys = Dictionary.Keys(d)
                    for aSourceKey in sourceKeys:
                        if aSourceKey not in sinkKeys:
                            sinkKeys.append(aSourceKey)
                            sinkValues.append("")
                    for i in range(len(sourceKeys)):
                        index = sinkKeys.index(sourceKeys[i])
                        sourceValue = Dictionary.ValueAtKey(d,sourceKeys[i])
                        if sourceValue != None:
                            if sinkValues[index] != "":
                                if isinstance(sinkValues[index], list):
                                    if not sourceValue in sinkValues[index]:
                                        sinkValues[index].append(sourceValue)
                                else:
                                    if not sourceValue == sinkValues[index]:
                                        sinkValues[index] = [sinkValues[index], sourceValue]
                            else:
                                sinkValues[index] = sourceValue
        if len(sinkKeys) > 0 and len(sinkValues) > 0:
            newDict = Dictionary.ByKeysValues(sinkKeys, sinkValues)
            return newDict
        return None
    '''
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
    '''

    @staticmethod
    def ByPythonDictionary(pythonDictionary, silent: bool = False):
        """
        Creates a dictionary equivalent to the input python dictionary.

        Parameters
        ----------
        pythonDictionary : dict
            The input python dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

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

        Parameters
        ----------
        dictionary : topologic_core.Dictionary
            The input dictionary.

        Returns
        -------
        topologic_core.Dictionary
            A copy of the input dictionary.

        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(dictionary, "dictionary"):
            if not silent:
                print("Dictionary.Copy - Error: The input dictionary parameter is not a valid dictionary. Returning None.")
            return None
        keys = Dictionary.Keys(dictionary)
        values = Dictionary.Values(dictionary)
        return Dictionary.ByKeysValues(keys, values)
        
    @staticmethod
    def Filter(elements, dictionaries, searchType="any", key=None, value=None):
        """
        Filters the input list of dictionaries based on the input parameters.

        Parameters
        ----------
        elements : list
            The input list of elements to be filtered according to the input dictionaries.
        dictionaries : list
            The input list of dictionaries to be filtered.
        searchType : str , optional
            The type of search query to conduct in the topology's dictionary. This can be one of "any", "equal to", "contains", "starts with", "ends with", "not equal to", "does not contain". The default is "any".
        key : str , optional
            The dictionary key to search within. The default is None which means it will filter by topology type only.
        value : str , optional
            The value to search for at the specified key. The default is None which means it will filter by topology type only.

        Returns
        -------
        dict
            A dictionary of filtered and other elements. The dictionary has two keys:
            - "filtered" The filtered dictionaries.
            - "other" The other dictionaries that did not meet the filter criteria.
            - "filteredIndices" The filtered indices of dictionaries.
            - "otherIndices" The other indices of dictionaries that did not meet the filter criteria.

        """

        from topologicpy.Topology import Topology

        def listToString(item):
            returnString = ""
            if isinstance(item, list):
                if len(item) < 2:
                    return str(item[0])
                else:
                    returnString = item[0]
                    for i in range(1, len(item)):
                        returnString = returnString+str(item[i])
            return returnString
        
        filteredDictionaries = []
        otherDictionaries = []
        filteredElements = []
        otherElements = []
        filteredIndices = []
        otherIndices = []
        for i, aDictionary in enumerate(dictionaries):
            if not Topology.IsInstance(aDictionary, "Dictionary"):
                continue
            if value == "" or key == "" or value == None or key == None:
                filteredDictionaries.append(aDictionary)
                filteredIndices.append(i)
            else:
                if isinstance(value, list):
                    value.sort()
                value = str(value)
                value.replace("*",".+")
                value = value.lower()
                v = Dictionary.ValueAtKey(aDictionary, key)
                if v == None:
                    otherDictionaries.append(aDictionary)
                    otherIndices.append(i)
                    otherElements.append(elements[i])
                else:
                    v = str(v).lower()
                    if searchType.lower() == "equal to":
                        searchResult = (value == v)
                    elif searchType.lower() == "contains":
                        searchResult = (value in v)
                    elif searchType.lower() == "starts with":
                        searchResult = (value == v[0: len(value)])
                    elif searchType.lower() == "ends with":
                        searchResult = (value == v[len(v)-len(value):len(v)])
                    elif searchType.lower() == "not equal to":
                        searchResult = not (value == v)
                    elif searchType.lower() == "does not contain":
                        searchResult = not (value in v)
                    else:
                        searchResult = False
                    if searchResult == True:
                        filteredDictionaries.append(aDictionary)
                        filteredIndices.append(i)
                        filteredElements.append(elements[i])
                    else:
                        otherDictionaries.append(aDictionary)
                        otherIndices.append(i)
                        otherElements.append(elements[i])
        return {"filteredDictionaries": filteredDictionaries, "otherDictionaries": otherDictionaries, "filteredIndices": filteredIndices, "otherIndices": otherIndices, "filteredElements": filteredElements, "otherElements": otherElements}

    @staticmethod
    def Keys(dictionary, silent: bool = False):
        """
        Returns the keys of the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        list
            The list of keys of the input dictionary.

        """
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.Keys - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if isinstance(dictionary, dict):
            return list(dictionary.keys())
        elif Topology.IsInstance(dictionary, "Dictionary"):
            return dictionary.Keys()
        else:
            return None

    @staticmethod
    def ListAttributeValues(listAttribute):
        """
        Returns the list of values embedded in the input listAttribute.

        Parameters
        ----------
        listAttribute : listAttribute
            The input list attribute.
 
        Returns
        -------
        list
            The list of values found in the input list attribute

        """
        listAttributes = listAttribute.ListValue()
        returnList = [Dictionary._ConvertAttribute(attr) for attr in listAttributes]
        return returnList    
       
    @staticmethod
    def PythonDictionary(dictionary, silent: bool = False):
        """
        Returns the input dictionary as a python dictionary

        Parameters
        ----------
        dictionary : topologic_core.Dictionary
            The input dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        dict
            The python dictionary equivalent of the input dictionary

        """
        from topologicpy.Topology import Topology

        if isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.PythonDictionary - Warning: The input dictionary parameter is already a python dictionary. Returning that dictionary.")
            return dictionary
        if not Topology.IsInstance(dictionary, "Dictionary"):
            if not silent:
                print("Dictionary.PythonDictionary - Error: The input dictionary parameter is not a valid topologic dictionary. Returning None.")
            return None
        keys = dictionary.Keys()
        pythonDict = {}
        for key in keys:
            try:
                attr = dictionary.ValueAtKey(key)
            except:
                raise Exception("Dictionary.Values - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute): # Hook to Core
                pythonDict[key] = (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute): # Hook to Core
                pythonDict[key] = (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute): # Hook to Core
                temp_str = attr.StringValue()
                if temp_str == "__NONE__":
                    pythonDict[key] = None
                else:
                    pythonDict[key] = (temp_str)
            elif isinstance(attr, topologic.ListAttribute): # Hook to Core
                pythonDict[key] = (Dictionary.ListAttributeValues(attr))
            else:
                pythonDict[key]=("")
        return pythonDict

    @staticmethod
    def RemoveKey(dictionary, key, silent: bool = False):
        """
        Removes the key (and its associated value) from the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        key : string
            The input key.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        topologic_core.Dictionary or dict
            The input dictionary with the key/value removed from it.

        """
        from topologicpy.Topology import Topology

        def processPythonDictionary (dictionary, key):
            values = []
            keys = dictionary.keys()
            new_dict = {}
            for k in keys:
                if not key.lower() == k.lower():
                    new_dict[key] = dictionary[key]
            return new_dict

        def processTopologicDictionary(dictionary, key):
            keys = dictionary.Keys()
            new_keys = []
            new_values = []
            for k in keys:
                if not key.lower() == k.lower():
                    new_keys.append(k)
                    new_values.append(Dictionary.ValueAtKey(dictionary, k))
            return Dictionary.ByKeysValues(new_keys, new_values)
        
        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.RemoveKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.RemoveKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        if isinstance(dictionary, dict):
            return processPythonDictionary(dictionary, key)
        elif Topology.IsInstance(dictionary, "Dictionary"):
            return processTopologicDictionary(dictionary, key)
        else:
            return None
        
    @staticmethod
    def SetValueAtKey(dictionary, key, value, silent: bool = False):
        """
        Creates a key/value pair in the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        key : string
            The input key.
        value : int , float , string, or list
            The value associated with the key.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        topologic_core.Dictionary or dict
            The input dictionary with the key/value pair added to it.

        """
        from topologicpy.Topology import Topology

        def processPythonDictionary (dictionary, key, value):
            if value == "__NONE__":
                value = None
            dictionary[key] = value
            return dictionary

        def processTopologicDictionary(dictionary, key, value):
            keys = dictionary.Keys()
            if not key in keys:
                keys.append(key)
            values = []
            for k in keys:
                if k == key:
                    values.append(value)
                else:
                    values.append(Dictionary.ValueAtKey(dictionary, k))
            d = Dictionary.ByKeysValues(keys, values)
            return d
        
        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input key parameter is not a valid string. Returning None.")
            return None
        if value == None:
            value = "__NONE__"
        if isinstance(dictionary, dict):
            return processPythonDictionary(dictionary, key, value)
        elif Topology.IsInstance(dictionary, "Dictionary"):
            return processTopologicDictionary(dictionary, key, value)
        else:
            return None
    
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
            If set to True, error and warning messages are suppressed. The default is False.

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
    def _ConvertAttribute(attr):
        """
            Convert the found attribute into the proper value
        """
        from topologicpy.Topology import Topology
        import json

        def is_json_string(input_string):
            """
            Check if the input string is a valid JSON string.
            """
            try:
                json.loads(input_string)
                return True
            except json.JSONDecodeError:
                return False

        def json_to_dict(json_string):
            """
            Convert a JSON-formatted string to a Python dictionary.
            """
            return json.loads(json_string)
        
        if isinstance(attr, IntAttribute):
            return (attr.IntValue())
        elif isinstance(attr, DoubleAttribute):
            return (attr.DoubleValue())
        elif isinstance(attr, StringAttribute):
            temp_value = attr.StringValue()
            topologies = None
            try:
                topologies = Topology.ByJSONString(temp_value)
            except:
                topologies = None
            if isinstance(topologies, list):
                if len(topologies) == 0:
                    topologies = None
            if temp_value == "__NONE__":
                return None
            elif Topology.IsInstance(topologies, "Topology"):
                return topologies
            elif isinstance(topologies, list):
                if len(topologies) > 1:
                    return topologies
                elif len(topologies) == 1:
                    return topologies[0]
            elif is_json_string(temp_value):
                ret_value = json_to_dict(temp_value)
                return ret_value
            else:
                return (temp_value)
        elif isinstance(attr, ListAttribute):
            return (Dictionary.ListAttributeValues(attr))
        elif isinstance(attr, float) or isinstance(attr, int):
            return attr
        elif isinstance(attr, str):
            topologies = Topology.ByJSONString(attr)
            if attr == "__NONE__":
                return None
            elif len(topologies) > 1:
                return topologies
            elif len(topologies) == 1:
                return topologies[0]
            elif is_json_string(attr):
                return json_to_dict(attr)
            else:
                return (attr)
        elif isinstance(attr, tuple):
            return Dictionary.ListAttributeValues([Dictionary._ConvertAttribute(x) for x in list(attr)])
        elif isinstance(attr, list):
            return Dictionary.ListAttributeValues([Dictionary._ConvertAttribute(x) for x in attr])
        elif isinstance(attr, dict):
            return attr
        else:
            return None
    
    @staticmethod
    def ValueAtKey(dictionary, key, defaultValue=None, silent=False):
        """
        Returns the value of the input key in the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        key : string
            The input key.
        defaultValue : any , optional
            The default value to return if the key or value are not found. The default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        int , float, string, list , or dict
            The value found at the input key in the input dictionary.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent == True:
                print("Dictionary.ValueAtKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent == True:
                print("Dictionary.ValueAtKey - Error: The input key parameter is not a valid str. Returning None.")
            return None
        if Topology.IsInstance(dictionary, "Dictionary"):
            dic = Dictionary.PythonDictionary(dictionary)
            return dic.get(key, defaultValue)
        elif isinstance(dictionary, dict):
            return dictionary.get(key, defaultValue)
        return defaultValue
    
    @staticmethod
    def Values(dictionary, silent: bool = False):
        """
        Returns the list of values in the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        list
            The list of values found in the input dictionary.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent:
                print("Dictionary.Values - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        keys = None
        if isinstance(dictionary, dict):
            keys = dictionary.keys()
        elif Topology.IsInstance(dictionary, "Dictionary"):
            keys = dictionary.Keys()
        returnList = []
        for key in keys:
            try:
                if isinstance(dictionary, dict):
                    attr = dictionary.get(key, None)
                elif Topology.IsInstance(dictionary, "Dictionary"):
                    attr = Dictionary.ValueAtKey(dictionary,key)
                else:
                    attr = None
            except:
                print(f"Dictionary.Values - Error: Encountered an error in a key in the dictionary ({key}). Returning None.")
                return None
            returnList.append(attr)
        return returnList
    
    @staticmethod
    def ValuesAtKeys(dictionary, keys, defaultValue=None, silent: bool = False):
        """
        Returns the list of values of the input list of keys in the input dictionary.

        Parameters
        ----------
        dictionary : topologic_core.Dictionary or dict
            The input dictionary.
        keys : list
            The input list of keys.
        defaultValue : any , optional
            The default value to return if the key or value are not found. The default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. The default is False.

        Returns
        -------
        list
            The list of values found at the input list of keys in the input dictionary.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(dictionary, "Dictionary") and not isinstance(dictionary, dict):
            if not silent == True:
                print("Dictionary.ValuesAtKeys - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(keys, list):
            if not silent == True:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        
        local_keys = [k for k in keys if isinstance(k, str)]
        if len(local_keys) == 0:
            if not silent == True:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter does not contain valid key strings. Returning None.")
            return None
        if not len(local_keys) == len(keys):
            if not silent == True:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter contains invalid values. Returning None.")
            return None
        return [Dictionary.ValueAtKey(dictionary, key, defaultValue=defaultValue, silent=silent) for key in keys]
    

