# Copyright (C) 2024
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

class Dictionary(topologic.Dictionary):
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
    def ByKeyValue(key, value):
        """
        Creates a Dictionary from the input key and the input value.

        Parameters
        ----------
        key : str
            The string representing the key of the value in the dictionary.
        value : int, float, str, or list
            A value corresponding to the input key. A value can be an integer, a float, a string, or a list.

        Returns
        -------
        topologic.Dictionary
            The created dictionary.

        """
        if not isinstance(key, str):
            print("Dictionary.ByKeyValue - Error: The input key is not a valid string. Returning None.")
            return None
        return Dictionary.ByKeysValues([key], [value])
    
    
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
        
        attr = topologic.StringAttribute("__NONE__")
        if value == None:
            attr = topologic.StringAttribute("__NONE__")
        elif isinstance(value, bool):
            if value == False:
                attr = topologic.IntAttribute(0)
            else:
                attr = topologic.IntAttribute(1)
        elif isinstance(value, int):
            attr = topologic.IntAttribute(value)
        elif isinstance(value, float):
            attr = topologic.DoubleAttribute(value)
        elif isinstance(value, topologic.Topology):
            str_value = Topology.JSONString(value)
            attr = topologic.StringAttribute(str_value)
        elif isinstance(value, dict):
            str_value = dict_to_json(value)
            attr = topologic.StringAttribute(str_value)
        elif isinstance(value, str):
            attr = topologic.StringAttribute(value)
        elif isinstance(value, tuple):
            l = [Dictionary._ConvertValue(v) for v in list(value)]
            attr = topologic.ListAttribute(l)
        elif isinstance(value, list):
            l = [Dictionary._ConvertValue(v) for v in value]
            attr = topologic.ListAttribute(l)
        else:
            attr = topologic.StringAttribute("__NONE__")
        return attr
    
    @staticmethod
    def ByKeysValues(keys, values):
        """
        Creates a Dictionary from the input list of keys and the input list of values.

        Parameters
        ----------
        keys : list
            A list of strings representing the keys of the dictionary.
        values : list
            A list of values corresponding to the list of keys. Values can be integers, floats, strings, or lists

        Returns
        -------
        topologic.Dictionary
            The created dictionary.

        """
        
        if not isinstance(keys, list):
            print("Dictionary.ByKeysValues - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not isinstance(values, list):
            print("Dictionary.ByKeysValues - Error: The input values parameter is not a valid list. Returning None.")
            return None
        if len(keys) != len(values):
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
        return topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
    
    @staticmethod
    def ByMergedDictionaries(dictionaries):
        """
        Creates a dictionary by merging the list of input dictionaries.

        Parameters
        ----------
        dictionaries : list
            The input list of dictionaries to be merges.

        Returns
        -------
        topologic.Dictionary
            The created dictionary.

        """
        if not isinstance(dictionaries, list):
            print("Dictionary.ByMergedDictionaries - Error: The input dictionaries parameter is not a valid list. Returning None.")
            return None
        new_dictionaries = []
        for d in dictionaries:
            if isinstance(d, topologic.Dictionary):
                new_dictionaries.append(d)
            elif isinstance(d, dict):
                new_dictionaries.append(Dictionary.ByPythonDictionary(d))
        if len(new_dictionaries) == 0:
            print("Dictionary.ByMergedDictionaries - Error: The input dictionaries parameter does not contain valid dictionaries. Returning None.")
            return None
        elif len(new_dictionaries) == 1:
            print("Dictionary.ByMergedDictionaries - Warning: The input dictionaries parameter contains only one valid dictionary. Returning that dictionary.")
            return new_dictionaries[0]
        else:
            dictionaries = new_dictionaries
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
    def ByPythonDictionary(pythonDictionary):
        """
        Creates a dictionary equivalent to the input python dictionary.

        Parameters
        ----------
        pythonDictionary : dict
            The input python dictionary.

        Returns
        -------
        topologic.Dictionary
            The dictionary equivalent to the input python dictionary.

        """
        if not isinstance(pythonDictionary, dict):
            print("Dictionary.ByPythonDictionary - Error: The input dictionary parameter is not a valid python dictionary. Returning None.")
            return None
        keys = list(pythonDictionary.keys())
        values = []
        for key in keys:
            values.append(pythonDictionary[key])
        return Dictionary.ByKeysValues(keys, values)

    @staticmethod
    def Keys(dictionary):
        """
        Returns the keys of the input dictionary.

        Parameters
        ----------
        dictionary : topologic.Dictionary or dict
            The input dictionary.

        Returns
        -------
        list
            The list of keys of the input dictionary.

        """
        if not isinstance(dictionary, topologic.Dictionary) and not isinstance(dictionary, dict):
            print("Dictionary.Keys - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if isinstance(dictionary, dict):
            return list(dictionary.keys())
        elif isinstance(dictionary, topologic.Dictionary):
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
    def PythonDictionary(dictionary):
        """
        Returns the input dictionary as a python dictionary

        Parameters
        ----------
        dictionary : topologic.Dictionary
            The input dictionary.

        Returns
        -------
        dict
            The python dictionary equivalent of the input dictionary

        """
        if isinstance(dictionary, dict):
            print("Dictionary.PythonDictionary - Warning: The input dictionary parameter is already a python dictionary. Returning that dictionary.")
            return dictionary
        if not isinstance(dictionary, topologic.Dictionary):
            print("Dictionary.PythonDictionary - Error: The input dictionary parameter is not a valid topologic dictionary. Returning None.")
            return None
        keys = dictionary.Keys()
        pythonDict = {}
        for key in keys:
            try:
                attr = dictionary.ValueAtKey(key)
            except:
                raise Exception("Dictionary.Values - Error: Could not retrieve a Value at the specified key ("+key+")")
            if isinstance(attr, topologic.IntAttribute):
                pythonDict[key] = (attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                pythonDict[key] = (attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                temp_str = attr.StringValue()
                if temp_str == "__NONE__":
                    pythonDict[key] = None
                else:
                    pythonDict[key] = (temp_str)
            elif isinstance(attr, topologic.ListAttribute):
                pythonDict[key] = (Dictionary.ListAttributeValues(attr))
            else:
                pythonDict[key]=("")
        return pythonDict

    @staticmethod
    def RemoveKey(dictionary, key):
        """
        Removes the key (and its associated value) from the input dictionary.

        Parameters
        ----------
        dictionary : topologic.Dictionary or dict
            The input dictionary.
        key : string
            The input key.

        Returns
        -------
        topologic.Dictionary or dict
            The input dictionary with the key/value removed from it.

        """
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
        
        if not isinstance(dictionary, topologic.Dictionary) and not isinstance(dictionary, dict):
            print("Dictionary.RemoveKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            print("Dictionary.RemoveKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        if isinstance(dictionary, dict):
            return processPythonDictionary(dictionary, key)
        elif isinstance(dictionary, topologic.Dictionary):
            return processTopologicDictionary(dictionary, key)
        else:
            return None
        
    @staticmethod
    def SetValueAtKey(dictionary, key, value):
        """
        Creates a key/value pair in the input dictionary.

        Parameters
        ----------
        dictionary : topologic.Dictionary or dict
            The input dictionary.
        key : string
            The input key.
        value : int , float , string, or list
            The value associated with the key.

        Returns
        -------
        topologic.Dictionary or dict
            The input dictionary with the key/value pair added to it.

        """
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
        
        if not isinstance(dictionary, topologic.Dictionary) and not isinstance(dictionary, dict):
            print("Dictionary.SetValueAtKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            print("Dictionary.SetValueAtKey - Error: The input key parameter is not a valid string. Returning None.")
            return None
        if value == None:
            value = "__NONE__"
        if isinstance(dictionary, dict):
            return processPythonDictionary(dictionary, key, value)
        elif isinstance(dictionary, topologic.Dictionary):
            return processTopologicDictionary(dictionary, key, value)
        else:
            return None
 
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
                topologies = Topology.ByJSONString(temp_value, progressBar=False)
            except:
                topologies = None
            if isinstance(topologies, list):
                if len(topologies) == 0:
                    topologies = None
            if temp_value == "__NONE__":
                return None
            elif isinstance(topologies, topologic.Topology):
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
    def ValueAtKey(dictionary, key):
        """
        Returns the value of the input key in the input dictionary.

        Parameters
        ----------
        dictionary : topologic.Dictionary or dict
            The input dictionary.
        key : string
            The input key.

        Returns
        -------
        int , float, string, list , or dict
            The value found at the input key in the input dictionary.

        """
        import json
        from topologicpy.Topology import Topology
        
        
        if not isinstance(dictionary, topologic.Dictionary) and not isinstance(dictionary, dict):
            print("Dictionary.ValueAtKey - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            print("Dictionary.ValueAtKey - Error: The input key parameter is not a valid str. Returning None.")
            return None
        if isinstance(dictionary, dict):
            attr = dictionary[key]
        elif isinstance(dictionary, topologic.Dictionary):
            attr = dictionary.ValueAtKey(key)
        else:
            return None
        return_value = Dictionary._ConvertAttribute(attr)
        return return_value
        
    @staticmethod
    def Values(dictionary):
        """
        Returns the list of values in the input dictionary.

        Parameters
        ----------
        dictionary : topologic.Dictionary or dict
            The input dictionary.

        Returns
        -------
        list
            The list of values found in the input dictionary.

        """
        if not isinstance(dictionary, topologic.Dictionary) and not isinstance(dictionary, dict):
            print("Dictionary.Values - Error: The input dictionary parameter is not a valid topologic or python dictionary. Returning None.")
            return None
        keys = None
        if isinstance(dictionary, dict):
            keys = dictionary.keys()
        elif isinstance(dictionary, topologic.Dictionary):
            keys = dictionary.Keys()
        returnList = []
        if not keys:
            return None
        for key in keys:
            try:
                if isinstance(dictionary, dict):
                    attr = dictionary[key]
                elif isinstance(dictionary, topologic.Dictionary):
                    attr = Dictionary.ValueAtKey(dictionary,key)
                else:
                    attr = None
            except:
                return None
            returnList.append(attr)
            '''
            if isinstance(attr, topologic.IntAttribute):
                returnList.append(attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                returnList.append(attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                temp_attr = attr.StringValue()
                if temp_attr == "__NONE__":
                    returnList.append(None)
                else:
                    returnList.append(attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                returnList.append(Dictionary.ListAttributeValues(attr))
            elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str):
                returnList.append(attr)
            elif isinstance(attr, list):
                returnList.append(Dictionary.ListAttributeValues(attr))
            else:
                returnList.append("")
            '''
        return returnList
    
    

