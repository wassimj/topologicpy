import topologic
from topologic import IntAttribute, DoubleAttribute, StringAttribute, ListAttribute

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
        if not isinstance(keys, list) or not isinstance(values, list):
            return None
        if len(keys) != len(values):
            return None
        stl_keys = []
        stl_values = []
        for i in range(len(keys)):
            if isinstance(keys[i], str):
                stl_keys.append(keys[i])
            else:
                stl_keys.append(str(keys[i]))
            if isinstance(values[i], list) and len(values[i]) == 1:
                value = values[i][0]
            else:
                value = values[i]
            if isinstance(value, bool):
                if value == False:
                    stl_values.append(topologic.IntAttribute(0))
                else:
                    stl_values.append(topologic.IntAttribute(1))
            elif isinstance(value, int):
                stl_values.append(topologic.IntAttribute(value))
            elif isinstance(value, float):
                stl_values.append(topologic.DoubleAttribute(value))
            elif isinstance(value, str):
                stl_values.append(topologic.StringAttribute(value))
            elif isinstance(value, tuple):
                value = list(value)
                l = []
                for v in value:
                    if isinstance(v, bool):
                        l.append(topologic.IntAttribute(v))
                    elif isinstance(v, int):
                        l.append(topologic.IntAttribute(v))
                    elif isinstance(v, float):
                        l.append(topologic.DoubleAttribute(v))
                    elif isinstance(v, str):
                        l.append(topologic.StringAttribute(v))
                stl_values.append(topologic.ListAttribute(l))
            elif isinstance(value, list):
                l = []
                for v in value:
                    if isinstance(v, bool):
                        l.append(topologic.IntAttribute(v))
                    elif isinstance(v, int):
                        l.append(topologic.IntAttribute(v))
                    elif isinstance(v, float):
                        l.append(topologic.DoubleAttribute(v))
                    elif isinstance(v, str):
                        l.append(topologic.StringAttribute(v))
                stl_values.append(topologic.ListAttribute(l))
            else:
                return None
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
        topologic.DIctionary
            The created dictionary.

        """
        sinkKeys = []
        sinkValues = []
        d = dictionaries[0]
        if d != None:
            stlKeys = d.Keys()
            if len(stlKeys) > 0:
                sinkKeys = d.Keys()
                sinkValues = Dictionary.Values(d)
            for i in range(1,len(dictionaries)):
                d = dictionaries[i]
                if d == None:
                    continue
                stlKeys = d.Keys()
                if len(stlKeys) > 0:
                    sourceKeys = d.Keys()
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
                                    sinkValues[index].append(sourceValue)
                                else:
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
        returnList = []
        for attr in listAttributes:
            if isinstance(attr, IntAttribute):
                returnList.append(attr.IntValue())
            elif isinstance(attr, DoubleAttribute):
                returnList.append(attr.DoubleValue())
            elif isinstance(attr, StringAttribute):
                returnList.append(attr.StringValue())
            elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str) or isinstance(attr, dict):
                returnList.append(attr)
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
        if not isinstance(dictionary, topologic.Dictionary):
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
                pythonDict[key] = (attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                pythonDict[key] = (Dictionary.ListAttributeValues(attr))
            else:
                pythonDict[key]=("")
        return pythonDict

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
            return Dictionary.ByKeysValues(keys, values)

        if isinstance(dictionary, dict):
            return processPythonDictionary(dictionary, key, value)
        elif isinstance(dictionary, topologic.Dictionary):
            return processTopologicDictionary(dictionary, key, value)
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
        if isinstance(dictionary, dict):
            attr = dictionary[key]
        elif isinstance(dictionary, topologic.Dictionary):
            attr = dictionary.ValueAtKey(key)
        else:
            return None
        
        if isinstance(attr, IntAttribute):
            return (attr.IntValue())
        elif isinstance(attr, DoubleAttribute):
            return (attr.DoubleValue())
        elif isinstance(attr, StringAttribute):
            return (attr.StringValue())
        elif isinstance(attr, ListAttribute):
            return (Dictionary.ListAttributeValues(attr))
        elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str):
            return attr
        elif isinstance(attr, list):
            return Dictionary.ListAttributeValues(attr)
        elif isinstance(attr, dict):
            return attr
        else:
            return None
        
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
                    attr = dictionary.ValueAtKey(key)
                else:
                    attr = None
            except:
                return None
            if isinstance(attr, topologic.IntAttribute):
                returnList.append(attr.IntValue())
            elif isinstance(attr, topologic.DoubleAttribute):
                returnList.append(attr.DoubleValue())
            elif isinstance(attr, topologic.StringAttribute):
                returnList.append(attr.StringValue())
            elif isinstance(attr, topologic.ListAttribute):
                returnList.append(Dictionary.ListAttributeValues(attr))
            elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str):
                returnList.append(attr)
            elif isinstance(attr, list):
                returnList.append(Dictionary.ListAttributeValues(attr))
            else:
                returnList.append("")
        return returnList
    
    

