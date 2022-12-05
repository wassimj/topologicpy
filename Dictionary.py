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
    def ByKeysValues(keys, values):
        """
        Description
        __________
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
        Description
        __________
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
    def Keys(dictionary):
        """
        Description
        __________
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
        elif isinstance(dictionary, Dictionary):
            return dictionary.Keys()
        else:
            return None
        
    @staticmethod
    def SetValueAtKey(dictionary, key, value):
        """
        Description
        __________
            Creates a key/value pair in the input dictionary.

        Parameters
        ----------
        dicitonary : topologic.DIctionary or dict
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
    def _listAttributeValues(listAttribute):
        """
        Returns the list of values embedded in the input listAttribute

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
    def ValueAtKey(dictionary, key):
        """
        Description
        __________
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
            return (Dictionary.listAttributeValues(attr))
        elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str):
            return attr
        elif isinstance(attr, list):
            return Dictionary.listAttributeValues(attr)
        elif isinstance(attr, dict):
            return attr
        else:
            return None
        
    @staticmethod
    def Values(dictionary):
        """
        Description
        __________
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
        if isinstance(dictionary, dict):
            keys = dictionary.keys()
        elif isinstance(dictionary, Dictionary):
            keys = dictionary.Keys()
        returnList = []
        for key in keys:
            try:
                if isinstance(dictionary, dict):
                    attr = dictionary[key]
                elif isinstance(dictionary, Dictionary):
                    attr = dictionary.ValueAtKey(key)
                else:
                    attr = None
            except:
                return None
            if isinstance(attr, IntAttribute):
                returnList.append(attr.IntValue())
            elif isinstance(attr, DoubleAttribute):
                returnList.append(attr.DoubleValue())
            elif isinstance(attr, StringAttribute):
                returnList.append(attr.StringValue())
            elif isinstance(attr, ListAttribute):
                returnList.append(Dictionary.listAttributeValues(attr))
            elif isinstance(attr, float) or isinstance(attr, int) or isinstance(attr, str):
                returnList.append(attr)
            elif isinstance(attr, list):
                returnList.append(Dictionary.listAttributeValues(attr))
            else:
                returnList.append("")
        return returnList
    
    

