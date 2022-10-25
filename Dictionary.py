import topologic
from topologic import IntAttribute, DoubleAttribute, StringAttribute, ListAttribute

class Dictionary(topologic.Dictionary):
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
    
    @staticmethod
    def processKeysValues(keys, values):
        """
        Parameters
        ----------
        keys : TYPE
            DESCRIPTION.
        values : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        myDict : TYPE
            DESCRIPTION.

        """
        if len(keys) != len(values):
            raise Exception("DictionaryByKeysValues - Keys and Values do not have the same length")
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
                raise Exception("Error: Value type is not supported. Supported types are: Boolean, Integer, Double, String, or List.")
        myDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
        return myDict
    
    @staticmethod
    def ByKeysValues(keys, values):
        """
        Parameters
        ----------
        keys : TYPE
            DESCRIPTION.
        values : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # keys = item[0]
        # values = item[1]
        
        if isinstance(keys, list) == False:
            keys = [keys]
        if isinstance(values, list) == False:
            values = [values]
        return Dictionary.processKeysValues(keys, values)
    
    @staticmethod
    def ByMergedDictionaries(sources):
        """
        Parameters
        ----------
        sources : TYPE
            DESCRIPTION.

        Returns
        -------
        newDict : TYPE
            DESCRIPTION.

        """
        sinkKeys = []
        sinkValues = []
        d = sources[0]
        if d != None:
            stlKeys = d.Keys()
            if len(stlKeys) > 0:
                sinkKeys = d.Keys()
                sinkValues = Dictionary.Values(d)
            for i in range(1,len(sources)):
                d = sources[i]
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
    def Keys(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(item, dict):
            return list(item.keys())
        elif isinstance(item, Dictionary):
            return item.Keys()
        else:
            return None
        
    @staticmethod
    def SetValueAtKey(item, key, value):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
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
        def processPythonDictionary (item, key, value):
            item[key] = value
            return item

        def processTopologicDictionary(item, key, value):
            keys = item.Keys()
            if not key in keys:
                keys.append(key)
            values = []
            for k in keys:
                if k == key:
                    values.append(value)
                else:
                    values.append(Dictionary.ValueAtKey(item, k))
            return Dictionary.processKeysValues(keys, values)

        if isinstance(item, dict):
            return processPythonDictionary(item, key, value)
        elif isinstance(item, Dictionary):
            return processTopologicDictionary(item, key, value)
        else:
            raise Exception("Dictionary.SetValueAtKey - Error: Input is not a dictionary")
    
    @staticmethod
    def listAttributeValues(listAttribute):
        """
        Parameters
        ----------
        listAttribute : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

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
    def ValueAtKey(d, key):
        """
        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        key : TYPE
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
        # d, key = item
        
        try:
            if isinstance(d, dict):
                attr = d[key]
            elif isinstance(d, Dictionary):
                attr = d.ValueAtKey(key)
        except:
            raise Exception("Dictionary.ValueAtKey - Error: Could not retrieve a Value at the specified key ("+key+")")
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
    def Values(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        if isinstance(item, dict):
            keys = item.keys()
        elif isinstance(item, Dictionary):
            keys = item.Keys()
        returnList = []
        for key in keys:
            try:
                if isinstance(item, dict):
                    attr = item[key]
                elif isinstance(item, Dictionary):
                    attr = item.ValueAtKey(key)
                else:
                    attr = None
            except:
                raise Exception("Dictionary.Values - Error: Could not retrieve a Value at the specified key ("+key+")")
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
    
    

