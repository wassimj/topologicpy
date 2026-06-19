from __future__ import annotations


class Dictionary:
    def __init__(self, data=None):
        if isinstance(data, Dictionary):
            self._data = dict(data._data)
        elif isinstance(data, dict):
            self._data = dict(data)
        elif hasattr(data, "_data") and isinstance(data._data, dict):
            self._data = dict(data._data)
        elif hasattr(data, "data") and isinstance(data.data, dict):
            self._data = dict(data.data)
        else:
            self._data = {}

    @staticmethod
    def ByKeysValues(keys, values):
        if keys is None or values is None:
            return Dictionary({})
        try:
            if len(keys) != len(values):
                return Dictionary({})
        except Exception:
            return Dictionary({})
        return Dictionary({str(k): v for k, v in zip(keys, values)})

    @staticmethod
    def ByKeyValue(key, value):
        if key is None:
            return Dictionary({})
        return Dictionary({str(key): value})

    @staticmethod
    def ByPythonDictionary(data):
        return Dictionary(data)

    def Keys(self_or_dictionary=None):
        dictionary = self_or_dictionary
        if dictionary is None:
            return []
        if isinstance(dictionary, Dictionary):
            return list(dictionary._data.keys())
        if isinstance(dictionary, dict):
            return list(dictionary.keys())
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            return list(dictionary._data.keys())
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            return list(dictionary.data.keys())
        if hasattr(dictionary, "PythonDictionary"):
            try:
                d = dictionary.PythonDictionary()
                if isinstance(d, dict):
                    return list(d.keys())
            except Exception:
                pass
        return []

    def Values(self_or_dictionary=None):
        dictionary = self_or_dictionary
        if dictionary is None:
            return []
        if isinstance(dictionary, Dictionary):
            return list(dictionary._data.values())
        if isinstance(dictionary, dict):
            return list(dictionary.values())
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            return list(dictionary._data.values())
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            return list(dictionary.data.values())
        if hasattr(dictionary, "PythonDictionary"):
            try:
                d = dictionary.PythonDictionary()
                if isinstance(d, dict):
                    return list(d.values())
            except Exception:
                pass
        return []

    def ValueAtKey(self_or_dictionary=None, key=None):
        dictionary = self_or_dictionary
        if dictionary is None or key is None:
            return None
        if isinstance(dictionary, Dictionary):
            return dictionary._data.get(key)
        if isinstance(dictionary, dict):
            return dictionary.get(key)
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            return dictionary._data.get(key)
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            return dictionary.data.get(key)
        if hasattr(dictionary, "PythonDictionary"):
            try:
                d = dictionary.PythonDictionary()
                if isinstance(d, dict):
                    return d.get(key)
            except Exception:
                pass
        return None

    def SetValueAtKey(self_or_dictionary=None, key=None, value=None):
        dictionary = self_or_dictionary
        if key is None:
            return dictionary
        key = str(key)
        if dictionary is None:
            return Dictionary({key: value})
        if isinstance(dictionary, Dictionary):
            dictionary._data[key] = value
            return dictionary
        if isinstance(dictionary, dict):
            dictionary[key] = value
            return dictionary
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            dictionary._data[key] = value
            return dictionary
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            dictionary.data[key] = value
            return dictionary
        return Dictionary({key: value})

    def RemoveKey(self_or_dictionary=None, key=None):
        dictionary = self_or_dictionary
        if dictionary is None or key is None:
            return dictionary
        if isinstance(dictionary, Dictionary):
            dictionary._data.pop(key, None)
            return dictionary
        if isinstance(dictionary, dict):
            dictionary.pop(key, None)
            return dictionary
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            dictionary._data.pop(key, None)
            return dictionary
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            dictionary.data.pop(key, None)
            return dictionary
        return dictionary

    def PythonDictionary(self_or_dictionary=None):
        dictionary = self_or_dictionary
        if dictionary is None:
            return {}
        if isinstance(dictionary, Dictionary):
            return dict(dictionary._data)
        if isinstance(dictionary, dict):
            return dict(dictionary)
        if hasattr(dictionary, "_data") and isinstance(dictionary._data, dict):
            return dict(dictionary._data)
        if hasattr(dictionary, "data") and isinstance(dictionary.data, dict):
            return dict(dictionary.data)
        return {}

    def __repr__(self):
        return f"Dictionary({self._data})"
