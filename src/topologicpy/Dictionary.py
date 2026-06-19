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
    def _IsTGraph(obj) -> bool:
        """
        Returns True if the input object is a topologicpy.TGraph instance.
        """
        try:
            from topologicpy.TGraph import TGraph
            return isinstance(obj, TGraph)
        except Exception:
            return False

    @staticmethod
    def _IsTGraphEdgeRecord(obj) -> bool:
        """
        Returns True if the input object behaves like a TGraph edge record.
        """
        return (
            isinstance(obj, dict) and
            "index" in obj and
            "src" in obj and
            "dst" in obj and
            isinstance(obj.get("dictionary", None), dict)
        )

    @staticmethod
    def _IsTGraphVertexRecord(obj) -> bool:
        """
        Returns True if the input object behaves like a TGraph vertex record.
        """
        return (
            isinstance(obj, dict) and
            "index" in obj and
            "src" not in obj and
            "dst" not in obj and
            isinstance(obj.get("dictionary", None), dict)
        )

    @staticmethod
    def _IsTGraphDictionaryContainer(obj) -> bool:
        """
        Returns True if the input object is a TGraph, a TGraph vertex record, or
        a TGraph edge record.
        """
        return (
            Dictionary._IsTGraph(obj) or
            Dictionary._IsTGraphVertexRecord(obj) or
            Dictionary._IsTGraphEdgeRecord(obj)
        )

    @staticmethod
    def _TGraphDictionary(obj):
        """
        Returns the Python dictionary attached to a TGraph, TGraph vertex record,
        or TGraph edge record.
        """
        if Dictionary._IsTGraph(obj):
            d = getattr(obj, "_dictionary", None)
            if not isinstance(d, dict):
                try:
                    obj._dictionary = {}
                    d = obj._dictionary
                except Exception:
                    d = {}
            return d

        if Dictionary._IsTGraphVertexRecord(obj) or Dictionary._IsTGraphEdgeRecord(obj):
            d = obj.get("dictionary", None)
            if not isinstance(d, dict):
                obj["dictionary"] = {}
                d = obj["dictionary"]
            return d

        return None

    @staticmethod
    def _SetTGraphDictionary(obj, py_dict):
        """
        Sets the Python dictionary attached to a TGraph, TGraph vertex record, or
        TGraph edge record and returns the original object.
        """
        if not isinstance(py_dict, dict):
            return obj

        if Dictionary._IsTGraph(obj):
            try:
                obj.SetDictionary(dict(py_dict))
            except Exception:
                try:
                    obj._dictionary = dict(py_dict)
                    if callable(getattr(obj, "_invalidate_cache", None)):
                        obj._invalidate_cache()
                except Exception:
                    pass
            return obj

        if Dictionary._IsTGraphVertexRecord(obj) or Dictionary._IsTGraphEdgeRecord(obj):
            obj["dictionary"] = dict(py_dict)

            # Preserve structural metadata inside the dictionary when useful.
            if "index" in obj:
                obj["dictionary"].setdefault("index", obj.get("index"))
            if "active" in obj:
                obj["dictionary"].setdefault("active", obj.get("active", True))
            if Dictionary._IsTGraphEdgeRecord(obj):
                obj["dictionary"].setdefault("src", obj.get("src"))
                obj["dictionary"].setdefault("dst", obj.get("dst"))
                obj["dictionary"].setdefault("directed", obj.get("directed", False))
            return obj

        return obj

    @staticmethod
    def _ToPythonDictionary(dictionary, copy: bool = True):
        """
        Converts any supported dictionary-like object to a Python dictionary.

        Supported inputs include Python dictionaries, Core/topologic dictionaries,
        backend dictionary objects, topologicpy.TGraph instances, TGraph vertex
        records, and TGraph edge records.
        """
        if dictionary is None:
            return None

        if Dictionary._IsTGraphDictionaryContainer(dictionary):
            d = Dictionary._TGraphDictionary(dictionary)
            if not isinstance(d, dict):
                return {}
            return dict(d) if copy else d

        if isinstance(dictionary, dict):
            return dict(dictionary) if copy else dictionary

        try:
            if callable(getattr(dictionary, "PythonDictionary", None)):
                py_dict = dictionary.PythonDictionary()
                if isinstance(py_dict, dict):
                    return dict(py_dict) if copy else py_dict
        except Exception:
            pass

        if not Dictionary._IsDictionary(dictionary):
            return None

        keys = Dictionary._Keys(dictionary)
        if keys is None:
            return None

        py = {}
        for key in keys:
            raw_value = Dictionary._RawValueAtKey(dictionary, key)
            py[key] = Dictionary._ConvertAttribute(raw_value)
        return py

    @staticmethod
    def _SetPythonDictionary(dictionary, py_dict):
        """
        Writes a Python dictionary back into a supported mutable container where
        possible. For immutable Core/topologic dictionaries, a new Core/topologic
        dictionary is returned.
        """
        if not isinstance(py_dict, dict):
            return None

        if Dictionary._IsTGraphDictionaryContainer(dictionary):
            return Dictionary._SetTGraphDictionary(dictionary, py_dict)

        if isinstance(dictionary, dict):
            dictionary.clear()
            dictionary.update(py_dict)
            return dictionary

        return Dictionary.ByPythonDictionary(py_dict, silent=True)

    @staticmethod
    def _ValuesMatch(a, b, caseSensitive: bool = True) -> bool:
        """
        Returns True if two dictionary values should be considered equal.
        """
        if a == b:
            return True

        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(a) == float(b)
        except Exception:
            pass

        try:
            sa = str(a)
            sb = str(b)
            if not caseSensitive:
                sa = sa.lower()
                sb = sb.lower()
            return sa == sb
        except Exception:
            return False

    @staticmethod
    def _IsDictionary(dictionary) -> bool:
        """
        Returns True if the input behaves like a supported dictionary.

        Supported inputs are:
        - native Python dictionaries;
        - topologicpy.TGraph instances;
        - TGraph vertex records;
        - TGraph edge records;
        - the active Core/topologic dictionary class;
        - backend dictionary objects exposing Keys() and ValueAtKey(key);
        - backend dictionary objects exposing PythonDictionary().
        """
        if dictionary is None:
            return False

        if Dictionary._IsTGraphDictionaryContainer(dictionary):
            return True

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
        if Dictionary._IsTGraphDictionaryContainer(dictionary):
            d = Dictionary._TGraphDictionary(dictionary)
            return list(d.keys()) if isinstance(d, dict) else []

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
        if Dictionary._IsTGraphDictionaryContainer(dictionary):
            d = Dictionary._TGraphDictionary(dictionary)
            if isinstance(d, dict):
                return d.get(key, None)
            return None

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
    def AdjacencyDictionary(topology, subTopologyType: str = None, labelKey: str = None, weightKey: str = None, includeWeights: bool = False, mantissa: int = 6, silent: bool = False):
        """
        Returns the adjacency dictionary of the input topology or graph.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
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
        from topologicpy.Helper import Helper

        is_tgraph = False
        try:
            from topologicpy.TGraph import TGraph
            is_tgraph = isinstance(topology, TGraph)
        except Exception:
            TGraph = None
            is_tgraph = False

        # ------------------------------------------------------------------
        # Special case for TGraph. This must come before the legacy Graph
        # branch because Topology.IsInstance(tgraph, "Graph") may return True.
        # ------------------------------------------------------------------
        if is_tgraph:
            return TGraph.AdjacencyDictionary(
                topology,
                vertexLabelKey=labelKey,
                edgeKey=weightKey,
                includeWeights=includeWeights
            )

        # ------------------------------------------------------------------
        # Special case for legacy Graph.
        # ------------------------------------------------------------------
        if Topology.IsInstance(topology, "Graph"):
            from topologicpy.Graph import Graph

            return Graph.AdjacencyDictionary(
                topology,
                vertexLabelKey=labelKey,
                edgeKey=weightKey,
                includeWeights=includeWeights,
                mantissa=mantissa
            )

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input topology parameter is not a valid topology or graph. Returning None.")
            return None

        internalLabelKey = labelKey is None

        if labelKey is None:
            labelKey = "__label__"

        if not isinstance(labelKey, str):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input labelKey is not a valid string. Returning None.")
            return None

        if subTopologyType is not None and not isinstance(subTopologyType, str):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input subTopologyType is not a valid string. Returning None.")
            return None

        # ------------------------------------------------------------------
        # Determine default subtopology type.
        # ------------------------------------------------------------------
        if Topology.IsInstance(topology, "cellcomplex"):
            if subTopologyType is None:
                subTopologyType = "cell"
        elif Topology.IsInstance(topology, "cell") or Topology.IsInstance(topology, "shell"):
            if subTopologyType is None:
                subTopologyType = "face"
        elif Topology.IsInstance(topology, "face") or Topology.IsInstance(topology, "wire"):
            if subTopologyType is None:
                subTopologyType = "edge"
        else:
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input topology type is not supported. Returning None.")
            return None

        subTopologyType = subTopologyType.lower()

        if subTopologyType not in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex"]:
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: The input subTopologyType parameter is not recognized. Returning None.")
            return None

        all_subtopologies = Topology.SubTopologies(
            topology,
            subTopologyType=subTopologyType,
            silent=silent
        )

        if not isinstance(all_subtopologies, list):
            if not silent:
                print("Dictionary.AdjacencyDictionary - Error: Could not retrieve subtopologies. Returning None.")
            return None

        if len(all_subtopologies) < 1:
            return {}

        # ------------------------------------------------------------------
        # Label subtopologies.
        # ------------------------------------------------------------------
        labels = []
        n = max(len(str(len(all_subtopologies))), 3)

        for i, subtopology in enumerate(all_subtopologies):
            d = Topology.Dictionary(subtopology)
            value = Dictionary.ValueAtKey(d, labelKey, None)

            if value is None:
                value = str(i + 1).zfill(n)

            if d is None:
                d = Dictionary.ByKeyValue(labelKey, value)
            else:
                d = Dictionary.SetValueAtKey(d, labelKey, value, silent=silent)

            subtopology = Topology.SetDictionary(subtopology, d, silent=silent)
            all_subtopologies[i] = subtopology
            labels.append(value)

        all_subtopologies = Helper.Sort(all_subtopologies, labels)
        labels = sorted(labels)

        adjDict = {}

        # ------------------------------------------------------------------
        # Build adjacency dictionary.
        # ------------------------------------------------------------------
        for i, subtopology in enumerate(all_subtopologies):
            subt_label = labels[i]

            adjacent_topologies = Topology.AdjacentTopologies(
                subtopology,
                hostTopology=topology,
                topologyType=subTopologyType
            )

            if not isinstance(adjacent_topologies, list):
                adjacent_topologies = []

            temp_list = []

            for adj_topology in adjacent_topologies:
                adj_label = Dictionary.ValueAtKey(
                    Topology.Dictionary(adj_topology),
                    labelKey,
                    None
                )

                if adj_label not in labels:
                    continue

                if includeWeights:
                    if weightKey is None:
                        weight = 1

                    elif isinstance(weightKey, str) and "length" in weightKey.lower():
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        edges = shared_topologies.get("edges", []) if isinstance(shared_topologies, dict) else []
                        weight = sum([Edge.Length(edge, mantissa=mantissa) for edge in edges])

                    elif isinstance(weightKey, str) and "area" in weightKey.lower():
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        faces = shared_topologies.get("faces", []) if isinstance(shared_topologies, dict) else []
                        weight = sum([Face.Area(face, mantissa=mantissa) for face in faces])

                    else:
                        shared_topologies = Topology.SharedTopologies(subtopology, adj_topology)
                        if isinstance(shared_topologies, dict):
                            vertices = shared_topologies.get("vertices", [])
                            edges = shared_topologies.get("edges", [])
                            wires = shared_topologies.get("wires", [])
                            faces = shared_topologies.get("faces", [])
                            everything = vertices + edges + wires + faces
                        else:
                            everything = []

                        weight = sum([
                            Dictionary.ValueAtKey(Topology.Dictionary(x), weightKey, 0)
                            for x in everything
                        ])
                        weight = round(weight, mantissa)

                    temp_list.append((adj_label, weight))

                else:
                    temp_list.append(adj_label)

            temp_list.sort()
            adjDict[subt_label] = temp_list

        # ------------------------------------------------------------------
        # Remove temporary labels if this method created them.
        # ------------------------------------------------------------------
        if internalLabelKey:
            for subtopology in all_subtopologies:
                d = Topology.Dictionary(subtopology)
                d = Dictionary.RemoveKey(d, labelKey)
                subtopology = Topology.SetDictionary(subtopology, d, silent=silent)

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

        import inspect
        
        
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input keys parameter is not a valid list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not isinstance(values, list):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input values parameter is not a valid list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if len(keys) != len(values):
            if not silent:
                print("Dictionary.ByKeysValues - Error: The input keys and values parameters are not of equal length. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
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
        Creates a dictionary by merging the input dictionaries.

        The inputs can be Python dictionaries, Core/topologic dictionaries,
        backend dictionary objects, topologicpy.TGraph instances, TGraph vertex
        records, or TGraph edge records. None values are preserved when the key
        exists; missing keys and stored None values are not conflated.

        Parameters
        ----------
        dictionaries : list or comma separated dictionaries
            The input dictionaries to be merged.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The created dictionary.
        """
        from topologicpy.Helper import Helper

        if isinstance(dictionaries, tuple):
            dictionaries = Helper.Flatten(list(dictionaries))
        elif not isinstance(dictionaries, list):
            dictionaries = [dictionaries]

        py_dicts = []
        for d in dictionaries:
            py = Dictionary._ToPythonDictionary(d, copy=True)
            if isinstance(py, dict):
                py_dicts.append(py)

        if len(py_dicts) == 0:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Error: the input dictionaries parameter does not contain any valid dictionaries. Returning None.")
            return None

        if len(py_dicts) == 1:
            if not silent:
                print("Dictionary.ByMergedDictionaries - Warning: the input dictionaries parameter contains only one valid dictionary. Returning that dictionary.")
            return Dictionary.ByPythonDictionary(py_dicts[0], silent=silent)

        sink = {}

        for py in py_dicts:
            for sourceKey, sourceValue in py.items():
                if sourceKey not in sink:
                    sink[sourceKey] = sourceValue
                    continue

                sinkValue = sink[sourceKey]

                if sinkValue is None or sinkValue == "":
                    sink[sourceKey] = sourceValue
                elif isinstance(sinkValue, list):
                    if sourceValue not in sinkValue:
                        sinkValue.append(sourceValue)
                elif sourceValue != sinkValue:
                    sink[sourceKey] = [sinkValue, sourceValue]

        return Dictionary.ByPythonDictionary(sink, silent=silent)


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
        Creates a Core/topologic dictionary equivalent to the input Python
        dictionary or dictionary-bearing TGraph object/record.

        Parameters
        ----------
        pythonDictionary : dict, topologicpy.TGraph, TGraph vertex record, or TGraph edge record
            The input dictionary-like object.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Dictionary
            The dictionary equivalent to the input dictionary.
        """
        if Dictionary._IsTGraphDictionaryContainer(pythonDictionary):
            pythonDictionary = Dictionary._ToPythonDictionary(pythonDictionary, copy=True)

        if not isinstance(pythonDictionary, dict):
            if not silent:
                print("Dictionary.ByPythonDictionary - Error: The input dictionary parameter is not a valid python dictionary. Returning None.")
            return None

        keys = list(pythonDictionary.keys())
        values = [pythonDictionary[key] for key in keys]
        return Dictionary.ByKeysValues(keys, values, silent=silent)

    @staticmethod
    def Copy(dictionary, silent: bool = False):
        """
        Creates a copy of the input dictionary-like object as a Core/topologic
        dictionary.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.Copy - Error: The input dictionary parameter is not a valid dictionary. Returning None.")
            return None
        return Dictionary.ByPythonDictionary(py, silent=silent)

    @staticmethod
    def Difference(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the difference of dictionaryA and dictionaryB (A \\ B), based on keys.
        """
        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        if pyA is None:
            if dictionaryA is None:
                if not silent:
                    print("Dictionary.Difference - Warning: The dictionaryA input parameter is None. Returning None.")
            elif not silent:
                print("Dictionary.Difference - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None

        if dictionaryB is None:
            if not silent:
                print("Dictionary.Difference - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return Dictionary.ByPythonDictionary(pyA, silent=silent)

        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)
        if pyB is None:
            if not silent:
                print("Dictionary.Difference - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        return Dictionary.ByPythonDictionary({k: v for k, v in pyA.items() if k not in pyB}, silent=silent)

    @staticmethod
    def Filter(elements, dictionaries, searchType="any", key=None, value=None):
        """
        Filters the input list of dictionaries based on the input parameters.

        If searchType is "any", all dictionary values are searched for the input
        value. Otherwise, the supplied key is used.
        """
        import re

        if not isinstance(dictionaries, list):
            dictionaries = []
        if not isinstance(elements, list):
            elements = []

        filteredDictionaries = []
        otherDictionaries = []
        filteredElements = []
        otherElements = []
        filteredIndices = []
        otherIndices = []

        def _match(candidate, query, st):
            if st in ["equal to", "equals", "=", "=="]:
                return Dictionary._ValuesMatch(candidate, query, caseSensitive=False)
            if st in ["not equal to", "not equals", "!="]:
                return not Dictionary._ValuesMatch(candidate, query, caseSensitive=False)

            c = str(candidate).lower()
            q = str(query).replace("*", ".+").lower()

            if st == "contains":
                return q in c
            if st == "starts with":
                return c.startswith(q)
            if st == "ends with":
                return c.endswith(q)
            if st == "does not contain":
                return q not in c
            if st == "matches":
                try:
                    return re.search(q, c) is not None
                except Exception:
                    return False
            if st == "any":
                return q in c
            return False

        st = str(searchType or "any").strip().lower()

        for i, aDictionary in enumerate(dictionaries):
            py = Dictionary._ToPythonDictionary(aDictionary, copy=True)
            if py is None:
                continue

            if value == "" or value is None:
                searchResult = True
            elif st == "any" or key in [None, ""]:
                searchResult = any(_match(v, value, "any") for v in py.values())
            else:
                if key not in py:
                    searchResult = False
                else:
                    searchResult = _match(py.get(key), value, st)

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

        return {
            "filteredDictionaries": filteredDictionaries,
            "otherDictionaries": otherDictionaries,
            "filteredIndices": filteredIndices,
            "otherIndices": otherIndices,
            "filteredElements": filteredElements,
            "otherElements": otherElements,
        }

    @staticmethod
    def Intersection(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the intersection of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None or dictionaryB is None:
            if not silent:
                print("Dictionary.Intersection - Warning: One or both of the input dictionaries is None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)

        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)

        if pyA is None:
            if not silent:
                print("Dictionary.Intersection - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if pyB is None:
            if not silent:
                print("Dictionary.Intersection - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        common = {k: pyA[k] for k in pyA.keys() if k in pyB}
        if not common:
            return Dictionary.ByKeysValues([], [], silent=silent)

        # Preserve previous merge semantics for common keys: equal values stay
        # scalar; different values become a two-value list.
        merged = {}
        for k in common:
            a = pyA.get(k)
            b = pyB.get(k)
            merged[k] = a if a == b else [a, b]

        return Dictionary.ByPythonDictionary(merged, silent=silent)

    @staticmethod
    def Impose(dictionaryA, dictionaryB, silent: bool = False):
        """
        Imposes dictionaryB onto dictionaryA.
        """
        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        if pyA is None:
            if dictionaryA is None:
                if not silent:
                    print("Dictionary.Impose - Warning: The dictionaryA input parameter is None. Returning None.")
            elif not silent:
                print("Dictionary.Impose - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None

        if dictionaryB is None:
            if not silent:
                print("Dictionary.Impose - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return Dictionary.ByPythonDictionary(pyA, silent=silent)

        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)
        if pyB is None:
            if not silent:
                print("Dictionary.Impose - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        pyA.update(pyB)
        return Dictionary.ByPythonDictionary(pyA, silent=silent)

    @staticmethod
    def Imprint(dictionaryA, dictionaryB, silent: bool = False):
        """
        Imprints dictionaryB onto dictionaryA by replacing values only for keys
        already present in dictionaryA.
        """
        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        if pyA is None:
            if dictionaryA is None:
                if not silent:
                    print("Dictionary.Imprint - Warning: The dictionaryA input parameter is None. Returning None.")
            elif not silent:
                print("Dictionary.Imprint - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None

        if dictionaryB is None:
            if not silent:
                print("Dictionary.Imprint - Warning: The dictionaryB input parameter is None. Returning dictionaryA.")
            return Dictionary.ByPythonDictionary(pyA, silent=silent)

        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)
        if pyB is None:
            if not silent:
                print("Dictionary.Imprint - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        out = {k: (pyB[k] if k in pyB else pyA[k]) for k in pyA.keys()}
        return Dictionary.ByPythonDictionary(out, silent=silent)

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
    def KeysAtValue(dictionary, value, silent=False):
        """
        Returns all keys in the input dictionary whose value matches the input value.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.KeysAtValue - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning empty list.")
            return []
        return [k for k, v in py.items() if Dictionary._ValuesMatch(v, value)]

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
        One-hot encodes multiple categorical dictionary values in one pass.

        For Python dictionaries, TGraphs, and TGraph vertex/edge records, the
        original container is updated and returned. For Core/topologic
        dictionaries, a new Core/topologic dictionary is returned.
        See https://en.wikipedia.org/wiki/One-hot.
        """
        py = Dictionary._ToPythonDictionary(d, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.OneHotEncode - Error: Input is not a valid Dictionary.")
            return None

        if keys is None:
            if not silent:
                print("Dictionary.OneHotEncode - Error: keys is None.")
            return None

        if not isinstance(keys, list):
            keys = list(keys)

        if len(keys) == 0:
            return d

        if not isinstance(categoriesByKey, dict):
            if not silent:
                print("Dictionary.OneHotEncode - Error: categoriesByKey must be a dict mapping key -> categories.")
            return None

        values_by_key = {k: py.get(k, None) for k in keys}

        for k in keys:
            py.pop(k, None)

        for k in keys:
            cats = categoriesByKey.get(k, None)
            if not isinstance(cats, (list, tuple)) or len(cats) == 0:
                if not silent:
                    print(f"Dictionary.OneHotEncode - Warning: No categories provided for key '{k}'. Skipping.")
                continue

            value = values_by_key.get(k, None)
            for i, cat in enumerate(cats):
                py[f"{k}_{i}"] = 1 if value == cat else 0

        return Dictionary._SetPythonDictionary(d, py)

    @staticmethod
    def PythonDictionary(dictionary, silent: bool = False):
        """
        Returns the input dictionary-like object as a Python dictionary.

        Parameters
        ----------
        dictionary : dict, topologic_core.Dictionary, backend dictionary, topologicpy.TGraph, TGraph vertex record, or TGraph edge record
            The input dictionary or dictionary-bearing object.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict or None
            A Python dictionary copy.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is not None:
            if isinstance(dictionary, dict) and not silent:
                print("Dictionary.PythonDictionary - Warning: The input dictionary parameter is already a python dictionary. Returning a copy of that dictionary.")
            return py

        if not silent:
            print("Dictionary.PythonDictionary - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning None.")
        return None

    @staticmethod
    def RemoveKey(dictionary, key, silent: bool = False, caseSensitive: bool = False):
        """
        Removes the key and its associated value from the input dictionary.

        For Python dictionaries, TGraphs, and TGraph vertex/edge records, the
        original container is updated and returned. For Core/topologic
        dictionaries, a new Core/topologic dictionary is returned.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.RemoveKey - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.RemoveKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        remove_key = None
        for k in list(py.keys()):
            if caseSensitive:
                if k == key:
                    remove_key = k
                    break
            elif isinstance(k, str) and k.lower() == key.lower():
                remove_key = k
                break

        if remove_key is not None:
            py.pop(remove_key, None)

        return Dictionary._SetPythonDictionary(dictionary, py)

    @staticmethod
    def SetValueAtKey(dictionary, key, value, silent: bool = False):
        """
        Creates or updates a key/value pair in the input dictionary.

        For Python dictionaries, TGraphs, and TGraph vertex/edge records, the
        original container is updated and returned. For Core/topologic
        dictionaries, a new Core/topologic dictionary is returned.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning None.")
            return None
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.SetValueAtKey - Error: The input key parameter is not a valid string. Returning None.")
            return None

        py[key] = value
        return Dictionary._SetPythonDictionary(dictionary, py)

    @staticmethod
    def SetValuesAtKeys(dictionary, keys, values, silent: bool = False):
        """
        Creates or updates multiple key/value pairs in the input dictionary.

        For Python dictionaries, TGraphs, and TGraph vertex/edge records, the
        original container is updated and returned. For Core/topologic
        dictionaries, a new Core/topologic dictionary is returned.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input dictionary parameter is not a valid dictionary. Returning None.")
            return None
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not isinstance(values, list):
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input values parameter is not a valid list. Returning None.")
            return None
        if len(keys) != len(values):
            if not silent:
                print("Dictionary.SetValuesAtKeys - Error: The input keys and values parameters are not of equal length. Returning None.")
            return None

        for key, value in zip(keys, values):
            if not isinstance(key, str):
                key = str(key)
            py[key] = value

        return Dictionary._SetPythonDictionary(dictionary, py)
    
    @staticmethod
    def SymDif(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        return Dictionary.SymmetricDifference(dictionaryA, dictionaryB, silent=silent)

    @staticmethod
    def SymmetricDifference(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.SymmetricDifference - Warning: Both of the input dictionaries are None. Returning an empty dictionary.")
            return Dictionary.ByKeysValues([], [], silent=silent)

        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)

        if pyA is None and dictionaryA is not None:
            if not silent:
                print("Dictionary.SymmetricDifference - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if pyB is None and dictionaryB is not None:
            if not silent:
                print("Dictionary.SymmetricDifference - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        pyA = pyA or {}
        pyB = pyB or {}
        out = {}

        for k, v in pyA.items():
            if k not in pyB:
                out[k] = v
        for k, v in pyB.items():
            if k not in pyA:
                out[k] = v

        return Dictionary.ByPythonDictionary(out, silent=silent)

    @staticmethod
    def Union(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the union (merge) of dictionaryA and dictionaryB.
        """
        if dictionaryA is None and dictionaryB is None:
            if not silent:
                print("Dictionary.Union - Warning: Both of the input dictionaries are None. Returning None.")
            return None

        pyA = Dictionary._ToPythonDictionary(dictionaryA, copy=True)
        pyB = Dictionary._ToPythonDictionary(dictionaryB, copy=True)

        if pyA is None and dictionaryA is not None:
            if not silent:
                print("Dictionary.Union - Error: the dictionaryA input parameter is not a valid dictionary. Returning None.")
            return None
        if pyB is None and dictionaryB is not None:
            if not silent:
                print("Dictionary.Union - Error: the dictionaryB input parameter is not a valid dictionary. Returning None.")
            return None

        if pyA is None:
            return Dictionary.ByPythonDictionary(pyB or {}, silent=silent)
        if pyB is None:
            return Dictionary.ByPythonDictionary(pyA or {}, silent=silent)

        return Dictionary.ByMergedDictionaries(pyA, pyB, silent=silent)

    @staticmethod
    def ValueAtKey(dictionary, key, defaultValue=None, silent=False):
        """
        Returns the value at the input key in the input dictionary.

        If the dictionary or key is invalid, or the key does not exist,
        defaultValue is returned. If a key exists with value None, None is
        returned, matching Python dict.get semantics.
        """
        if not isinstance(key, str):
            if not silent:
                print("Dictionary.ValueAtKey - Error: The input key parameter is not a valid str. Returning defaultValue.")
            return defaultValue

        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.ValueAtKey - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning defaultValue.")
            return defaultValue

        if key not in py:
            return defaultValue
        return py.get(key)

    @staticmethod
    def Values(dictionary, silent: bool = False):
        """
        Returns the list of values in the input dictionary-like object.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.Values - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning None.")
            return None
        return list(py.values())

    @staticmethod
    def ValuesAtKeys(dictionary, keys, defaultValue=None, silent: bool = False):
        """
        Returns the list of values of the input list of keys in the input dictionary.
        """
        py = Dictionary._ToPythonDictionary(dictionary, copy=True)
        if py is None:
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input dictionary parameter is not a valid topologic, python, backend, TGraph, TGraph vertex, or TGraph edge dictionary. Returning None.")
            return None
        if not isinstance(keys, list):
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not all(isinstance(k, str) for k in keys):
            if not silent:
                print("Dictionary.ValuesAtKeys - Error: The input keys parameter contains invalid values. Returning None.")
            return None
        return [py[k] if k in py else defaultValue for k in keys]

    @staticmethod
    def XOR(dictionaryA, dictionaryB, silent: bool = False):
        """
        Returns the symmetric difference (XOR) of dictionaryA and dictionaryB, based on keys.
        """
        return Dictionary.SymmetricDifference(dictionaryA, dictionaryB, silent=silent)

