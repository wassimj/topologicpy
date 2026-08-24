# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numbers
from collections import deque
from typing import Any, Iterable, List

try:
    import topologicpy
except Exception:  # pragma: no cover - defensive for isolated source inspection.
    topologicpy = None

try:  # Optional compatibility aliases. Do not install packages at import time.
    import numpy as np
    import numpy.linalg as la
except Exception:  # pragma: no cover - numpy is not required by Helper itself.
    np = None
    la = None


class Helper:
    """General-purpose helper methods used throughout TopologicPy."""

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _Mantissa(mantissa: int = 6) -> int:
        try:
            return max(0, int(mantissa))
        except Exception:
            return 6

    @staticmethod
    def _Tolerance(tolerance: float = 0.0001) -> float:
        try:
            tolerance = float(tolerance)
            if tolerance < 0:
                tolerance = abs(tolerance)
            return tolerance
        except Exception:
            return 0.0001

    @staticmethod
    def _IsNumber(value) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool)

    @staticmethod
    def _NumericList(values, silent: bool = False, caller: str = "Helper"):
        if not isinstance(values, list):
            if not silent:
                print(f"{caller} - Error: The input listA parameter is not a valid list. Returning None.")
            return None
        out = []
        for value in values:
            if not Helper._IsNumber(value):
                if not silent:
                    print(f"{caller} - Error: The input listA parameter contains a non-numeric value. Returning None.")
                return None
            out.append(float(value))
        return out

    @staticmethod
    def _Hashable(value):
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    @staticmethod
    def _DictionaryKeys(dictionary):
        try:
            from topologicpy.Dictionary import Dictionary
            keys = Dictionary.Keys(dictionary)
            return keys if isinstance(keys, list) else []
        except Exception:
            if isinstance(dictionary, dict):
                return list(dictionary.keys())
            return []

    @staticmethod
    def _DictionaryValue(dictionary, key, default=None):
        try:
            from topologicpy.Dictionary import Dictionary
            return Dictionary.ValueAtKey(dictionary, key)
        except Exception:
            if isinstance(dictionary, dict):
                return dictionary.get(key, default)
            return default

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    @staticmethod
    def BinAndAverage(listA, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Groups numbers within a specified tolerance into bins, calculates the average
        of each bin, and returns a sorted list of these averages.

        Parameters
        ----------
        listA : list
            The input list.
        mantissa : int , optional
            The desired length of the mantissa. Default is 6
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The sorted list of bin averages.

        """
        values = Helper._NumericList(listA, silent=silent, caller="Helper.BinAndAverage")
        if values is None:
            return None
        if len(values) < 1:
            if not silent:
                print("Helper.BinAndAverage - Error: The input listA parameter is an empty list. Returning None.")
            return None
        mantissa = Helper._Mantissa(mantissa)
        tolerance = Helper._Tolerance(tolerance)
        if len(values) == 1:
            return [round(values[0], mantissa)]

        values = sorted(values)
        bins = []
        current_bin = [values[0]]
        for num in values[1:]:
            if abs(num - current_bin[-1]) <= tolerance:
                current_bin.append(num)
            else:
                bins.append(current_bin)
                current_bin = [num]
        bins.append(current_bin)
        return sorted([round(sum(bin_values) / len(bin_values), mantissa) for bin_values in bins])

    @staticmethod
    def CheckVersion(library: str = None, version: str = None, silent: bool = False):
        """
        Compares an input version with the latest version of a Python library
        available on PyPI.

        Parameters
        ----------
        library : str, optional
            The input Python library name. Default is None.
        version : str, optional
            The input software version number to compare. Default is None.
        silent : bool, optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        str
            A message indicating whether the input version is older than, equal
            to, or newer than the latest version available on PyPI.
            Returns None if the version check cannot be completed.
        """
        if not isinstance(library, str) or not library.strip():
            if not silent:
                print(
                    "Helper.CheckVersion - Error: The input library parameter "
                    "is not valid. Returning None."
                )
            return None

        if version is None or not str(version).strip():
            if not silent:
                print(
                    "Helper.CheckVersion - Error: The input version parameter "
                    "is not valid. Returning None."
                )
            return None

        try:
            import requests
            from packaging import version as packaging_version
        except Exception:
            if not silent:
                print(
                    "Helper.CheckVersion - Error: Could not import the required "
                    "libraries. Returning None."
                )
            return None

        library = library.strip()
        current_version_string = str(version).strip()

        try:
            response = requests.get(
                f"https://pypi.org/pypi/{library}/json",
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            latest_version_string = data["info"]["version"]

            if not latest_version_string:
                raise ValueError("PyPI returned an empty version.")

        except Exception:
            if not silent:
                print(
                    "Helper.CheckVersion - Error: Could not fetch data from "
                    "PyPI. Returning None."
                )
            return None

        try:
            current_version = packaging_version.parse(current_version_string)
            latest_version = packaging_version.parse(
                str(latest_version_string)
            )
        except Exception:
            if not silent:
                print(
                    "Helper.CheckVersion - Error: Could not parse the version "
                    "numbers. Returning None."
                )
            return None

        if current_version < latest_version:
            return (
                f"The version that you are using ({current_version_string}) is "
                f"OLDER than the latest version ({latest_version_string}) "
                f"available on PyPI. Please consider upgrading to the latest "
                f"version."
            )

        if current_version == latest_version:
            return (
                f"The version that you are using ({current_version_string}) is "
                f"EQUAL TO the latest version ({latest_version_string}) "
                f"available on PyPI."
            )

        return (
            f"The version that you are using ({current_version_string}) is "
            f"NEWER than the latest version ({latest_version_string}) "
            f"available on PyPI."
        )

    @staticmethod
    def ClosestMatch(item, listA):
        """
        Returns the index of the closest match in the input list to the input item.
        This works for lists made out of numeric or string values.

        Parameters
        ----------
        item : int, float, or str
            The input item.
        listA : list
            The input list.

        Returns
        -------
        int
            The index of the best match in listA for the input item.

        """
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

        if not isinstance(listA, list) or len(listA) == 0:
            print("Helper.ClosestMatch - Error: The input listA parameter is not a valid non-empty list. Returning None.")
            return None

        if isinstance(item, str):
            candidates = [(i, value) for i, value in enumerate(listA) if isinstance(value, str)]
            if len(candidates) < 1:
                return None
            return min(candidates, key=lambda pair: levenshtein_distance(item, pair[1]))[0]

        if Helper._IsNumber(item) or isinstance(item, bool):
            candidates = [(i, value) for i, value in enumerate(listA) if isinstance(value, numbers.Real)]
            if len(candidates) < 1:
                return None
            return min(candidates, key=lambda pair: abs(float(pair[1]) - float(item)))[0]

        return None

    @staticmethod
    def ClusterByKeys(elements, dictionaries, *keys, silent=False):
        """
        Clusters the input list of elements and dictionaries based on the input key or keys.

        Parameters
        ----------
        elements : list
            The input list of elements to be clustered.
        dictionaries : list[Topology.Dictionary]
            The input list of dictionaries to be consulted for clustering. This is assumed to be in the same order as the list of elements.
        keys : str or list or comma-separated str input parameters
            The key or keys in the topology's dictionary to use for clustering.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary containing the elements and the dictionaries, but clustered. The dictionary has two keys:
            "elements": list
                A nested list of elements where each item is a list of elements with the same key values.
            "dictionaries": list
                A nested list of dictionaries where each item is a list of dictionaries with the same key values.
        """
        keys_list = [x for x in Helper.Flatten(list(keys)) if isinstance(x, str) and x.strip()]
        if len(keys_list) == 0:
            if not silent:
                print("Helper.ClusterByKeys - Error: The input keys parameter does not contain any valid strings. Returning None.")
            return None
        if not isinstance(elements, list) or not isinstance(dictionaries, list):
            if not silent:
                print("Helper.ClusterByKeys - Error: The input elements or dictionaries parameter is not a valid list. Returning None.")
            return None
        if len(elements) != len(dictionaries):
            if not silent:
                print("Helper.ClusterByKeys - Error: The input elements parameter does not have the same length as the input dictionaries parameter. Returning None.")
            return None

        grouped = {}
        order = []
        for element, dictionary in zip(elements, dictionaries):
            if len(Helper._DictionaryKeys(dictionary)) < 1:
                signature = ("_NONE_",)
            else:
                values = []
                for key in keys_list:
                    value = Helper._DictionaryValue(dictionary, key, None)
                    if value is not None:
                        values.append(Helper._Hashable(value))
                signature = tuple(values) if len(values) > 0 else ("_NONE_",)
            if signature not in grouped:
                grouped[signature] = {"elements": [], "dictionaries": []}
                order.append(signature)
            grouped[signature]["elements"].append(element)
            grouped[signature]["dictionaries"].append(dictionary)

        return {
            "elements": [grouped[key]["elements"] for key in order],
            "dictionaries": [grouped[key]["dictionaries"] for key in order],
        }

    @staticmethod
    def Flatten(listA):
        """
        Flattens the input nested list.

        Parameters
        ----------
        listA : list
            The input nested list.

        Returns
        -------
        list
            The flattened list.

        """
        if not isinstance(listA, list):
            return [listA]
        flat_list = []
        for item in listA:
            flat_list.extend(Helper.Flatten(item))
        return flat_list

    @staticmethod
    def Grow(seed_idx, group_size, adjacency, visited_global):
        """
        Attempts to grow a spatially connected group of a specified size starting from a given seed index.

        This method uses a breadth-first search strategy to explore neighboring indices from the seed index,
        guided by the provided adjacency dictionary. It avoids reusing indices that are globally visited.
        The growth continues until the desired group size is reached or no further expansion is possible.

        Parameters
        ----------
        seed_idx : int
            The index from which to start growing the group.
        group_size : int
            The target size of the group to be grown.
        adjacency : dict
            A dictionary mapping each index to a list of adjacent indices. This defines the connectivity.
        visited_global : set
            A set of indices that have already been used in previously grown groups and should be avoided.

        Returns
        -------
        list[int] or None
            A list of indices representing a connected group of the specified size if successful, otherwise None.

        Notes
        -----
        This method is intended for internal use in functions that generate connected subgroups
        of spatial elements (e.g., cells) based on adjacency. The result may vary between runs due to random shuffling
        of neighbor order to diversify outputs.
        """
        import random

        try:
            group_size = int(group_size)
        except Exception:
            return None
        if group_size < 1 or not isinstance(adjacency, dict):
            return None
        if visited_global is None:
            visited_global = set()
        try:
            visited_global = set(visited_global)
        except Exception:
            visited_global = set()
        if seed_idx in visited_global:
            return None

        group = [seed_idx]
        visited = {seed_idx}
        queue = deque([seed_idx])
        while queue and len(group) < group_size:
            current = queue.popleft()
            neighbors = list(adjacency.get(current, []) or [])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in visited_global:
                    group.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(group) >= group_size:
                        break
        return group if len(group) == group_size else None

    @staticmethod
    def Iterate(listA):
        """
        Iterates the input nested list so that each sublist has the same number of members. To fill extra members, the shorter lists are iterated from their first member.
        For example Iterate([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3, 1, 2], ['m', 'n', 'o', 'p', 'm'], ['a', 'b', 'c', 'd', 'e']]

        Parameters
        ----------
        listA : list
            The input nested list.

        Returns
        -------
        list
            The iterated list.

        """
        if not isinstance(listA, list):
            return None
        sublists = [list(item) for item in listA if isinstance(item, list)]
        if len(sublists) < 1:
            return None
        max_length = max((len(item) for item in sublists), default=0)
        out = []
        for sublist in sublists:
            if max_length == 0:
                out.append([])
            elif len(sublist) == 0:
                out.append([None] * max_length)
            else:
                out.append([sublist[i % len(sublist)] for i in range(max_length)])
        return out

    @staticmethod
    def MakeUnique(listA):
        """
        Forces the strings in the input list to be unique if they have duplicates.

        Parameters
        ----------
        listA : list
            The input list of strings.

        Returns
        -------
        list
            The input list, but with each item ensured to be unique if they have duplicates.

        """
        if not isinstance(listA, list):
            return None
        counts = {}
        unique_strings = []
        for item in listA:
            string = str(item)
            if string in counts:
                counts[string] += 1
                unique_strings.append(f"{string}_{counts[string]}")
            else:
                counts[string] = 0
                unique_strings.append(string)
        return unique_strings

    @staticmethod
    def MergeByThreshold(listA, threshold=0.0001):
        """
        Merges the numbers in the input list so that numbers within the input threshold are averaged into one number.

        Parameters
        ----------
        listA : list
            The input nested list.
        threshold : float , optional
            The desired merge threshold value. Default is 0.0001.

        Returns
        -------
        list
            The merged list. The list is sorted in ascending numeric order.

        """
        values = Helper._NumericList(listA, silent=True, caller="Helper.MergeByThreshold")
        if values is None:
            return None
        if len(values) == 0:
            return []
        threshold = Helper._Tolerance(threshold)
        values = sorted(values)
        bins = [[values[0]]]
        for value in values[1:]:
            if value - bins[-1][-1] <= threshold:
                bins[-1].append(value)
            else:
                bins.append([value])
        return [sum(bin_values) / len(bin_values) for bin_values in bins]

    @staticmethod
    def MaximumIndices(listA, silent: bool = False):
        """
        Returns a list of indices of the maximum value in the input list.
        For example, if the input list is [7,3,4,7,5,7] then the returned list is [0,3,5] to indicate the indices of the maximum value (7).

        Parameters
        ----------
        listA : list
            The input list.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The resulting list.

        """
        if not isinstance(listA, list):
            if not silent:
                print("Helper.MaximumIndices - Error: The input listA parameter is not a valid list. Returning None.")
            return None
        if len(listA) == 0:
            return []
        try:
            max_value = max(listA)
        except Exception:
            if not silent:
                print("Helper.MaximumIndices - Error: Could not evaluate the maximum value. Returning None.")
            return None
        return [i for i, value in enumerate(listA) if value == max_value]

    @staticmethod
    def MinimumIndices(listA, silent: bool = False):
        """
        Returns a list of indices of the minimum value in the input list.
        For example, if the input list is [1,3,4,1,5,1] then the returned list is [0,3,5] to indicate the indices of the minimum value (1).

        Parameters
        ----------
        listA : list
            The input list.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The resulting list.

        """
        if not isinstance(listA, list):
            if not silent:
                print("Helper.MinimumIndices - Error: The input listA parameter is not a valid list. Returning None.")
            return None
        if len(listA) == 0:
            return []
        try:
            min_value = min(listA)
        except Exception:
            if not silent:
                print("Helper.MinimumIndices - Error: Could not evaluate the minimum value. Returning None.")
            return None
        return [i for i, value in enumerate(listA) if value == min_value]

    @staticmethod
    def Normalize(listA, mantissa: int = 6):
        """
        Normalizes the input list so that it is in the range 0 to 1

        Parameters
        ----------
        listA : list
            The input nested list.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        list
            The normalized list.

        """
        if not isinstance(listA, list):
            print("Helper.Normalize - Error: The input list is not valid. Returning None.")
            return None
        values = [float(x) for x in listA if Helper._IsNumber(x)]
        if len(values) < 1:
            print("Helper.Normalize - Error: The input list does not contain numeric values. Returning None.")
            return None
        mantissa = Helper._Mantissa(mantissa)
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return [0 for _ in values]
        return [round((x - min_val) / (max_val - min_val), mantissa) for x in values]

    @staticmethod
    def Position(item, listA):
        """
        Returns the position of the item in the list or the position it would have been inserts.
        item is assumed to be numeric. listA is assumed to contain only numeric values and sorted from lowest to highest value.

        Parameters
        ----------
        item : int or float
            The input number to be positioned.
        listA : list
            The input sorted list.

        Returns
        -------
        int
            The position of the item within the list.

        """
        if not Helper._IsNumber(item) or not isinstance(listA, list):
            return None
        left = 0
        right = len(listA) - 1
        while left <= right:
            mid = (left + right) // 2
            if listA[mid] == item:
                return mid
            if listA[mid] < item:
                left = mid + 1
            else:
                right = mid - 1
        return left

    @staticmethod
    def RemoveEven(listA):
        """
        Removes the even indexed members of the input list.

        Parameters
        ----------
        listA : list
            The input list.

        Returns
        -------
        list
            The resulting list.

        """
        if not isinstance(listA, list):
            return None
        return [listA[i] for i in range(1, len(listA), 2)]

    @staticmethod
    def RemoveOdd(listA):
        """
        Removes the odd indexed members of the input list.

        Parameters
        ----------
        listA : list
            The input list.

        Returns
        -------
        list
            The resulting list.

        """
        if not isinstance(listA, list):
            return None
        return [listA[i] for i in range(0, len(listA), 2)]

    @staticmethod
    def Repeat(listA):
        """
        Repeats the input nested list so that each sublist has the same number of members. To fill extra members, the last item in the shorter lists are repeated and appended.
        For example Iterate([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3, 3, 3], ['m', 'n', 'o', 'p', 'p'], ['a', 'b', 'c', 'd', 'e']]

        Parameters
        ----------
        listA : list
            The input nested list.

        Returns
        -------
        list
            The repeated list.

        """
        if not isinstance(listA, list):
            return None
        sublists = [list(item) for item in listA if isinstance(item, list)]
        if len(sublists) < 1:
            return None
        max_length = max((len(item) for item in sublists), default=0)
        out = []
        for sublist in sublists:
            if len(sublist) == 0:
                out.append([None] * max_length)
            else:
                out.append(sublist + [sublist[-1]] * (max_length - len(sublist)))
        return out

    @staticmethod
    def Sort(listA, *otherLists, reverseFlags=None, silent: bool = False):
        """
        Sorts the first input list according to the values in the subsequent input lists in order. For example,
        your first list can be a list of topologies and the next set of lists can be their volume, surface area, and z level.
        The list of topologies will then be sorted first by volume, then by surface, and lastly by z level. You can choose
        to reverse the order of sorting by including a list of TRUE/FALSE values in the reverseFlags input parameter.
        For example, if you wish to sort the volume in reverse order (from large to small), but sort the other parameters
        normally, you would include the following list for reverseFlag: [True, False, False].

        Parameters
        ----------
        listA : list
            The first input list to be sorts
        *otherLists : any number of lists to use for sorting listA, optional.
            Any number of lists that are used to sort the listA input parameter. The order of these input
            parameters determines the order of sorting (from left to right). If no lists are included, the input list will be sorted as is.
        reverseFlags : list, optional.
            The list of booleans (TRUE/FALSE) to indicated if sorting based on a particular list should be conducted in reverse order.
            The length of the reverseFlags list should match the number of the lists in the input otherLists parameter. If set to None,
            a default list of FALSE values is created to match the number of the lists in the input otherLists parameter. The default
            is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The sorted list.

        """
        if not isinstance(listA, list):
            if not silent:
                print("Helper.Sort - Error: The input listA parameter is not a valid list. Returning None.")
            return None
        if len(otherLists) < 1:
            try:
                return sorted(listA)
            except Exception:
                if not silent:
                    print("Helper.Sort - Error: Could not sort listA without sorting keys. Returning None.")
                return None
        if reverseFlags is None:
            reverseFlags = [False] * len(otherLists)
        if not isinstance(reverseFlags, list) or len(reverseFlags) != len(otherLists):
            if not silent:
                print("Helper.Sort - Error: The length of the reverseFlags input parameter is not equal to the number of input lists. Returning None.")
            return None
        for other in otherLists:
            if not isinstance(other, list) or len(other) != len(listA):
                if not silent:
                    print("Helper.Sort - Error: Each sorting list must be a list with the same length as listA. Returning None.")
                return None

        def sorted_unique(values):
            unique = []
            for value in values:
                if not any(value == existing for existing in unique):
                    unique.append(value)
            try:
                return sorted(unique)
            except Exception:
                return sorted(unique, key=lambda value: (type(value).__name__, repr(value)))

        ranks = []
        for values in otherLists:
            unique = sorted_unique(values)
            ranks.append([next(i for i, value in enumerate(unique) if item == value) for item in values])

        order = list(range(len(listA)))
        order.sort(key=lambda idx: tuple((-rank[idx] if reverseFlags[j] else rank[idx]) for j, rank in enumerate(ranks)))
        return [listA[i] for i in order]

    @staticmethod
    def TopPercentIndices(values: list, percent: float) -> list:
        """
        Returns the indices of the highest values corresponding to the specified percentage of the input list.

        Parameters
        ----------
        values : list
            The input list of numerical values.
        percent : float
            The percentage of highest values to return. This value should be between 0 and 100.

        Returns
        -------
        list
            The indices of the highest values in the input list, ordered from highest to lowest value.

        """
        import math

        if not isinstance(values, list):
            print("Helper.TopPercentIndices - Error: The input values parameter is not a valid list. Returning None.")
            return None

        if len(values) == 0:
            return []

        if not all(isinstance(value, (int, float)) for value in values):
            print("Helper.TopPercentIndices - Error: The input values list contains non-numerical values. Returning None.")
            return None

        if not isinstance(percent, (int, float)):
            print("Helper.TopPercentIndices - Error: The input percent parameter is not a valid number. Returning None.")
            return None

        if percent <= 0 or percent > 100:
            print("Helper.TopPercentIndices - Error: The input percent parameter must be greater than 0 and less than or equal to 100. Returning None.")
            return None

        count = math.ceil(len(values) * percent / 100.0)

        return sorted(
            range(len(values)),
            key=lambda i: values[i],
            reverse=True
        )[:count]

    @staticmethod
    def Transpose(listA):
        """
        Transposes the input list (swaps rows and columns).

        Parameters
        ----------
        listA : list
            The input list.

        Returns
        -------
        list
            The transposed list.

        """
        if not isinstance(listA, list):
            return None
        rows = [row for row in listA if isinstance(row, list)]
        if len(rows) != len(listA):
            return None
        if len(rows) == 0:
            return []
        min_length = min(len(row) for row in rows)
        return [[row[i] for row in rows] for i in range(min_length)]

    @staticmethod
    def Trim(listA):
        """
        Trims the input nested list so that each sublist has the same number of members. All lists are trimmed to match the length of the shortest list.
        For example Trim([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3], ['m', 'n', 'o'], ['a', 'b', 'c']]

        Parameters
        ----------
        listA : list
            The input nested list.

        Returns
        -------
        list
            The repeated list.

        """
        if not isinstance(listA, list):
            return None
        sublists = [item for item in listA if isinstance(item, list)]
        if len(sublists) != len(listA):
            return None
        if len(sublists) == 0:
            return []
        min_length = min(len(item) for item in sublists)
        return [item[:min_length] for item in sublists]

    @staticmethod
    def Version(check: bool = True, silent: bool = False):
        """
        Returns the current version of the software.

        Parameters
        ----------
        check : bool , optional
            if set to True, the version number is checked with the latest version on PyPi. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        str
            The current version of the software. Optionally, includes a check with PyPi.

        """
        result = getattr(topologicpy, "__version__", None) if topologicpy is not None else None
        if result is None:
            if not silent:
                print("Helper.Version - Error: Could not determine the current TopologicPy version. Returning None.")
            return None
        if check is True:
            return Helper.CheckVersion("topologicpy", result, silent=silent)
        return result
