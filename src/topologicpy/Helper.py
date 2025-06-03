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

import topologicpy
import os
import warnings

try:
    import numpy as np
    import numpy.linalg as la
except:
    print("Helper - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        import numpy.linalg as la
        print("Helper - numpy library installed correctly.")
    except:
        warnings.warn("Helper - Error: Could not import numpy.")

class Helper:
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
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        list
            The sorted list of bin averages.

        """
        if not isinstance(listA, list):
            if not silent:
                print("Helper.BinAndAverage = Error: The input listA parameter is not a valid list. Returning None.")
            return None
        
        if len(listA) < 1:
            if not silent:
                print("Helper.BinAndAverage = Error: The input listA parameter is not an empty list. Returning None.")
            return None
        
        if len(listA) == 1:
            return listA
            
        # Sort numbers to facilitate grouping
        listA = sorted(listA)
        bins = []
        current_bin = [listA[0]]

        # Group numbers into bins based on tolerance
        for num in listA[1:]:
            if abs(num - current_bin[-1]) <= tolerance:
                current_bin.append(num)
            else:
                bins.append(current_bin)
                current_bin = [num]
        bins.append(current_bin)  # Add the last bin

        # Calculate averages of each bin
        bin_averages = [round(sum(bin) / len(bin), mantissa) for bin in bins]

        return sorted(bin_averages)
    
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
        import numbers
        import random
        import string
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]
       
        def generate_unlikely_string(length=16):
            characters = string.ascii_letters + string.digits + string.punctuation
            return ''.join(random.choice(characters) for _ in range(length))

        if not listA:
            print("Helper.ClosestMatch - Error: THe input listA parameter is not a valid list. Returning None.")
            return None  # Handle empty list case

        if isinstance(item, str):
            listA = [generate_unlikely_string(length=32) if not isinstance(x, str) else x for x in listA]
            # For string inputs, find the closest match using Levenshtein distance
            closest_index = min(range(len(listA)), key=lambda i: levenshtein_distance(item, listA[i]))
        else:
            listA = [float('-inf') if not isinstance(x, numbers.Real) else x for x in listA]
            # For numeric or boolean inputs, find the closest match based on absolute difference
            closest_index = min(range(len(listA)), key=lambda i: abs(listA[i] - item))

        return closest_index

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
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.


        Returns
        -------
        dict
            A dictionary containing the elements and the dictionaries, but clustered. The dictionary has two keys:
            "elements": list
                A nested list of elements where each item is a list of elements with the same key values.
            "dictionaries": list
                A nested list of dictionaries where each item is a list of dictionaries with the same key values.
        """
        
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import inspect
        keys_list = list(keys)
        keys_list = Helper.Flatten(keys_list)
        keys_list = [x for x in keys_list if isinstance(x, str)]

        if len(keys_list) == 0:
            if not silent:
                print("Helper.ClusterByKeys - Error: The input keys parameter is an empty list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        
        if len(keys_list) == 0:
            if not silent:
                print("Helper.ClusterByKeys - Error: The input keys parameter does not contain any valid strings. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not (len(elements) == len(dictionaries)):
            if not silent:
                print("Helper.ClusterByKeys - Error: The input elements parameter does not have the same length as the input dictionaries parameter. Returning None.")
            return None
        
        elements_clusters = []
        dict_clusters = []
        values = []
        new_dictionaries = []
        for i, d in enumerate(dictionaries):
            d_keys = Dictionary.Keys(dictionaries[i])
            if len(d_keys) > 0:
                values_list = []
                for key in keys_list:
                    v = Dictionary.ValueAtKey(d, key)
                    if not v == None:
                        values_list.append(v)
                values_list = str(values_list)
                dictionaries[i] = Dictionary.SetValueAtKey(dictionaries[i], "_clustering_key_", values_list)
                values.append(values_list)
            else:
                dictionaries[i] = Dictionary.SetValueAtKey(dictionaries[i], "_clustering_key_", "_NONE_")
                values.append("_NONE_")
            new_dictionaries.append(dictionaries[i])

        
        values = list(set(values))
        remaining_dictionaries = [x for x in new_dictionaries]
        remaining_elements = [x for x in elements]
        remaining_indices = [i for i, x in enumerate(elements)]

        if len(values) == 0:
            return {"elements": [elements], "dictionaries": [dictionaries]}
        for value in values:
            if len(remaining_dictionaries) == 0:
                break
            dict = Dictionary.Filter(remaining_elements, remaining_dictionaries, searchType="equal to", key="_clustering_key_", value=value)
            filtered_indices = dict['filteredIndices']
            final_dictionaries = []
            final_elements = []
            if len(filtered_indices) > 0:
                for filtered_index in filtered_indices:
                    filtered_dictionary = remaining_dictionaries[filtered_index]
                    filtered_dictionary = Dictionary.RemoveKey(filtered_dictionary, "_clustering_key_")
                    final_dictionaries.append(filtered_dictionary)
                    filtered_element = remaining_elements[filtered_index]
                    final_elements.append(filtered_element)
                dict_clusters.append(final_dictionaries)
                elements_clusters.append(final_elements)
            remaining_dictionaries = dict['otherDictionaries']
            remaining_elements = dict['otherElements']
            remaining_indices = dict['otherIndices']
        if len(remaining_elements) > 0:
            temp_dict_cluster = []
            temp_element_cluster = []
            for remaining_index in remaining_indices:
                temp_element_cluster.append(remaining_elements[remaining_index])
                temp_dict_cluster.append(remaining_dictionaries[remaining_index])
            if len(temp_element_cluster) > 0:
                dict_clusters.append(temp_dict_cluster)
                elements_clusters.append(temp_element_cluster)
        return {"elements": elements_clusters, "dictionaries": dict_clusters}

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
            flat_list = flat_list + Helper.Flatten(item)
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
        from collections import deque
        import random

        group = [seed_idx]
        visited = set(group)
        queue = deque([seed_idx])

        while queue and len(group) < group_size:
            current = queue.popleft()
            neighbors = adjacency.get(current, [])
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
        # From https://stackoverflow.com/questions/34432056/repeat-elements-of-list-between-each-other-until-we-reach-a-certain-length
        def onestep(cur,y,base):
            # one step of the iteration
            if cur is not None:
                y.append(cur)
                base.append(cur)
            else:
                y.append(base[0])  # append is simplest, for now
                base = base[1:]+[base[0]]  # rotate
            return base

        maxLength = len(listA[0])
        iterated_list = []
        for aSubList in listA:
            newLength = len(aSubList)
            if newLength > maxLength:
                maxLength = newLength
        for anItem in listA:
            for i in range(len(anItem), maxLength):
                anItem.append(None)
            y=[]
            base=[]
            for cur in anItem:
                base = onestep(cur,y,base)
            iterated_list.append(y)
        return iterated_list
    
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
        # Create a dictionary to store counts of each string
        counts = {}
        # Create a list to store modified strings
        unique_strings = []
        
        for string in listA:
            # If the string already exists in the counts dictionary
            if string in counts:
                # Increment the count
                counts[string] += 1
                # Append the modified string with underscore and count
                unique_strings.append(f"{string}_{counts[string]}")
            else:
                # If it's the first occurrence of the string, add it to the counts dictionary
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
            The desired merge threshold value. The default is 0.0001.

        Returns
        -------
        list
            The merged list. The list is sorted in ascending numeric order.

        """
        # Sort the list in ascending order
        listA.sort()
        merged_list = []

        # Initialize the first element in the merged list
        merged_list.append(listA[0])

        # Merge numbers within the threshold
        for i in range(1, len(listA)):
            if listA[i] - merged_list[-1] <= threshold:
                # Merge the current number with the last element in the merged list
                merged_list[-1] = (merged_list[-1] + listA[i]) / 2
            else:
                # If the current number is beyond the threshold, add it as a new element
                merged_list.append(listA[i])

        return merged_list
    
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
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        if len(listA) == 1:
            return [0]
    
        # Find the maximum value in the list
        max_value = min(listA)
        
        # Find all indices where this minimum value occurs
        indices = [i for i, value in enumerate(listA) if value == max_value]
        
        return indices
    
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
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        if len(listA) == 1:
            return [0]
    
        # Find the minimum value in the list
        min_value = min(listA)
        
        # Find all indices where this minimum value occurs
        indices = [i for i, value in enumerate(listA) if value == min_value]
        
        return indices
    
    @staticmethod
    def Normalize(listA, mantissa: int = 6):
        """
        Normalizes the input list so that it is in the range 0 to 1

        Parameters
        ----------
        listA : list
            The input nested list.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The normalized list.

        """
        if not isinstance(listA, list):
            print("Helper.Normalize - Error: The input list is not valid. Returning None.")
            return None
        
        # Make sure the list is numeric
        l = [x for x in listA if type(x) == int or type(x) == float]
        if len(l) < 1:
            print("Helper.Normalize - Error: The input list does not contain numeric values. Returning None.")
            return None
        min_val = min(l)
        max_val = max(l)
        if min_val == max_val:
            normalized_list = [0 for x in l]
        else:
            normalized_list = [round((x - min_val) / (max_val - min_val), mantissa) for x in l]
        return normalized_list

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
        left = 0
        right = len(listA) - 1

        while left <= right:
            mid = (left + right) // 2
            if listA[mid] == item:
                return mid
            elif listA[mid] < item:
                left = mid + 1
            else:
                right = mid - 1

        # If the target is not found, return the position where it would be inserted
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
        return_list = []
        for i in range(1, len(listA), 2):
            return_list.append(listA[i])
        return return_list
    
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
        return_list = []
        for i in range(0, len(listA), 2):
            return_list.append(listA[i])
        return return_list
    
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
        repeated_list = [x for x in listA if isinstance(x, list)]
        if len(repeated_list) < 1:
            return None
        maxLength = len(repeated_list[0])
        for aSubList in repeated_list:
            newLength = len(aSubList)
            if newLength > maxLength:
                maxLength = newLength
        for anItem in repeated_list:
            if (len(anItem) > 0):
                itemToAppend = anItem[-1]
            else:
                itemToAppend = None
            for i in range(len(anItem), maxLength):
                anItem.append(itemToAppend)
        return repeated_list

    @staticmethod
    def Sort(listA, *otherLists, reverseFlags=None):
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

        Returns
        -------
        list
            The sorted list.

        """
       
        # If reverseFlags is not provided, assume all lists should be sorted in ascending order
        if reverseFlags is None:
            reverseFlags = [False] * len(otherLists)
        if not isinstance(otherLists, tuple):
            print("Helper.Sort - Error: No other lists to use for sorting have been provided. Returning None.")
            return None
        if len(otherLists) < 1:
            print("Helper.Sort - Error: The otherLists input parameter does not contain any valid lists. Returning None.")
            return None
        if not len(reverseFlags) == len(otherLists):
            print("Helper.Sort - Error: The length of the reverseFlags input parameter is not equal to the number of input lists. Returning None.")
            return None
        # Convert other_lists to numeric and reverse if needed.
        sorting_lists = []
        for i, a_list in enumerate(otherLists):
            temp_list = []
            temp_set = list(set(a_list))
            temp_set = sorted(temp_set)
            if reverseFlags[i] == True:
                temp_set.reverse()
            for item in a_list:
                temp_list.append(temp_set.index(item))
            sorting_lists.append(temp_list)
    
        combined_lists = list(zip(listA, *sorting_lists))
        # Sort the combined list based on all the elements and reverse the lists as needed
        combined_lists.sort(key=lambda x: tuple((-val) if reverse else val for val, reverse in zip(x[1:], reverseFlags)))
        sorted_listA = [item[0] for item in combined_lists]
        return sorted_listA

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
        length = len(listA[0])
        transposed_list = []
        for i in range(length):
            tempRow = []
            for j in range(len(listA)):
                tempRow.append(listA[j][i])
            transposed_list.append(tempRow)
        return transposed_list
    
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
        minLength = len(listA[0])
        returnList = []
        for aSubList in listA:
            newLength = len(aSubList)
            if newLength < minLength:
                minLength = newLength
        for anItem in listA:
            anItem = anItem[:minLength]
            returnList.append(anItem)
        return returnList
    
    @staticmethod
    def Version():
        """
        Returns the current version of the software.

        Parameters
        ----------

        Returns
        -------
        str
            The current version of the software.

        """
        
        import requests
        from packaging import version

        def compare_version_with_pypi(library_name, input_version):
            """
            Compare an input version with the latest version of a Python library on PyPI.

            Args:
                library_name (str): The name of the Python library on PyPI.
                input_version (str): The version number to compare (e.g., "0.7.58").

            Returns:
                str: A message indicating whether the input version is less than,
                    equal to, or greater than the latest version on PyPI.
            """
            try:
                # Fetch library data from PyPI
                url = f"https://pypi.org/pypi/{library_name}/json"
                response = requests.get(url)
                response.raise_for_status()

                # Extract the latest version from the JSON response
                data = response.json()
                latest_version = data['info']['version']

                # Compare versions using the packaging library
                if version.parse(input_version) < version.parse(latest_version):
                    return (f"The version that you are using ({input_version}) is OLDER than the latest version {latest_version} from PyPI. Please consider upgrading to the latest version.")
                elif version.parse(input_version) == version.parse(latest_version):
                    return (f"The version that you are using ({input_version}) is the latest version available on PyPI.")
                else:
                    return (f"The version that you are using ({input_version}) is NEWER than the latest version ({latest_version}) available from PyPI.")

            except requests.exceptions.RequestException as e:
                return f"Error fetching data from PyPI: {e}"
            except KeyError:
                return "Error: Unable to find the latest version in the PyPI response."
        
        current_version = topologicpy.__version__
        result = compare_version_with_pypi("topologicpy", current_version)
        return result
