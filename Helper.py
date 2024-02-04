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

import topologicpy
import os

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
        raise Exception("Helper - Error: Could not import numpy.")

class Helper:
    @staticmethod
    def Flatten(listA):
        """
        Flattens the input nested list.

        Parameters
        ----------
        l : list
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
    def K_Means(data, k=4, maxIterations=100):
        import random
        def euclidean_distance(p, q):
            return sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5

        # Initialize k centroids randomly
        centroids = random.sample(data, k)

        for _ in range(maxIterations):
            # Assign each data point to the nearest centroid
            clusters = [[] for _ in range(k)]
            for point in data:
                distances = [euclidean_distance(point, centroid) for centroid in centroids]
                nearest_centroid_index = distances.index(min(distances))
                clusters[nearest_centroid_index].append(point)

            # Compute the new centroids as the mean of the points in each cluster
            new_centroids = []
            for cluster in clusters:
                if not cluster:
                    # If a cluster is empty, keep the previous centroid
                    new_centroids.append(centroids[clusters.index(cluster)])
                else:
                    new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])

            # Check if the centroids have converged
            if new_centroids == centroids:
                break

            centroids = new_centroids

        return {'clusters': clusters, 'centroids': centroids}
    
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
    def Normalize(listA, mantissa: int = 6):
        """
        Normalizes the input list so that it is in the range 0 to 1

        Parameters
        ----------
        listA : list
            The input nested list.
        mantissa : int , optional
            The desired mantissa value. The default is 4.

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
        None
            

        Returns
        -------
        str
            The current version of the software.

        """
        return topologicpy.__version__
