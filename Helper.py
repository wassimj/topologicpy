import topologicpy
import numpy as np
import numpy.linalg as la
import math

class Helper:
    @staticmethod
    def Flatten(l):
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

        if not isinstance(l, list):
            return [l]
        flat_list = []
        for item in l:
            flat_list = flat_list + Helper.Flatten(item)
        return flat_list

    @staticmethod
    def Iterate(l):
        """
        Iterates the input nested list so that each sublist has the same number of members. To fill extra members, the shorter lists are iterated from their first member.
        For example Iterate([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3, 1, 2], ['m', 'n', 'o', 'p', 'm'], ['a', 'b', 'c', 'd', 'e']]

        Parameters
        ----------
        l : list
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

        maxLength = len(l[0])
        iterated_list = []
        for aSubList in l:
            newLength = len(aSubList)
            if newLength > maxLength:
                maxLength = newLength
        for anItem in l:
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
    def Normalize(l, mantissa=4):
        """
        Normalizes the input list so that it is in the range 0 to 1

        Parameters
        ----------
        l : list
            The input nested list.
        mantissa : int , optional
            The desired mantissa value. The default is 4.

        Returns
        -------
        list
            The normalized list.

        """
        if l == None:
            print("Helper.Normalize - Error: The input list is not valid. Returning None.")
            return None
        
        # Make sure the list is numeric
        l = [x for x in l if type(x) == int or type(x) == float]
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
    def Repeat(l):
        """
        Repeats the input nested list so that each sublist has the same number of members. To fill extra members, the last item in the shorter lists are repeated and appended.
        For example Iterate([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3, 3, 3], ['m', 'n', 'o', 'p', 'p'], ['a', 'b', 'c', 'd', 'e']]

        Parameters
        ----------
        l : list
            The input nested list.

        Returns
        -------
        list
            The repeated list.

        """
        if not isinstance(l, list):
            return None
        repeated_list = [x for x in l if isinstance(x, list)]
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
    def Sort(lA, lB):
        """
        Sorts the first input list according to the values in the second input list.

        Parameters
        ----------
        lA : list
            The first input list to be sorts
        lB : list
            The second input list to use for sorting the first input list.

        Returns
        -------
        list
            The sorted list.

        """
        lA.sort(key=dict(zip(lA, lB)).get)
        return lA

    @staticmethod
    def Transpose(l):
        """
        Transposes (swaps rows and columns) the input list.

        Parameters
        ----------
        l : list
            The input list.

        Returns
        -------
        list
            The transposed list.

        """
        if not isinstance(l, list):
            return None
        length = len(l[0])
        transposed_list = []
        for i in range(length):
            tempRow = []
            for j in range(len(l)):
                tempRow.append(l[j][i])
            transposed_list.append(tempRow)
        return transposed_list
    
    @staticmethod
    def Trim(l):
        """
        Trims the input nested list so that each sublist has the same number of members. All lists are trimmed to match the length of the shortest list.
        For example Trim([[1,2,3],['m','n','o','p'],['a','b','c','d','e']]) yields [[1, 2, 3], ['m', 'n', 'o'], ['a', 'b', 'c']]

        Parameters
        ----------
        l : list
            The input nested list.

        Returns
        -------
        list
            The repeated list.

        """
        minLength = len(l[0])
        returnList = []
        for aSubList in l:
            newLength = len(aSubList)
            if newLength < minLength:
                minLength = newLength
        for anItem in l:
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
