from topologicpy import topologic
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
