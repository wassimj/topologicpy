import numpy as np
import numpy.linalg as la
import math

class Vector(list):
    @staticmethod
    def Angle(vectorA, vectorB, mantissa=3):
        """
        Description
        ----------
        Returns the angle in degrees between the two input vectors

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 3.

        Returns
        -------
        float
            The angle in degrees between the two input vectors.

        """
        n_v1=la.norm(vectorA)
        n_v2=la.norm(vectorB)
        if (abs(np.log10(n_v1/n_v2)) > 10):
            vectorA = vectorA/n_v1
            vectorB = vectorB/n_v2
        cosang = np.dot(vectorA, vectorB)
        sinang = la.norm(np.cross(vectorA, vectorB))
        return round(math.degrees(np.arctan2(sinang, cosang)), mantissa)

    @staticmethod
    def Multiply(vector, magnitude, tolerance=0.0001):
        """
        Description
        ----------
        Multiplies the input vector by the input magnitude.

        Parameters
        ----------
        vector : list
            The input vector.
        magnitude : float
            The input magnitude.
        tolerance : float, optional
            the desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The created vector that multiplies the input vector by the input magnitude.

        """
        oldMag = 0
        for value in vector:
            oldMag += value ** 2
        oldMag = oldMag ** 0.5
        if oldMag < tolerance:
            return [0,0,0]
        newVector = []
        for i in range(len(vector)):
            newVector.append(vector[i] * magnitude / oldMag)
        return newVector

    @staticmethod
    def Magnitude(vector, mantissa=3):
        """
        Description
        -----------
        Returns the magnitude of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.
        mantissa : int
            The length of the desired mantissa. The default is 3.

        Returns
        -------
        float
            The magnitude of the input vector.
        """

        return math.round(np.linalg.norm(np.array(vector)), mantissa)

    @staticmethod
    def Normalize(vector):
        """
        Description
        -----------
        Returns a normalized vector of the input vector. A normalized vector has the same direction as the input vector, but its magnitude is 1.

        Parameters
        ----------
        vector : list
            The input vector.

        Returns
        -------
        list
            The normalized vector.
        """

        return vector / np.linalg.norm(vector)

    @staticmethod
    def Cross(vectorA, vectorB):
        """
        Description
        -----------
        Returns the cross product of the two input vectors.
        The cross product of two input vectors is a vector perpendicular to both input vectors.

        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.

        Returns
        -------
        list
            The cross product of the two input vectors.
        """

        return list(np.cross(vectorA, vectorB))

    @staticmethod
    def IsCollinear(vectorA, vectorB, tolerance=0.1):
        """
        Description
        -----------
        Returns True if the input vectors are collinear. Returns False otherwise.

        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.

        Returns
        -------
        bool
            Returns True if the input vectors are collinear. Returns False otherwise.
        """

        return Vector.Angle(vectorA, vectorB) < tolerance
