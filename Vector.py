import numpy as np
import numpy.linalg as la
from numpy import pi, arctan2, rad2deg
import math

class Vector(list):
    @staticmethod
    def Angle(vectorA, vectorB, mantissa=4):
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
            The length of the desired mantissa. The default is 4.

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
    def CompassAngle(vectorA, vectorB, mantissa=4, tolerance=0.0001):
        """
        Description
        ----------
        Returns the horizontal compass angle in degrees between the two input vectors. The angle is measured in counter-clockwise fashion. Only the first two elements in the input vectors are considered.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        float
            The horizontal compass angle in degrees between the two input vectors.

        """
        if abs(vectorA[0]) < tolerance and abs(vectorA[1]) < tolerance:
            return None
        if abs(vectorB[0]) < tolerance and abs(vectorB[1]) < tolerance:
            return None
        p1 = (vectorA[0], vectorA[1])
        p2 = (vectorB[0], vectorB[1])
        ang1 = arctan2(*p1[::-1])
        ang2 = arctan2(*p2[::-1])
        return round(rad2deg((ang1 - ang2) % (2 * pi)), mantissa)

    @staticmethod
    def Cross(vectorA, vectorB, mantissa=4, tolerance=0.0001):
        """
        Description
        ----------
        Returns the cross product of the two input vectors. The resulting vector is perpendicular to the plane defined by the two input vectors.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 4.
        tolerance : float, optional
            the desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The vector representing the cross product of the two input vectors.

        """
        if not isinstance(vectorA, list) or not isinstance(vectorB, list):
            return None
        if Vector.Magnitude(vector=vectorA, mantissa=mantissa) < tolerance or Vector.Magnitude(vector=vectorB, mantissa=mantissa) < tolerance:
            return None
        vecA = np.array(vectorA)
        vecB = np.array(vectorB)
        vecC = list(np.cross(vecA, vecB))
        if Vector.Magnitude(vecC) < tolerance:
            return None
        return [round(vecC[0], mantissa), round(vecC[1], mantissa), round(vecC[2], mantissa)]

    @staticmethod
    def Magnitude(vector, mantissa=4):
        """
        Description
        -----------
        Returns the magnitude of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.
        mantissa : int
            The length of the desired mantissa. The default is 4.

        Returns
        -------
        float
            The magnitude of the input vector.
        """

        return math.round(np.linalg.norm(np.array(vector)), mantissa)
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
