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

import math
import os
import warnings

try:
    import numpy as np
    import numpy.linalg as la
    from numpy import pi, arctan2, rad2deg
except:
    print("Vector - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        import numpy.linalg as la
        from numpy import pi, arctan2, rad2deg
        print("Vector - numpy library installed successfully.")
    except:
        warnings.warn("Vector - Error: Could not import numpy.")

class Vector(list):
    @staticmethod
    def Add(vectorA, vectorB):
        """
        Adds the two input vectors.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.

        Returns
        -------
        list
            The sum vector of the two input vectors.

        """
        return [a + b for a, b in zip(vectorA, vectorB)]
    
    @staticmethod
    def Angle(vectorA, vectorB, mantissa: int = 6):
        """
        Returns the angle in degrees between the two input vectors

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.

        Returns
        -------
        float
            The angle in degrees between the two input vectors.

        """
        if vectorA == None:
            print("Vector.Angle - Error: The input vectorA is None. Returning None.")
            return None
        if vectorB == None:
            print("Vector.Angle - Error: The input vectorB is None. Returning None.")
            return None
        n_v1=la.norm(vectorA)
        n_v2=la.norm(vectorB)
        if n_v1 == 0 or n_v2 == 0:
            # Handle the case where one or both vectors have a magnitude of zero.
            return 0.0
    
        if (abs(np.log10(n_v1/n_v2)) > 10):
            vectorA = vectorA/n_v1
            vectorB = vectorB/n_v2
        cosang = np.dot(vectorA, vectorB)
        sinang = la.norm(np.cross(vectorA, vectorB))
        return round(math.degrees(np.arctan2(sinang, cosang)), mantissa)
    
    @staticmethod
    def Average(vectors: list):
        """
        Returns the average vector of the input vectors.

        Parameters
        ----------
        vectors : list
            The input list of vectors.
        
        Returns
        -------
        list
            The average vector of the input list of vectors.
        
        """
        if not isinstance(vectors, list):
            print("Vector.Average - Error: The input vectors parameter is not a valid list. Returning None.")
            return None
        vectors = [vec for vec in vectors if isinstance(vec, list)]
        if len(vectors) < 1:
            print("Vector.Average - Error: The input vectors parameter does not contain any valid vectors. Returning None.")
            return None
        
        dimensions = len(vectors[0])
        num_vectors = len(vectors)
        
        # Initialize a list to store the sum of each dimension
        sum_dimensions = [0] * dimensions
        
        # Calculate the sum of each dimension across all vectors
        for vector in vectors:
            for i in range(dimensions):
                sum_dimensions[i] += vector[i]
        
        # Calculate the average for each dimension
        average_dimensions = [sum_dim / num_vectors for sum_dim in sum_dimensions]
    
        return average_dimensions
    
    @staticmethod
    def AzimuthAltitude(vector, mantissa: int = 6):
        """
        Returns a dictionary of azimuth and altitude angles in degrees for the input vector. North is assumed to be the positive Y axis [0, 1, 0]. Up is assumed to be the positive Z axis [0, 0, 1].
        Azimuth is calculated in a counter-clockwise fashion from North where 0 is North, 90 is East, 180 is South, and 270 is West. Altitude is calculated in a counter-clockwise fashion where -90 is straight down (negative Z axis), 0 is in the XY plane, and 90 is straight up (positive Z axis).
        If the altitude is -90 or 90, the azimuth is assumed to be 0.

        Parameters
        ----------
        vectorA : list
            The input vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.

        Returns
        -------
        dict
            The dictionary containing the azimuth and altitude angles in degrees. The keys in the dictionary are 'azimuth' and 'altitude'. 

        """
        x, y, z = vector
        if x == 0 and y == 0:
            if z > 0:
                return {"azimuth":0, "altitude":90}
            elif z < 0:
                return {"azimuth":0, "altitude":-90}
            else:
                # undefined
                return None
        else:
            azimuth = math.degrees(math.atan2(y, x))
            if azimuth > 90:
                azimuth -= 360
            azimuth = round(90-azimuth, mantissa)
            xy_distance = math.sqrt(x**2 + y**2)
            altitude = math.degrees(math.atan2(z, xy_distance))
            altitude = round(altitude, mantissa)
            return {"azimuth":azimuth, "altitude":altitude}
    
    @staticmethod
    def Bisect(vectorA, vectorB):
        """
        Compute the bisecting vector of two input vectors.
        
        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.
            
        Returns
        -------
        dict
            The bisecting vector.
        
        """
        import numpy as np

        # Ensure vectors are numpy arrays
        vector1 = np.array(vectorA)
        vector2 = np.array(vectorB)
        
        # Normalize input vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        # Check if the angle between vectors is either 0 or 180 degrees
        dot_product = np.dot(vector1_norm, vector2_norm)
        if np.isclose(dot_product, 1.0) or np.isclose(dot_product, -1.0):
            print("Vector.Bisect - Warning: The two vectors are collinear and thus the bisecting vector is not well-defined.")
            # Angle is either 0 or 180 degrees, return any perpendicular vector
            bisecting_vector = np.array([vector1[1] - vector1[2], vector1[2] - vector1[0], vector1[0] - vector1[1]])
            bisecting_vector /= np.linalg.norm(bisecting_vector)
        else:
            # Compute bisecting vector
            bisecting_vector = (vector1_norm + vector2_norm) / np.linalg.norm(vector1_norm + vector2_norm)
        return list(bisecting_vector)
    
    @staticmethod
    def ByAzimuthAltitude(azimuth: float, altitude: float, north: float = 0, reverse: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the vector specified by the input azimuth and altitude angles.

        Parameters
        ----------
        azimuth : float
            The input azimuth angle in degrees. The angle is computed in an anti-clockwise fashion. 0 is considered North, 90 East, 180 is South, 270 is West
        altitude : float
            The input altitude angle in degrees from the XY plane. Positive is above the XY plane. Negative is below the XY plane
        north : float , optional
            The angle of the north direction in degrees measured from positive Y-axis. The angle is added in anti-clockwise fashion. 0 is considered along the positive Y-axis,
            90 is along the positive X-axis, 180 is along the negative Y-axis, and 270 along the negative Y-axis.
        reverse : bool , optional
            If set to True the direction of the vector is computed from the end point towards the origin. Otherwise, it is computed from the origin towards the end point.
        mantissa : int , optional
            The desired mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        list
            The resulting vector.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        
        e = Edge.ByVertices([Vertex.Origin(), Vertex.ByCoordinates(0, 1, 0)], tolerance=tolerance)
        e = Topology.Rotate(e, origin=Vertex.Origin(), axis=[1, 0, 0], angle=altitude)
        e = Topology.Rotate(e, origin=Vertex.Origin(), axis=[0, 0, 1], angle=-azimuth-north)
        if reverse:
            return Vector.Reverse(Edge.Direction(e))
        return Edge.Direction(e, mantissa=mantissa)
    
    @staticmethod
    def ByCoordinates(x, y, z):
        """
        Creates a vector by the specified x, y, z inputs.

        Parameters
        ----------
        x : float
            The X coordinate.
        y : float
            The Y coordinate.
        z : float
            The Z coodinate.

        Returns
        -------
        list
            The created vector.

        """
        return [x, y, z]
    
    @staticmethod
    def ByVertices(*vertices, normalize: bool = True, mantissa: int = 6, silent: bool = False):
        """
        Creates a vector by the specified input list of vertices.

        Parameters
        ----------
        *vertices : list
            The the input list of topologic vertices. The first element in the list is considered the start vertex. The last element in the list is considered the end vertex.
        normalize : bool , optional
            If set to True, the resulting vector is normalized (i.e. its length is set to 1)
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        list
            The created vector.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        import inspect

        if len(vertices) == 0:
            if not silent:
                print("Vector.ByVertices - Error: The input vertices parameter is an empty list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if len(vertices) == 1:
            vertices = vertices[0]
            if isinstance(vertices, list):
                if len(vertices) == 0:
                    if not silent:
                        print("Vector.ByVertices - Error: The input vertices parameter is an empty list. Returning None.")
                        curframe = inspect.currentframe()
                        calframe = inspect.getouterframes(curframe, 2)
                        print('caller name:', calframe[1][3])
                    return None
                else:
                    vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
                    if len(vertexList) == 0:
                        if not silent:
                            print("Vector.ByVertices - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
                            curframe = inspect.currentframe()
                            calframe = inspect.getouterframes(curframe, 2)
                            print('caller name:', calframe[1][3])
                        return None
            else:
                if not silent:
                    print("Vector.ByVertices - Warning: The input vertices parameter contains only one vertex. Returning None.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                return None
        else:
            vertexList = Helper.Flatten(list(vertices))
            vertexList = [x for x in vertexList if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) == 0:
            if not silent:
                print("Vector.ByVertices - Error: The input parameters do not contain any valid vertices. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        if len(vertexList) < 2:
            if not silent:
                print("Vector.ByVertices - Error: The input parameters do not contain a minimum of two valid vertices. Returning None.")
            return None
        v1 = vertexList[0]
        v2 = vertexList[-1]
        vector = [Vertex.X(v2, mantissa=mantissa)-Vertex.X(v1, mantissa=mantissa), Vertex.Y(v2, mantissa=mantissa)-Vertex.Y(v1, mantissa=mantissa), Vertex.Z(v2, mantissa=mantissa)-Vertex.Z(v1, mantissa=mantissa)]
        if normalize:
            vector = Vector.Normalize(vector)
        return vector

    @staticmethod
    def CompassAngle(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the horizontal compass angle in degrees between the two input vectors. The angle is measured in clockwise fashion.
        0 is along the positive Y-axis, 90 is along the positive X axis.
        Only the first two elements in the input vectors are considered.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        float
            The horizontal compass angle in degrees between the two input vectors.

        """
        if not isinstance(vectorA, list):
            if not silent:
                print("Vector.Coordinates - Error: The input vectorA parameter is not a valid vector. Returning Nonne.")
            return None
        if not isinstance(vectorB, list):
            if not silent:
                print("Vector.Coordinates - Error: The input vectorB parameter is not a valid vector. Returning Nonne.")
            return None
        if abs(vectorA[0]) <= tolerance and abs(vectorA[1]) <= tolerance:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorA parameter is vertical in the Z Axis. Returning Nonne.")
            return None
        if abs(vectorB[0]) <= tolerance and abs(vectorB[1]) <= tolerance:
            if not silent:
                print("Vector.CompassAngle - Error: The input vectorB parameter is vertical in the Z Axis. Returning Nonne.")
            return None
        p1 = (vectorA[0], vectorA[1])
        p2 = (vectorB[0], vectorB[1])
        ang1 = arctan2(*p1[::-1])
        ang2 = arctan2(*p2[::-1])
        return round(rad2deg((ang1 - ang2) % (2 * pi)), mantissa)

    @staticmethod
    def CompassDirection(vector, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the compass direction of the input direction.
        The possible returned values are:
        - North, East, South, West
        - Northeast, Southeast, Southwest, Northwest
        - A combination of the above (e.g. Up_Noertheast)
        - Up, Down
        - Origin

        Parameters
        ----------
        vector : list
            The input vector.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        str
            The compass direction of the input vector. The possible values are:
            - North, East, South, West
            - Northeast, Southeast, Southwest, Northwest
            - A combination of the above (e.g. Up_Noertheast)
            - Up, Down
            - Origin

        """
        import math

        if not isinstance(vector, list):
            if not silent:
                print("Vector.CompassDirection - Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        if len(vector) != 3:
            if not silent:
                print("Vector.CompassDirection - Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        
        x, y, z = vector
        
        # Handle the origin
        if abs(x) <= tolerance and abs(y) <= tolerance and abs(z) <= tolerance:
            return "Origin"
        
        # Normalize vector to prevent magnitude bias
        magnitude = math.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / magnitude, y / magnitude, z / magnitude
        
        # Apply tolerance to components
        x = 0 if abs(x) <= tolerance else x
        y = 0 if abs(y) <= tolerance else y
        z = 0 if abs(z) <= tolerance else z
        
        # Compass-based direction in the XY-plane
        if x == 0 and y > 0:
            horizontal_dir = "North"
        elif x == 0 and y < 0:
            horizontal_dir = "South"
        elif y == 0 and x > 0:
            horizontal_dir = "East"
        elif y == 0 and x < 0:
            horizontal_dir = "West"
        elif x > 0 and y > 0:
            horizontal_dir = "Northeast"
        elif x < 0 and y > 0:
            horizontal_dir = "Northwest"
        elif x < 0 and y < 0:
            horizontal_dir = "Southwest"
        elif x > 0 and y < 0:
            horizontal_dir = "Southeast"
        else:
            horizontal_dir = ""
        
        # Add vertical direction
        if z > 0:
            if horizontal_dir:
                return f"Up_{horizontal_dir}"
            return "Up"
        elif z < 0:
            if horizontal_dir:
                return f"Down_{horizontal_dir}"
            return "Down"
        else:
            return horizontal_dir
    
    @staticmethod
    def CompassDirections():
        """
        Returns the list of allowed compass directions.

        Parameters
        ----------

        Returns
        -------
        list
            The list of compass directions. These are:
            - ["Origin", "Up", "Down",
              "North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest",
              "Up_North", "Up_Northeast", "Up_East", "Up_Southeast", "Up_South", "Up_Southwest", "Up_West", "Up_Northwest",
               "Down_North", "Down_Northeast", "Down_East", "Down_Southeast", "Down_South", "Down_Southwest", "Down_West", "Down_Northwest",
              ]

        """
        return ["Origin", "Up", "Down",
                "North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest",
                "Up_North", "Up_Northeast", "Up_East", "Up_Southeast", "Up_South", "Up_Southwest", "Up_West", "Up_Northwest",
                "Down_North", "Down_Northeast", "Down_East", "Down_Southeast", "Down_South", "Down_Southwest", "Down_West", "Down_Northwest"
                ]
    
    @staticmethod
    def Coordinates(vector, outputType="xyz", mantissa: int = 6, silent: bool = False):
        """
        Returns the coordinates of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.
        outputType : string, optional
            The desired output type. Could be any permutation or substring of "xyz" or the string "matrix". The default is "xyz". The input is case insensitive and the coordinates will be returned in the specified order.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The coordinates of the input vertex.

        """
        if not isinstance(vector, list):
            if not silent:
                print("Vector.Coordinates - Error: The input vector parameter is not a valid vector. Returning None.")
            return None
        x = round(vector[0], mantissa)
        y = round(vector[1], mantissa)
        z = round(vector[2], mantissa)
        matrix = [[1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]]
        output = []
        outputType = outputType.lower()
        if outputType == "matrix":
            return matrix
        else:
            outputType = list(outputType)
            for axis in outputType:
                if axis == "x":
                    output.append(x)
                elif axis == "y":
                    output.append(y)
                elif axis == "z":
                    output.append(z)
        return output
    
    @staticmethod
    def Cross(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the cross product of the two input vectors. The resulting vector is perpendicular to the plane defined by the two input vectors.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.
        tolerance : float, optional
            the desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The vector representing the cross product of the two input vectors.

        """
        if not isinstance(vectorA, list):
            if not silent:
                print("Vector.Cross - Error: The input vectorA parameter is not a valid vector. Returning None.")
            return None
        if not isinstance(vectorB, list):
            if not silent:
                print("Vector.Cross - Error: The input vectorB parameter is not a valid vector. Returning None.")
            return None
        if Vector.Magnitude(vector=vectorA, mantissa=mantissa) <= tolerance:
            if not silent:
                print("Vector.Cross - Error: The magnitude of the input vectorA parameter is less than the input tolerance parameter. Returning None.")
            return None
        if Vector.Magnitude(vector=vectorB, mantissa=mantissa) <= tolerance:
            if not silent:
                print("Vector.Cross - Error: The magnitude of the input vectorB parameter is less than the input tolerance parameter. Returning None.")
            return None
        vecA = np.array(vectorA)
        vecB = np.array(vectorB)
        vecC = list(np.cross(vecA, vecB))
        if Vector.Magnitude(vecC) <= tolerance:
            return [0, 0, 0]
        return [round(vecC[0], mantissa), round(vecC[1], mantissa), round(vecC[2], mantissa)]

    @staticmethod
    def Dot(vectorA, vectorB, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the dot product of the two input vectors which is a measure of how much they are aligned.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.
        tolerance : float, optional
            the desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The vector representing the cross product of the two input vectors.

        """
        if not isinstance(vectorA, list):
            if not silent:
                print("Vector.Dot - Error: The input vectorA parameter is not a valid vector. Returning None.")
            return None
        if not isinstance(vectorB, list):
            if not silent:
                print("Vector.Dot - Error: The input vectorB parameter is not a valid vector. Returning None.")
            return None
        if Vector.Magnitude(vector=vectorA, mantissa=mantissa) <= tolerance:
            if not silent:
                print("Vector.Dot - Error: The magnitude of the input vectorA parameter is less than the input tolerance parameter. Returning None.")
            return None
        if Vector.Magnitude(vector=vectorB, mantissa=mantissa) <= tolerance:
            if not silent:
                print("Vector.Dot - Error: The magnitude of the input vectorB parameter is less than the input tolerance parameter. Returning None.")
            return None
        return round(sum(a*b for a, b in zip(vectorA, vectorB)), mantissa)
    
    @staticmethod
    def Down():
        """
        Returns the vector representing the *down* direction. In Topologic, the negative ZAxis direction is considered *down* ([0, 0, -1]).

        Returns
        -------
        list
            The vector representing the *down* direction.
        
        """
        return [0, 0, -1]
    
    @staticmethod
    def East():
        """
        Returns the vector representing the *east* direction. In Topologic, the positive XAxis direction is considered *east* ([1, 0, 0]).

        Returns
        -------
        list
            The vector representing the *east* direction.
        
        """
        return [1, 0, 0]
    
    @staticmethod
    def IsAntiParallel(vectorA, vectorB):
        """
        Returns True if the input vectors are anti-parallel. Returns False otherwise.
        
        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.
            
        Returns
        -------
        bool
            True if the input vectors are anti-parallel. False otherwise.
        
        """
        import numpy as np

        # Ensure vectors are numpy arrays
        vector1 = np.array(vectorA)
        vector2 = np.array(vectorB)
        
        # Normalize input vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        
        # Check if the angle between vectors is either 0 or 180 degrees
        dot_product = np.dot(vector1_norm, vector2_norm)
        if np.isclose(dot_product, -1.0):
            return True
        else:
            # Compute bisecting vector
            return False
    
    @staticmethod
    def IsCollinear(vectorA, vectorB, tolerance=0.0001):
        """
        Returns True if the input vectors are collinear (parallel or anti-parallel) 
        within a given tolerance. Returns False otherwise.

        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.
        tolerance : float, optional
            The desired tolerance for determining collinearity. The default is 0.0001.
            
        Returns
        -------
        bool
            True if the input vectors are collinear (parallel or anti-parallel). 
            False otherwise.

        """
        import numpy as np

        # Ensure vectors are numpy arrays
        vector1 = np.array(vectorA)
        vector2 = np.array(vectorB)
        
        # Normalize input vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        
        # Check if the dot product is within the tolerance of 1.0 or -1.0
        dot_product = np.dot(vector1_norm, vector2_norm)
        if (1.0 - tolerance) <= dot_product <= (1.0 + tolerance) or \
        (-1.0 - tolerance) <= dot_product <= (-1.0 + tolerance):
            return True
        else:
            return False
    
    @staticmethod
    def IsParallel(vectorA, vectorB, tolerance=0.0001):
        """
        Returns True if the input vectors are parallel within a given tolerance.
        Returns False otherwise.
        
        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.
        tolerance : float, optional
            The desired tolerance for determining parallelism. The default is 0.0001.
            
        Returns
        -------
        bool
            True if the input vectors are parallel. False otherwise.
        """
        import numpy as np

        # Ensure vectors are numpy arrays
        vector1 = np.array(vectorA)
        vector2 = np.array(vectorB)
        
        # Normalize input vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        
        # Check if the dot product is within the tolerance of 1.0
        dot_product = np.dot(vector1_norm, vector2_norm)
        if (1.0 - tolerance) <= dot_product <= (1.0 + tolerance):
            return True
        else:
            return False

    @staticmethod
    def IsSame(vectorA, vectorB, tolerance=0.0001):
        """
        Returns True if the input vectors are the same. Returns False otherwise.
        
        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
            
        Returns
        -------
        bool
            True if the input vectors are the same. False otherwise.
        
        """
        return all(abs(a - b) <= tolerance for a, b in zip(vectorA, vectorB))

    @staticmethod
    def Length(vector, mantissa: int = 6):
        """
        Returns the length of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.
        mantissa : int
            The length of the desired mantissa. The default is 6.

        Returns
        -------
        float
            The length of the input vector.
        
        """
        return Vector.Magnitude(vector, mantissa = mantissa)
    
    @staticmethod
    def Magnitude(vector, mantissa: int = 6):
        """
        Returns the magnitude of the input vector.

        Parameters
        ----------
        vector : list
            The input vector.
        mantissa : int
            The length of the desired mantissa. The default is 6.

        Returns
        -------
        float
            The magnitude of the input vector.
        
        """
        return round(np.linalg.norm(np.array(vector)), mantissa)

    @staticmethod
    def Multiply(vector, magnitude, tolerance=0.0001):
        """
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
        if abs(magnitude) <= tolerance:
            return [0.0] * len(vector)
        scaled_vector = [component * (magnitude) for component in vector]
        return scaled_vector

    @staticmethod
    def Normalize(vector):
        """
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
        return list(vector / np.linalg.norm(vector))

    @staticmethod
    def North():
        """
        Returns the vector representing the *north* direction. In Topologic, the positive YAxis direction is considered *north* ([0, 1, 0]).

        Returns
        -------
        list
            The vector representing the *north* direction.
        
        """
        return [0, 1, 0]
    
    @staticmethod
    def NorthEast():
        """
        Returns the vector representing the *northeast* direction. In Topologic, the positive YAxis direction is considered *north* and the positive XAxis direction is considered *east*. Therefore *northeast* is ([1, 1, 0]).

        Returns
        -------
        list
            The vector representing the *northeast* direction.
        
        """
        return [1, 1, 0]
    
    @staticmethod
    def NorthWest():
        """
        Returns the vector representing the *northwest* direction. In Topologic, the positive YAxis direction is considered *north* and the negative XAxis direction is considered *west*. Therefore *northwest* is ([-1, 1, 0]).

        Returns
        -------
        list
            The vector representing the *northwest* direction.
        
        """
        return [-1, 1, 0]
    
    @staticmethod
    def Reverse(vector):
        """
        Returns a reverse vector of the input vector. A reverse vector multiplies all components by -1.

        Parameters
        ----------
        vector : list
            The input vector.

        Returns
        -------
        list
            The normalized vector.
        
        """
        if not isinstance(vector, list):
            return None
        return [x*-1 for x in vector]
    
    @staticmethod
    def SetMagnitude(vector: list, magnitude: float) -> list:
        """
        Sets the magnitude of the input vector to the input magnitude.

        Parameters
        ----------
        vector : list
            The input vector.
        magnitude : float
            The desired magnitude.

        Returns
        -------
        list
            The created vector.
        
        """
        return (Vector.Multiply(vector=Vector.Normalize(vector), magnitude=magnitude))
    
    @staticmethod
    def South():
        """
        Returns the vector representing the *south* direction. In Topologic, the negative YAxis direction is considered *south* ([0, -1, 0]).

        Returns
        -------
        list
            The vector representing the *south* direction.
        
        """
        return [0, -1, 0]
    
    @staticmethod
    def SouthEast():
        """
        Returns the vector representing the *southeast* direction. In Topologic, the negative YAxis direction is considered *south* and the positive XAxis direction is considered *east*. Therefore *southeast* is ([1, -1, 0]).

        Returns
        -------
        list
            The vector representing the *southeast* direction.
        
        """
        return [1, -1, 0]
    
    @staticmethod
    def SouthWest():
        """
        Returns the vector representing the *southwest* direction. In Topologic, the negative YAxis direction is considered *south* and the negative XAxis direction is considered *west*. Therefore *southwest* is ([-1, -1, 0]).

        Returns
        -------
        list
            The vector representing the *southwest* direction.
        
        """
        return [-1, -1, 0]
    
    @staticmethod
    def Subtract(vectorA, vectorB):
        """
        Subtracts the second input vector from the first input vector.

        Parameters
        ----------
        vectorA : list
            The first vector.
        vectorB : list
            The second vector.

        Returns
        -------
        list
            The vector resulting from subtracting the second input vector from the first input vector.

        """
        return [a - b for a, b in zip(vectorA, vectorB)]
    
    @staticmethod
    def Sum(vectors: list):
        """
        Returns the sum vector of the input vectors.

        Parameters
        ----------
        vectors : list
            The input list of vectors.
        
        Returns
        -------
        list
            The sum vector of the input list of vectors.
        
        """
        if not isinstance(vectors, list):
            print("Vector.Average - Error: The input vectors parameter is not a valid list. Returning None.")
            return None
        vectors = [vec for vec in vectors if isinstance(vec, list)]
        if len(vectors) < 1:
            print("Vector.Average - Error: The input vectors parameter does not contain any valid vectors. Returning None.")
            return None
        
        dimensions = len(vectors[0])
        num_vectors = len(vectors)
        
        # Initialize a list to store the sum of each dimension
        sum_dimensions = [0] * dimensions
        
        # Calculate the sum of each dimension across all vectors
        for vector in vectors:
            for i in range(dimensions):
                sum_dimensions[i] += vector[i]
    
        return sum_dimensions
    
    @staticmethod
    def TransformationMatrix(vectorA, vectorB):
        """
        Returns the transformation matrix needed to align vectorA with vectorB.

        Parameters
        ----------
        vectorA : list
            The input vector to be transformed.
        vectorB : list
            The desired vector with which to align vectorA.
        
        Returns
        -------
        list
            Transformation matrix that follows the Blender software convention (nested list)
        
        """
        import numpy as np
        from topologicpy.Matrix import Matrix

        def transformation_matrix(vec1, vec2, translation_vector=None):
            """
            Compute a 4x4 transformation matrix that aligns vec1 to vec2.
            
            :param vec1: A 3D "source" vector
            :param vec2: A 3D "destination" vector
            :param translation_vector: Optional translation vector (default is None)
            :return: The 4x4 transformation matrix
            """
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            dot_product = np.dot(vec1, vec2)
            
            if np.isclose(dot_product, 1.0):
                # Vectors are parallel; return the identity matrix
                return np.eye(4)
            elif np.isclose(dot_product, -1.0):
                # Vectors are antiparallel; reflect one of the vectors about the origin
                reflection_matrix = np.eye(4)
                reflection_matrix[2, 2] = -1
                return reflection_matrix
            
            cross_product = np.cross(vec1, vec2)
            
            skew_symmetric_matrix = np.array([[0, -cross_product[2], cross_product[1], 0],
                                            [cross_product[2], 0, -cross_product[0], 0],
                                            [-cross_product[1], cross_product[0], 0, 0],
                                            [0, 0, 0, 1]])
            
            rotation_matrix = np.eye(4) + skew_symmetric_matrix + \
                            np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * \
                            (1 / (1 + dot_product))
            
            if translation_vector is not None:
                translation_matrix = np.eye(4)
                translation_matrix[:3, 3] = translation_vector
                transformation_matrix = np.dot(translation_matrix, rotation_matrix)
            else:
                transformation_matrix = rotation_matrix
            
            return transformation_matrix
        tran_mat = transformation_matrix(vectorA, vectorB)
        
        return [list(tran_mat[0]), list(tran_mat[1]), list(tran_mat[2]), list(tran_mat[3])]
    
    @staticmethod
    def Up():
        """
        Returns the vector representing the up direction. In Topologic, the positive ZAxis direction is considered "up" ([0, 0, 1]).

        Returns
        -------
        list
            The vector representing the "up" direction.
        
        """
        return [0, 0, 1]
    
    @staticmethod
    def West():
        """
        Returns the vector representing the *west* direction. In Topologic, the negative XAxis direction is considered *west* ([-1, 0, 0]).

        Returns
        -------
        list
            The vector representing the *west* direction.
        
        """
        return [-1, 0, 0]
    
    @staticmethod
    def XAxis():
        """
        Returns the vector representing the XAxis ([1, 0, 0])

        Returns
        -------
        list
            The vector representing the XAxis.
        
        """
        return [1, 0, 0]

    @staticmethod
    def YAxis():
        """
        Returns the vector representing the YAxis ([0, 1, 0])

        Returns
        -------
        list
            The vector representing the YAxis.
        
        """
        return [0, 1, 0]
    
    @staticmethod
    def ZAxis():
        """
        Returns the vector representing the ZAxis ([0, 0, 1])

        Returns
        -------
        list
            The vector representing the ZAxis.
        
        """
        return [0, 0, 1]
