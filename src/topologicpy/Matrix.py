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

class Matrix:
    @staticmethod
    def Add(matA, matB):
        """
        Adds the two input matrices.
        
        Parameters
        ----------
        matA : list
            The first input matrix.
        matB : list
            The second input matrix.

        Returns
        -------
        list
            The matrix resulting from the addition of the two input matrices.

        """
        matC = []
        if not isinstance(matA, list):
            return None
        if not isinstance(matB, list):
            return None
        for i in range(len(matA)):
            tempRow = []
            for j in range(len(matB)):
                tempRow.append(matA[i][j] + matB[i][j])
            matC.append(tempRow)
        return matC

    @staticmethod
    def ByCoordinateSystems(source, target, mantissa: int = 6, silent: bool = False):
        """
        Calculates the 4x4 transformation matrix that maps the source coordinate system to the target coordinate system.
        An example of a coordinate system matrix is:
        source = [
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1]   # Z-axis
        ]
        
        Parameters
        ----------
        source : list
            The 4X3 matrix representing source coordinate system. The rows are in the order: Origin, X-Axis, Y-Axis, Z-Axis.
        target : list
            The 4X3 matrix representing target coordinate system. The rows are in the order: Origin, X-Axis, Y-Axis, Z-Axis.
        mantissa : int , optional
                The desired length of the mantissa. The default is 6.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The 4x4 transformation matrix.
        """
        import numpy as np

        if not isinstance(source, list):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The source input parameter is not a valid list. Returning None.")
            return None
        if not isinstance(target, list):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The taget input parameter is not a valid list. Returning None.")
            return None
        
        # Convert to numpy arrays
        source_matrix = np.array(source)
        target_matrix = np.array(target)

        if source_matrix.shape != (4, 3):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The source input parameter must be 4x3 matrix. Returning None.")
            return None
        if target_matrix.shape != (4, 3):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The target input parameter must be 4x3 matrix. Returning None.")
            return None
        
        # Convert input matrices to homogeneous transformations
        source_to_world = np.eye(4)
        source_to_world[:3, 0] = source_matrix[1, :] # X-axis
        source_to_world[:3, 1] = source_matrix[2, :] # Y-axis
        source_to_world[:3, 2] = source_matrix[3, :] # Z-axis
        source_to_world[:3, 3] = source_matrix[0, :] # Origin

        target_to_world = np.eye(4)
        target_to_world[:3, 0] = target_matrix[1, :] # X-axis
        target_to_world[:3, 1] = target_matrix[2, :] # Y-axis
        target_to_world[:3, 2] = target_matrix[3, :] # Z-axis
        target_to_world[:3, 3] = target_matrix[0, :] # Origin

        # Compute the world-to-source transformation (inverse of source_to_world)
        world_to_source = np.linalg.inv(source_to_world)

        # Compute the source-to-target transformation
        source_to_target = target_to_world @ world_to_source

        # Convert the result to a list and round values
        result_list = source_to_target.tolist()
        rounded_result = [[round(value, mantissa) for value in row] for row in result_list]

        return rounded_result

    @staticmethod
    def ByRotation(angleX=0, angleY=0, angleZ=0, order="xyz"):
        def rotateXMatrix(radians):
            c = math.cos(radians)
            s = math.sin(radians)
            return [[1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]]

        def rotateYMatrix(radians):
            c = math.cos(radians)
            s = math.sin(radians)
            return [[c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]]

        def rotateZMatrix(radians):
            c = math.cos(radians)
            s = math.sin(radians)
            return [[c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

        xMat = rotateXMatrix(math.radians(angleX))
        yMat = rotateYMatrix(math.radians(angleY))
        zMat = rotateZMatrix(math.radians(angleZ))

        if order.lower() == "xyz":
            return Matrix.Multiply(Matrix.Multiply(zMat, yMat), xMat)
        if order.lower() == "xzy":
            return Matrix.Multiply(Matrix.Multiply(yMat, zMat), xMat)
        if order.lower() == "yxz":
            return Matrix.Multiply(Matrix.Multiply(zMat, xMat), yMat)
        if order.lower() == "yzx":
            return Matrix.Multiply(Matrix.Multiply(xMat, zMat), yMat)
        if order.lower() == "zxy":
            return Matrix.Multiply(Matrix.Multiply(yMat, xMat), zMat)
        if order.lower() == "zyx":
            return Matrix.Multiply(Matrix.Multiply(xMat, yMat), zMat)

    
    @staticmethod
    def ByScaling(scaleX=1.0, scaleY=1.0, scaleZ=1.0):
        """
        Creates a 4x4 scaling matrix.

        Parameters
        ----------
        scaleX : float , optional
            The desired scaling factor along the X axis. The default is 1.
        scaleY : float , optional
            The desired scaling factor along the X axis. The default is 1.
        scaleZ : float , optional
            The desired scaling factor along the X axis. The default is 1.
        
        Returns
        -------
        list
            The created 4X4 scaling matrix.

        """
        return [[scaleX,0,0,0],
                [0,scaleY,0,0],
                [0,0,scaleZ,0],
                [0,0,0,1]]
    
    @staticmethod
    def ByTranslation(translateX=0, translateY=0, translateZ=0):
        """
        Creates a 4x4 translation matrix.

        Parameters
        ----------
        translateX : float , optional
            The desired translation distance along the X axis. The default is 0.
        translateY : float , optional
            The desired translation distance along the X axis. The default is 0.
        translateZ : float , optional
            The desired translation distance along the X axis. The default is 0.
        
        Returns
        -------
        list
            The created 4X4 translation matrix.

        """
        return [[1,0,0,translateX],
                [0,1,0,translateY],
                [0,0,1,translateZ],
                [0,0,0,1]]
    
    @staticmethod
    def ByVectors(vectorA: list, vectorB: list, orientationA: list = [1, 0, 0], orientationB: list = [1, 0, 0]):
        """
        Creates a rotation matrix that aligns vectorA with vectorB and adjusts orientationA to match orientationB.

        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector to align with.
        orientationA : list
            The orientation vector associated with vectorA.
        orientationB : list
            The orientation vector associated with vectorB.

        Returns
        -------
        list
            The 4x4 transformation matrix.
        """
        from topologicpy.Vector import Vector
        import numpy as np

        def to_numpy(vector):
            """Converts a list or array-like to a numpy array."""
            return np.array(vector, dtype=np.float64)

        # Normalize input vectors and convert them to numpy arrays
        vectorA = to_numpy(Vector.Normalize(vectorA))
        vectorB = to_numpy(Vector.Normalize(vectorB))
        orientationA = to_numpy(Vector.Normalize(orientationA))
        orientationB = to_numpy(Vector.Normalize(orientationB))

        # Step 1: Compute rotation matrix to align vectorA with vectorB
        axis = np.cross(vectorA, vectorB)
        angle = np.arccos(np.clip(np.dot(vectorA, vectorB), -1.0, 1.0))

        if np.isclose(angle, 0):  # Vectors are already aligned
            rotation_matrix_normal = np.eye(3)
        elif np.isclose(angle, np.pi):  # Vectors are anti-parallel
            # Choose a perpendicular axis for rotation
            axis = to_numpy([1, 0, 0]) if not np.isclose(vectorA[0], 0) else to_numpy([0, 1, 0])
            rotation_matrix_normal = (
                np.eye(3)
                - 2 * np.outer(vectorA, vectorA)  # Reflect through the plane perpendicular to vectorA
            )
        else:
            axis = axis / np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix_normal = (
                np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            )

        # Step 2: Rotate orientationA using the first rotation matrix
        rotated_orientationA = np.dot(rotation_matrix_normal, orientationA)

        # Step 3: Compute rotation to align rotated_orientationA with orientationB in the plane of vectorB
        projected_orientationA = rotated_orientationA - np.dot(rotated_orientationA, vectorB) * vectorB
        projected_orientationB = orientationB - np.dot(orientationB, vectorB) * vectorB

        if np.linalg.norm(projected_orientationA) < 1e-6 or np.linalg.norm(projected_orientationB) < 1e-6:
            # If either projected vector is near zero, skip secondary rotation
            rotation_matrix_orientation = np.eye(3)
        else:
            projected_orientationA = projected_orientationA / np.linalg.norm(projected_orientationA)
            projected_orientationB = projected_orientationB / np.linalg.norm(projected_orientationB)
            axis = np.cross(projected_orientationA, projected_orientationB)
            angle = np.arccos(np.clip(np.dot(projected_orientationA, projected_orientationB), -1.0, 1.0))
            if np.isclose(angle, 0):  # Already aligned
                rotation_matrix_orientation = np.eye(3)
            else:
                axis = axis / np.linalg.norm(axis)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                rotation_matrix_orientation = (
                    np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                )

        # Step 4: Combine the two rotation matrices
        rotation_matrix = np.dot(rotation_matrix_orientation, rotation_matrix_normal)

        # Convert to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        return transform_matrix.tolist()

    def ByVectors_old(vectorA: list, vectorB: list, orientationA: list = [1,0,0], orientationB: list = [1,0,0]):
        """
        Creates a rotation matrix that aligns vectorA with vectorB and adjusts orientationA to match orientationB.
        
        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector to align with.
        orientationA : list
            The orientation vector associated with vectorA.
        orientationB : list
            The orientation vector associated with vectorB.
        
        Returns
        -------
        list
            The 4x4 transformation matrix.
        """
        from topologicpy.Vector import Vector
        import numpy as np

        # Normalize input vectors
        vectorA = Vector.Normalize(vectorA)
        vectorB = Vector.Normalize(vectorB)
        orientationA = Vector.Normalize(orientationA)
        orientationB = Vector.Normalize(orientationB)

        # Step 1: Compute rotation matrix to align vectorA with vectorB
        axis = np.cross(vectorA, vectorB)
        angle = np.arccos(np.clip(np.dot(vectorA, vectorB), -1.0, 1.0))

        if np.isclose(angle, 0):  # Vectors are already aligned
            rotation_matrix_normal = np.eye(3)
        elif np.isclose(angle, np.pi):  # Vectors are anti-parallel
            # Choose a perpendicular axis for rotation
            axis = np.array([1, 0, 0]) if not np.isclose(vectorA[0], 0) else np.array([0, 1, 0])
            rotation_matrix_normal = (
                np.eye(3)
                - 2 * np.outer(vectorA, vectorA)  # Reflect through the plane perpendicular to vectorA
            )
        else:
            axis = Vector.Normalize(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix_normal = (
                np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            )

        # Step 2: Rotate orientationA using the first rotation matrix
        rotated_orientationA = np.dot(rotation_matrix_normal, orientationA)

        # Step 3: Compute rotation to align rotated_orientationA with orientationB in the plane of vectorB
        projected_orientationA = rotated_orientationA - np.dot(rotated_orientationA, vectorB) * vectorB
        projected_orientationB = orientationB - np.dot(orientationB, vectorB) * vectorB

        if np.linalg.norm(projected_orientationA) < 1e-6 or np.linalg.norm(projected_orientationB) < 1e-6:
            # If either projected vector is near zero, skip secondary rotation
            rotation_matrix_orientation = np.eye(3)
        else:
            projected_orientationA = Vector.Normalize(projected_orientationA)
            projected_orientationB = Vector.Normalize(projected_orientationB)
            axis = np.cross(projected_orientationA, projected_orientationB)
            angle = np.arccos(np.clip(np.dot(projected_orientationA, projected_orientationB), -1.0, 1.0))
            if np.isclose(angle, 0):  # Already aligned
                rotation_matrix_orientation = np.eye(3)
            else:
                axis = Vector.Normalize(axis)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2, 0, -axis[0]]],
                    [-axis[1], axis[0], 0]
                ])
                rotation_matrix_orientation = (
                    np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                )

        # Step 4: Combine the two rotation matrices
        rotation_matrix = np.dot(rotation_matrix_orientation, rotation_matrix_normal)

        # Convert to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        return transform_matrix.tolist()

    
    @staticmethod
    def EigenvaluesAndVectors(matrix, mantissa: int = 6, silent: bool = False):
        import numpy as np
        """
        Returns the eigenvalues and eigenvectors of the input matrix. See https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
        
        Parameters
        ----------
        matrix : list
            The input matrix. Assumed to be a laplacian matrix.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
            The list of eigenvalues and eigenvectors of the input matrix.

        """
        from topologicpy.Helper import Helper
        import numpy as np

        if not isinstance(matrix, list):
            if not silent:
                print("Matrix.Eigenvalues - Error: The input matrix parameter is not a valid matrix. Returning None.")
            return None
        
        np_matrix = np.array(matrix)
        if not isinstance(np_matrix, np.ndarray):
            if not silent:
                print("Matrix.Eigenvalues - Error: The input matrix parameter is not a valid matrix. Returning None.")
            return None
        
        # Square check
        if np_matrix.shape[0] != np_matrix.shape[1]:
            if not silent:
                print("Matrix.Eigenvalues - Error: The input matrix parameter is not a square matrix. Returning None.")
            return None
        
        # Symmetry check
        if not np.allclose(np_matrix, np_matrix.T):
            if not silent:
                print("Matrix.Eigenvalues - Error: The input matrix is not symmetric. Returning None.")
            return None
        
        # # Degree matrix
        # degree_matrix = np.diag(np_matrix.sum(axis=1))
        
        # # Laplacian matrix
        # laplacian_matrix = degree_matrix - np_matrix
        
        # Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(np_matrix)
        
        e_values = [round(x, mantissa) for x in list(np.sort(eigenvalues))]
        e_vectors = []
        for eigenvector in eigenvectors:
            e_vectors.append([round(x, mantissa) for x in eigenvector])
        e_vectors = Helper.Sort(e_vectors, list(eigenvalues))
        return e_values, e_vectors

    @staticmethod
    def Identity():
        """
        Creates a 4x4 identity translation matrix.

        Parameters
        ----------
        
        Returns
        -------
        list
            The created 4X4 identity matrix.

        """
        return [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]
    
    @staticmethod
    def Invert(matA, silent: bool = False):
        """
        Inverts the input matrix.

        Parameters
        ----------
        matA : list of list of float
            The input matrix.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list of list of float
            The resulting matrix after it has been inverted.

        """
        import numpy as np

        if not isinstance(matA, list):
            if not silent:
                print(matA, matA.__class__)
                print("1. Matrix.Invert - Error: The input matA parameter is not a valid 4X4 matrix. Returning None.")
            return None
        np_matrix = np.array(matA)
        if np_matrix.shape != (4, 4):
            if not silent:
                print("2. Matrix.Invert - Error: The input matA parameter is not a valid 4X4 matrix. Returning None.")
            return None
        
        # Check if the matrix is invertible
        if np.isclose(np.linalg.det(np_matrix), 0):
            if not silent:
                print("Matrix.Invert - Error: The input matA parameter is not invertible. Returning None.")
            return None
        
        # Invert the matrix
        inverted_matrix = np.linalg.inv(np_matrix)
        return inverted_matrix.tolist()
    
    @staticmethod
    def Multiply(matA, matB):
        """
        Multiplies two matrices (matA and matB). The first matrix (matA) is applied first in the transformation,
        followed by the second matrix (matB).

        Parameters
        ----------
        matA : list of list of float
            The first input matrix.
        matB : list of list of float
            The second input matrix.

        Returns
        -------
        list of list of float
            The resulting matrix after multiplication.

        """
        # Input validation
        # if not (isinstance(matA, list) and all(isinstance(row, list) for row in matA) and
        #         isinstance(matB, list) and all(isinstance(row, list) for row in matB)):
        #     raise ValueError("Both inputs must be 2D lists representing matrices.")
        
        # Check matrix dimension compatibility
        if len(matA[0]) != len(matB):
            raise ValueError("Number of columns in matA must equal the number of rows in matB.")

        # Dimensions of the resulting matrix
        rows_A, cols_A = len(matA), len(matA[0])
        rows_B, cols_B = len(matB), len(matB[0])
        result = [[0.0] * cols_B for _ in range(rows_A)]

        # Matrix multiplication
        for i in range(rows_A):
            for j in range(cols_B):
                result[i][j] = sum(matA[i][k] * matB[k][j] for k in range(cols_A))

        return result

    @staticmethod
    def Subtract(matA, matB):
        """
        Subtracts the two input matrices.
        
        Parameters
        ----------
        matA : list
            The first input matrix.
        matB : list
            The second input matrix.

        Returns
        -------
        list
            The matrix resulting from the subtraction of the second input matrix from the first input matrix.

        """
        if not isinstance(matA, list):
            return None
        if not isinstance(matB, list):
            return None
        matC = []
        for i in range(len(matA)):
            tempRow = []
            for j in range(len(matB)):
                tempRow.append(matA[i][j] - matB[i][j])
            matC.append(tempRow)
        return matC

    @staticmethod
    def Transpose(matrix):
        """
        Transposes the input matrix.
        
        Parameters
        ----------
        matrix : list
            The input matrix.

        Returns
        -------
        list
            The transposed matrix.

        """
        return [list(x) for x in zip(*matrix)]
