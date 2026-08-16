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

import math


class Matrix:
    """Utility methods for creating and manipulating transformation matrices."""

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _MatrixShape(matrix):
        if not isinstance(matrix, list) or len(matrix) < 1:
            return None
        if not all(isinstance(row, list) for row in matrix):
            return None
        if len(matrix[0]) < 1:
            return None
        cols = len(matrix[0])
        if any(len(row) != cols for row in matrix):
            return None
        return len(matrix), cols

    @staticmethod
    def _IsNumericMatrix(matrix):
        shape = Matrix._MatrixShape(matrix)
        if shape is None:
            return False
        try:
            for row in matrix:
                for value in row:
                    float(value)
            return True
        except Exception:
            return False

    @staticmethod
    def _RoundValue(value, mantissa=6):
        try:
            result = round(float(value), int(mantissa))
            if abs(result) < 10 ** (-max(int(mantissa), 0)):
                result = 0.0
            return result
        except Exception:
            return value

    @staticmethod
    def _RoundMatrix(matrix, mantissa=6):
        return [[Matrix._RoundValue(value, mantissa=mantissa) for value in row] for row in matrix]

    @staticmethod
    def _Vector3(vector):
        if not isinstance(vector, (list, tuple)) or len(vector) < 3:
            return None
        try:
            return [float(vector[0]), float(vector[1]), float(vector[2])]
        except Exception:
            return None

    @staticmethod
    def _Dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _Cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _Norm(v):
        return math.sqrt(Matrix._Dot(v, v))

    @staticmethod
    def _Normalize(v, tolerance=1e-12):
        if v is None:
            return None
        n = Matrix._Norm(v)
        if n <= tolerance:
            return None
        return [v[0] / n, v[1] / n, v[2] / n]

    @staticmethod
    def _Perpendicular(v):
        # Pick the global axis least aligned with v, then cross it with v.
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        axis = min(axes, key=lambda a: abs(Matrix._Dot(a, v)))
        p = Matrix._Cross(v, axis)
        return Matrix._Normalize(p) or [1.0, 0.0, 0.0]

    @staticmethod
    def _ProjectedOrientation(orientation, direction, tolerance=1e-12):
        o = Matrix._Vector3(orientation)
        if o is None:
            o = Matrix._Perpendicular(direction)
        projection = [
            o[0] - Matrix._Dot(o, direction) * direction[0],
            o[1] - Matrix._Dot(o, direction) * direction[1],
            o[2] - Matrix._Dot(o, direction) * direction[2],
        ]
        projection = Matrix._Normalize(projection, tolerance=tolerance)
        if projection is None:
            projection = Matrix._Perpendicular(direction)
        return projection

    @staticmethod
    def _Frame(direction, orientation, tolerance=1e-12):
        x_axis = Matrix._Normalize(Matrix._Vector3(direction), tolerance=tolerance)
        if x_axis is None:
            return None
        y_axis = Matrix._ProjectedOrientation(orientation, x_axis, tolerance=tolerance)
        z_axis = Matrix._Normalize(Matrix._Cross(x_axis, y_axis), tolerance=tolerance)
        if z_axis is None:
            return None
        # Re-orthogonalise y to avoid drift from non-orthogonal input orientation.
        y_axis = Matrix._Normalize(Matrix._Cross(z_axis, x_axis), tolerance=tolerance)
        if y_axis is None:
            return None
        return [x_axis, y_axis, z_axis]

    @staticmethod
    def _Mat3Transpose(matrix):
        return [[matrix[j][i] for j in range(3)] for i in range(3)]

    @staticmethod
    def _Mat3Multiply(a, b):
        return [
            [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

    @staticmethod
    def _ColumnsFromFrame(frame):
        return [
            [frame[0][0], frame[1][0], frame[2][0]],
            [frame[0][1], frame[1][1], frame[2][1]],
            [frame[0][2], frame[1][2], frame[2][2]],
        ]

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
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
        shape_a = Matrix._MatrixShape(matA)
        shape_b = Matrix._MatrixShape(matB)
        if shape_a is None or shape_b is None or shape_a != shape_b:
            return None
        try:
            return [
                [matA[i][j] + matB[i][j] for j in range(shape_a[1])]
                for i in range(shape_a[0])
            ]
        except Exception:
            return None

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
                The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The 4x4 transformation matrix.
        """
        try:
            import numpy as np
        except Exception as exc:
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: numpy is required. Returning None.")
                print("Error:", exc)
            return None

        if not isinstance(source, list):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The source input parameter is not a valid list. Returning None.")
            return None
        if not isinstance(target, list):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The target input parameter is not a valid list. Returning None.")
            return None
        try:
            source_matrix = np.array(source, dtype=float)
            target_matrix = np.array(target, dtype=float)
        except Exception:
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The input coordinate systems must contain numeric values. Returning None.")
            return None

        if source_matrix.shape != (4, 3):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The source input parameter must be 4x3 matrix. Returning None.")
            return None
        if target_matrix.shape != (4, 3):
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The target input parameter must be 4x3 matrix. Returning None.")
            return None

        source_to_world = np.eye(4)
        source_to_world[:3, 0] = source_matrix[1, :]
        source_to_world[:3, 1] = source_matrix[2, :]
        source_to_world[:3, 2] = source_matrix[3, :]
        source_to_world[:3, 3] = source_matrix[0, :]

        target_to_world = np.eye(4)
        target_to_world[:3, 0] = target_matrix[1, :]
        target_to_world[:3, 1] = target_matrix[2, :]
        target_to_world[:3, 2] = target_matrix[3, :]
        target_to_world[:3, 3] = target_matrix[0, :]

        try:
            world_to_source = np.linalg.inv(source_to_world)
        except Exception:
            if not silent:
                print("Matrix.ByCoordinateSystems - Error: The source coordinate system is singular. Returning None.")
            return None

        source_to_target = target_to_world @ world_to_source
        return Matrix._RoundMatrix(source_to_target.tolist(), mantissa=mantissa)

    @staticmethod
    def ByRotation(angleX=0, angleY=0, angleZ=0, order="xyz"):
        """
        Creates a 4x4 rotation matrix from X, Y, and Z rotation angles in degrees.
        """
        try:
            angleX = float(angleX)
            angleY = float(angleY)
            angleZ = float(angleZ)
        except Exception:
            return None

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
        order = str(order or "").lower()

        if order == "xyz":
            return Matrix.Multiply(Matrix.Multiply(zMat, yMat), xMat)
        if order == "xzy":
            return Matrix.Multiply(Matrix.Multiply(yMat, zMat), xMat)
        if order == "yxz":
            return Matrix.Multiply(Matrix.Multiply(zMat, xMat), yMat)
        if order == "yzx":
            return Matrix.Multiply(Matrix.Multiply(xMat, zMat), yMat)
        if order == "zxy":
            return Matrix.Multiply(Matrix.Multiply(yMat, xMat), zMat)
        if order == "zyx":
            return Matrix.Multiply(Matrix.Multiply(xMat, yMat), zMat)
        return None

    @staticmethod
    def ByScaling(scaleX=1.0, scaleY=1.0, scaleZ=1.0):
        """
        Creates a 4x4 scaling matrix.

        Parameters
        ----------
        scaleX : float , optional
            The desired scaling factor along the X axis. Default is 1.
        scaleY : float , optional
            The desired scaling factor along the Y axis. Default is 1.
        scaleZ : float , optional
            The desired scaling factor along the Z axis. Default is 1.
        
        Returns
        -------
        list
            The created 4X4 scaling matrix.

        """
        try:
            scaleX = float(scaleX)
            scaleY = float(scaleY)
            scaleZ = float(scaleZ)
        except Exception:
            return None
        return [[scaleX, 0, 0, 0],
                [0, scaleY, 0, 0],
                [0, 0, scaleZ, 0],
                [0, 0, 0, 1]]

    @staticmethod
    def ByTranslation(translateX=0, translateY=0, translateZ=0):
        """
        Creates a 4x4 translation matrix.

        Parameters
        ----------
        translateX : float , optional
            The desired translation distance along the X axis. Default is 0.
        translateY : float , optional
            The desired translation distance along the Y axis. Default is 0.
        translateZ : float , optional
            The desired translation distance along the Z axis. Default is 0.
        
        Returns
        -------
        list
            The created 4X4 translation matrix.

        """
        try:
            translateX = float(translateX)
            translateY = float(translateY)
            translateZ = float(translateZ)
        except Exception:
            return None
        return [[1, 0, 0, translateX],
                [0, 1, 0, translateY],
                [0, 0, 1, translateZ],
                [0, 0, 0, 1]]

    @staticmethod
    def ByVectors(vectorA: list, vectorB: list, orientationA: list = None, orientationB: list = None):
        """
        Creates a rotation matrix that aligns vectorA with vectorB and adjusts orientationA to match orientationB.

        Parameters
        ----------
        vectorA : list
            The first input vector.
        vectorB : list
            The second input vector to align with.
        orientationA : list , optional
            The orientation vector associated with vectorA. Default is [1, 0, 0].
        orientationB : list , optional
            The orientation vector associated with vectorB. Default is [1, 0, 0].

        Returns
        -------
        list
            The 4x4 transformation matrix.
        """
        orientationA = [1, 0, 0] if orientationA is None else orientationA
        orientationB = [1, 0, 0] if orientationB is None else orientationB

        source_frame = Matrix._Frame(vectorA, orientationA)
        target_frame = Matrix._Frame(vectorB, orientationB)
        if source_frame is None or target_frame is None:
            return None

        source_columns = Matrix._ColumnsFromFrame(source_frame)
        target_columns = Matrix._ColumnsFromFrame(target_frame)
        rotation = Matrix._Mat3Multiply(target_columns, Matrix._Mat3Transpose(source_columns))
        return [
            [rotation[0][0], rotation[0][1], rotation[0][2], 0],
            [rotation[1][0], rotation[1][1], rotation[1][2], 0],
            [rotation[2][0], rotation[2][1], rotation[2][2], 0],
            [0, 0, 0, 1],
        ]

    @staticmethod
    def ByVectors_old(vectorA: list, vectorB: list, orientationA: list = None, orientationB: list = None):
        """
        Deprecated compatibility wrapper for ByVectors.
        """
        return Matrix.ByVectors(vectorA, vectorB, orientationA=orientationA, orientationB=orientationB)

    @staticmethod
    def EigenvaluesAndVectors(matrix, mantissa: int = 6, silent: bool = False):
        """
        Returns the eigenvalues and eigenvectors of the input matrix. See https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
        
        Parameters
        ----------
        matrix : list
            The input matrix. Assumed to be a symmetric matrix.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of eigenvalues and eigenvectors of the input matrix.

        """
        try:
            import numpy as np
        except Exception as exc:
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: numpy is required. Returning None.")
                print("Error:", exc)
            return None

        if not isinstance(matrix, list):
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: The input matrix parameter is not a valid matrix. Returning None.")
            return None
        try:
            np_matrix = np.array(matrix, dtype=float)
        except Exception:
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: The input matrix parameter must contain numeric values. Returning None.")
            return None
        if np_matrix.ndim != 2 or np_matrix.shape[0] != np_matrix.shape[1] or np_matrix.shape[0] < 1:
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: The input matrix parameter is not a square matrix. Returning None.")
            return None
        if not np.allclose(np_matrix, np_matrix.T):
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: The input matrix is not symmetric. Returning None.")
            return None

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np_matrix)
        except Exception:
            if not silent:
                print("Matrix.EigenvaluesAndVectors - Error: Could not compute eigenvalues/eigenvectors. Returning None.")
            return None

        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        e_values = [Matrix._RoundValue(value, mantissa=mantissa) for value in eigenvalues.tolist()]
        e_vectors = []
        for col in range(eigenvectors.shape[1]):
            e_vectors.append([Matrix._RoundValue(value, mantissa=mantissa) for value in eigenvectors[:, col].tolist()])
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
        return [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]

    @staticmethod
    def Invert(matA, silent: bool = False):
        """
        Inverts the input matrix.

        Parameters
        ----------
        matA : list of list of float
            The input matrix.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list of list of float
            The resulting matrix after it has been inverted.

        """
        try:
            import numpy as np
        except Exception as exc:
            if not silent:
                print("Matrix.Invert - Error: numpy is required. Returning None.")
                print("Error:", exc)
            return None

        if not isinstance(matA, list):
            if not silent:
                print("Matrix.Invert - Error: The input matA parameter is not a valid 4X4 matrix. Returning None.")
            return None
        try:
            np_matrix = np.array(matA, dtype=float)
        except Exception:
            if not silent:
                print("Matrix.Invert - Error: The input matA parameter must contain numeric values. Returning None.")
            return None
        if np_matrix.shape != (4, 4):
            if not silent:
                print("Matrix.Invert - Error: The input matA parameter is not a valid 4X4 matrix. Returning None.")
            return None
        try:
            det = float(np.linalg.det(np_matrix))
        except Exception:
            if not silent:
                print("Matrix.Invert - Error: Could not compute determinant. Returning None.")
            return None
        if np.isclose(det, 0):
            if not silent:
                print("Matrix.Invert - Error: The input matA parameter is not invertible. Returning None.")
            return None
        try:
            return np.linalg.inv(np_matrix).tolist()
        except Exception:
            if not silent:
                print("Matrix.Invert - Error: Could not invert matrix. Returning None.")
            return None

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
        shape_a = Matrix._MatrixShape(matA)
        shape_b = Matrix._MatrixShape(matB)
        if shape_a is None or shape_b is None:
            raise ValueError("Both inputs must be non-empty 2D lists representing matrices.")
        if shape_a[1] != shape_b[0]:
            raise ValueError("Number of columns in matA must equal the number of rows in matB.")

        rows_A, cols_A = shape_a
        cols_B = shape_b[1]
        result = [[0.0] * cols_B for _ in range(rows_A)]
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
        shape_a = Matrix._MatrixShape(matA)
        shape_b = Matrix._MatrixShape(matB)
        if shape_a is None or shape_b is None or shape_a != shape_b:
            return None
        try:
            return [
                [matA[i][j] - matB[i][j] for j in range(shape_a[1])]
                for i in range(shape_a[0])
            ]
        except Exception:
            return None

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
        shape = Matrix._MatrixShape(matrix)
        if shape is None:
            return None
        return [list(x) for x in zip(*matrix)]
