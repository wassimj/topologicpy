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
    def ByRotation(angleX=0, angleY=0, angleZ=0, order="xyz"):
        """
        Creates a 4x4 rotation matrix.

        Parameters
        ----------
        angleX : float , optional
            The desired rotation angle in degrees around the X axis. The default is 0.
        angleY : float , optional
            The desired rotation angle in degrees around the Y axis. The default is 0.
        angleZ : float , optional
            The desired rotation angle in degrees around the Z axis. The default is 0.
        order : string , optional
            The order by which the roatations will be applied. The possible values are any permutation of "xyz". This input is case insensitive. The default is "xyz".

        Returns
        -------
        list
            The created 4X4 rotation matrix.

        """
        def rotateXMatrix(radians):
            """ Return matrix for rotating about the x-axis by 'radians' radians """
            c = math.cos(radians)
            s = math.sin(radians)
            return [[1, 0, 0, 0],
                    [0, c,-s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]]

        def rotateYMatrix(radians):
            """ Return matrix for rotating about the y-axis by 'radians' radians """
            
            c = math.cos(radians)
            s = math.sin(radians)
            return [[ c, 0, s, 0],
                    [ 0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [ 0, 0, 0, 1]]

        def rotateZMatrix(radians):
            """ Return matrix for rotating about the z-axis by 'radians' radians """
            
            c = math.cos(radians)
            s = math.sin(radians)
            return [[c,-s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        
        xMat = rotateXMatrix(math.radians(angleX))
        yMat = rotateYMatrix(math.radians(angleY))
        zMat = rotateZMatrix(math.radians(angleZ))
        if order.lower() == "xyz":
            return Matrix.Multiply(Matrix.Multiply(zMat,yMat),xMat)
        if order.lower() == "xzy":
            return Matrix.Multiply(Matrix.Multiply(yMat,zMat),xMat)
        if order.lower() == "yxz":
            return Matrix.Multiply(Matrix.Multiply(zMat,xMat),yMat)
        if order.lower == "yzx":
            return Matrix.Multiply(Matrix.Multiply(xMat,zMat),yMat)
        if order.lower() == "zxy":
            return Matrix.Multiply(Matrix.Multiply(yMat,xMat),zMat)
        if order.lower() == "zyx":
            return Matrix.Multiply(Matrix.Multiply(xMat,yMat),zMat)
    
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
        return [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [translateX,translateY,translateZ,1]]
    
    @staticmethod
    def Multiply(matA, matB):
        """
        Multiplies the two input matrices. When transforming an object, the first input matrix is applied first
        then the second input matrix.
        
        Parameters
        ----------
        matA : list
            The first input matrix.
        matB : list
            The second input matrix.

        Returns
        -------
        list
            The matrix resulting from the multiplication of the two input matrices.

        """
        if not isinstance(matA, list):
            return None
        if not isinstance(matB, list):
            return None
        nr = len(matA)
        nc = len(matA[0])
        matC = []
        for i in range(nr):
            tempRow = []
            for j in range(nc):
                tempRow.append(0)
            matC.append(tempRow)
        if not isinstance(matA, list):
            return None
        if not isinstance(matB, list):
            return None
        # iterate through rows of X
        for i in range(len(matA)):
            # iterate through columns of Y
            tempRow = []
            for j in range(len(matB[0])):
                # iterate through rows of Y
                for k in range(len(matB)):
                    matC[i][j] += matA[i][k] * matB[k][j]
        return matC

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
