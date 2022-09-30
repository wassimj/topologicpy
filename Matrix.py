import math

class Matrix:
    @staticmethod
    def MatrixByRotation(rx, ry, rz, order):
        """
        Parameters
        ----------
        rx : TYPE
            DESCRIPTION.
        ry : TYPE
            DESCRIPTION.
        rz : TYPE
            DESCRIPTION.
        order : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # rx, ry, rz = item
        
        def rotateXMatrix(radians):
            """ Return matrix for rotating about the x-axis by 'radians' radians """
            c = math.cos(radians)
            s = math.sin(radians)
            return Matrix([[1, 0, 0, 0],
                    [0, c,-s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]]).transposed()

        def rotateYMatrix(radians):
            """ Return matrix for rotating about the y-axis by 'radians' radians """
            
            c = math.cos(radians)
            s = math.sin(radians)
            return Matrix([[ c, 0, s, 0],
                    [ 0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [ 0, 0, 0, 1]]).transposed()

        def rotateZMatrix(radians):
            """ Return matrix for rotating about the z-axis by 'radians' radians """
            
            c = math.cos(radians)
            s = math.sin(radians)
            return Matrix([[c,-s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]).transposed()
        
        xMat = rotateXMatrix(math.radians(rx))
        yMat = rotateYMatrix(math.radians(ry))
        zMat = rotateZMatrix(math.radians(rz))
        if order == "XYZ":
            return xMat @ yMat @ zMat
        if order == "XZY":
            return xMat @ zMat @ yMat
        if order == "YXZ":
            return yMat @ xMat @ zMat
        if order == "YZX":
            return yMat @ zMat @ xMat
        if order == "ZXY":
            return zMat @ xMat @ yMat
        if order == "ZYX":
            return zMat @ yMat @ xMat
    
    @staticmethod
    def MatrixByScaling(dx, dy, dz):
        """
        Parameters
        ----------
        dx : TYPE
            DESCRIPTION.
        dy : TYPE
            DESCRIPTION.
        dz : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # dx, dy, dz = item
        return Matrix([[dx,0,0,0],
                [0,dy,0,0],
                [0,0,dz,0],
                [0,0,0,1]])
    
    @staticmethod
    def MatrixByTranslation(dx, dy, dz):
        """
        Parameters
        ----------
        dx : TYPE
            DESCRIPTION.
        dy : TYPE
            DESCRIPTION.
        dz : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # dx, dy, dz = item
        return Matrix([[1,0,0,dx],
                [0,1,0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    
    @staticmethod
    def MatrixMultiply(matA, matB):
        """
        Parameters
        ----------
        matA : TYPE
            DESCRIPTION.
        matB : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # matA, matB = item
        return matA @ matB