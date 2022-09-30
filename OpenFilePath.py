import os

class OpenFilePath:
    @staticmethod
    def OpenFilePath(filePath):
        """
        Parameters
        ----------
        filePath : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        os.system("start "+filePath)
        return True