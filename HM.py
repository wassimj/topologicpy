from molior import Molior

class HM:
    @staticmethod
    def HMIFCByCellComplex(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        molior_object = Molior.from_cellcomplex(cellcomplex=item, file=None, name="Homemaker building")
        molior_object.execute()
        return molior_object.file
