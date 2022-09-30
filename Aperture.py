import topologic

class Aperture(topologic.Aperture):
    @staticmethod
    def ApertureByTopologyContext(topology, context):
        """
        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        context : TYPE
            DESCRIPTION.

        Returns
        -------
        aperture : TYPE
            DESCRIPTION.

        """
        # topology = item[0]
        # context = item[1]
        aperture = None
        try:
            aperture = topologic.Aperture.ByTopologyContext(topology, context)
        except:
            aperture = None
        return aperture
    
    @staticmethod
    def ApertureTopology(item):
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
        return topologic.Aperture.Topology(item)