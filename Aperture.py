import topologicpy
import topologic

class Aperture(topologic.Aperture):
    @staticmethod
    def ByTopologyContext(topology, context):
        """
        Description
        -----------
        Creates an aperture.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology to which the aperture will belong
        context : topologic.Context
            The context of the aperture.

        Returns
        -------
        topologic.Aperture
            The created aperture.

        """
        aperture = None
        try:
            aperture = topologic.Aperture.ByTopologyContext(topology, context)
        except:
            aperture = None
        return aperture

    @staticmethod
    def ApertureTopology(aperture):
        """
        Description
        -----------
        Returns the topology of the input aperture.
        
        Parameters
        ----------
        aperture : topologic.Aperture
            The input aperture.

        Returns
        -------
        topologic.Topology
            The topology of the input aperture.

        """
        return topologic.Aperture.Topology(aperture)