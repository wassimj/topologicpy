import topologicpy
import topologic

class Aperture(topologic.Aperture):
    @staticmethod
    def ApertureTopology(aperture: topologic.Aperture) -> topologic.Topology:
        """
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

    @staticmethod
    def ByTopologyContext(topology: topologic.Topology, context: topologic.Context) -> topologic.Aperture:
        """
        Creates an aperture object represented by the input topology and one that belongs to the input context.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology that represents the aperture.
        context : topologic.Context
            The context of the aperture. See Context class.

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