import topologicpy
import topologic

class Aperture(topologic.Aperture):
    @staticmethod
    def Topology(aperture: topologic.Aperture) -> topologic.Topology:
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
        if not isinstance(aperture, topologic.Aperture):
            print("Aperture.Topology - Error: The input aperture parameter is not a valid topologic aperture. Returning None.")
            return None
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
        if not isinstance(topology, topologic.Topology):
            print("Aperture.ByTopologyContext - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not isinstance(context, topologic.Context):
            print("Aperture.ByTopologyContext - Error: The input context parameter is not a valid topologic context. Returning None.")
            return None
        aperture = None
        try:
            aperture = topologic.Aperture.ByTopologyContext(topology, context)
        except:
            print("Aperture.ByTopologyContext - Error: The operation failed. Returning None.")
            aperture = None
        return aperture
    