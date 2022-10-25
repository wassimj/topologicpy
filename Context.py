import topologicpy
import topologic

class Context:
    @staticmethod
    def ByTopologyParameters(topology, u, v, w):
        """
        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        u : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        w : TYPE
            DESCRIPTION.

        Returns
        -------
        context : TYPE
            DESCRIPTION.

        """
        # topology = item[0]
        # u = item[1]
        # v = item[2]
        # w = item[3]
        context = None
        try:
            context = topologic.Context.ByTopologyParameters(topology, u, v, w)
        except:
            context = None
        return context
    
    @staticmethod
    def Topology(item):
        """
        Parameters
        ----------
        item : TYPE
            DESCRIPTION.

        Returns
        -------
        topology : TYPE
            DESCRIPTION.

        """
        context = item
        topology = None
        try:
            topology = context.Topology()
        except:
            topology = None
        return topology
