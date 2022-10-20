import topologic

class Cluster(topologic.Cluster):
    @staticmethod
    def ClusterByTopologies(topologies):
        """
        Description
        -----------
        Creates a topologic Cluster from the input list of topologies.

        Parameters
        ----------
        topologies : list
            The list of topologies.

        Returns
        -------
        topologic.Cluster
            The created topologic Cluster.

        """
        assert isinstance(topologies, list), "Cluster.ByTopologies - Error: Input is not a list"
        topologyList = [x for x in topologies if isinstance(x, topologic.Topology)]
        return topologic.Cluster.ByTopologies(topologyList, False)

    