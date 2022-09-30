import topologic

class Cluster(topologic.Cluster):
    @staticmethod
    def ClusterByTopologies(item):
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
        assert isinstance(item, list), "Cluster.ByTopologies - Error: Input is not a list"
        topologyList = [x for x in item if isinstance(x, topologic.Topology)]
        return topologic.Cluster.ByTopologies(topologyList, False)

    