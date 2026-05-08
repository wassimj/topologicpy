from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from ..Element import BIMElement

class Relations:
    """
    Minimal relationship encoding embedded in element dictionaries.
    (Optional later: also maintain a TopologicPy Graph for traversal.)
    """

    @staticmethod
    def Host(host, child):
        host = BIMElement.Ensure(host, silent=True)
        child = BIMElement.Ensure(child, silent=True)

        host_guid = BIMElement.GUID(host) or BIMElement.NewGUID()
        child_guid = BIMElement.GUID(child) or BIMElement.NewGUID()

        host = BIMElement.SetGUID(host, host_guid)
        child = BIMElement.SetGUID(child, child_guid)

        # child.host_guid = host_guid
        pd = Dictionary.ToPythonDictionary(Topology.Dictionary(child))
        rel = pd.get("bim_relationships", {})
        rel["host_guid"] = host_guid
        pd["bim_relationships"] = rel
        child = Topology.SetDictionary(child, Dictionary.ByPythonDictionary(pd))

        # host.opening_guids append child_guid if child is an opening type (door/window/opening)
        hpd = Dictionary.ToPythonDictionary(Topology.Dictionary(host))
        hrel = hpd.get("bim_relationships", {})
        og = hrel.get("opening_guids", [])
        if child_guid not in og:
            og.append(child_guid)
        hrel["opening_guids"] = og
        hpd["bim_relationships"] = hrel
        host = Topology.SetDictionary(host, Dictionary.ByPythonDictionary(hpd))

        return host, child
