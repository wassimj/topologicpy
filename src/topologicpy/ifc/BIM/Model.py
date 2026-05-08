from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology

from .Element import BIMElement

class BIMModel:
    """
    Lightweight container for BIM elements.
    Stores:
      - a Cluster of elements (geometry + dictionaries)
    """

    @staticmethod
    def ByElements(elements: list, name: str = ""):
        elements = elements or []
        cl = Cluster.ByTopologies(elements)
        cl = BIMElement.Ensure(cl, defaults={"bim_type": "Model", "bim_name": name}, silent=True)
        return cl

    @staticmethod
    def Elements(model):
        if model is None:
            return []
        # Best-effort extraction (Topologic core versions differ)
        try:
            return Topology.SubTopologies(model)
        except Exception:
            try:
                return Cluster.Topologies(model)
            except Exception:
                return []

    @staticmethod
    def ElementsByType(model, bim_type: str):
        out = []
        for e in BIMModel.Elements(model):
            if BIMElement.Type(e) == bim_type:
                out.append(e)
        return out

    @staticmethod
    def ElementByGUID(model, guid: str):
        guid = str(guid)
        for e in BIMModel.Elements(model):
            if BIMElement.GUID(e) == guid:
                return e
        return None
