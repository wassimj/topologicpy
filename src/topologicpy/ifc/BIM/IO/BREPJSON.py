from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.Cluster import Cluster

from ..Element import BIMElement

class BREPJSON:
    """
    Serialize elements using Topology.BREPString / Topology.ByBREPString
    + embed dictionaries as Python dicts.
    """

    @staticmethod
    def ElementToPayload(element) -> dict:
        element = BIMElement.Ensure(element, silent=True)
        d = Dictionary.ToPythonDictionary(Topology.Dictionary(element))
        return {
            "brep": Topology.BREPString(element),
            "dict": d
        }

    @staticmethod
    def PayloadToElement(payload: dict):
        brep = payload.get("brep", "")
        d = payload.get("dict", {})
        topo = Topology.ByBREPString(brep) if brep else None
        if topo is None:
            return None
        topo = Topology.SetDictionary(topo, Dictionary.ByPythonDictionary(d))
        return topo

    @staticmethod
    def ElementsToPayload(elements: list) -> dict:
        return {"elements": [BREPJSON.ElementToPayload(e) for e in (elements or [])]}

    @staticmethod
    def PayloadToElements(payload: dict) -> list:
        items = payload.get("elements", [])
        out = []
        for it in items:
            e = BREPJSON.PayloadToElement(it)
            if e is not None:
                out.append(e)
        return out

    @staticmethod
    def ModelToPayload(model) -> dict:
        elements = []
        try:
            elements = Cluster.Topologies(model)
        except Exception:
            elements = []
        return BREPJSON.ElementsToPayload(elements)
