from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from ..Element import BIMElement

class LayerSet:
    @staticmethod
    def ByLayers(layers: list, name: str = "", usage: str = "Generic") -> dict:
        return {"name": str(name), "usage": str(usage), "layers": layers or []}

    @staticmethod
    def TotalThickness(layerSet: dict) -> float:
        layers = (layerSet or {}).get("layers", [])
        return sum(float(l.get("thickness", 0.0)) for l in layers)

    @staticmethod
    def AssignToElement(element, layerSet: dict):
        element = BIMElement.Ensure(element, silent=True)
        pd = Dictionary.ToPythonDictionary(Topology.Dictionary(element))
        pd["bim_materials"] = {"layerset": layerSet}
        return Topology.SetDictionary(element, Dictionary.ByPythonDictionary(pd))
