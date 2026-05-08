from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from ..Element import BIMElement

class Material:
    """
    Material is stored as a Python dict payload.
    Assignment embeds material(s) in element dictionary under bim_materials.
    """

    @staticmethod
    def ByName(name: str, category: str = "Generic", properties: dict = None) -> dict:
        return {
            "name": str(name),
            "category": str(category),
            "properties": properties or {}
        }

    @staticmethod
    def AssignToElement(element, material: dict, slot: str = "default"):
        element = BIMElement.Ensure(element, silent=True)
        pd = Dictionary.ToPythonDictionary(Topology.Dictionary(element))
        mats = pd.get("bim_materials", {})
        mats[str(slot)] = material
        pd["bim_materials"] = mats
        return Topology.SetDictionary(element, Dictionary.ByPythonDictionary(pd))
