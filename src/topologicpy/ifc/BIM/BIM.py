from .Element import BIMElement
from .Model import BIMModel

from .Materials.Material import Material
from .Materials.Layer import Layer
from .Materials.LayerSet import LayerSet

from .Types.Walls import Walls
from .Types.Slabs import Slabs
from .Types.Floors import Floors
from .Types.Roofs import Roofs
from .Types.Doors import Doors
from .Types.Windows import Windows
from .Types.Spaces import Spaces
from .Types.Levels import Levels
from .Types.Columns import Columns
from .Types.Beams import Beams

from .Relations.Relations import Relations

class BIM:
    """
    Facade class for TopologicPy BIM module.
    Delegates creation and semantic operations to sub-classes.
    """

    # --- Model ---
    @staticmethod
    def NewModel(name: str = ""):
        return BIMModel.ByElements([], name=name)

    @staticmethod
    def Add(model, element):
        elements = BIMModel.Elements(model)
        elements.append(element)
        return BIMModel.ByElements(elements)

    # --- Common element ops ---
    @staticmethod
    def Ensure(element, defaults=None, overwrite=False, silent=False):
        return BIMElement.Ensure(element, defaults=defaults, overwrite=overwrite, silent=silent)

    @staticmethod
    def AssignMaterial(element, material: dict, slot: str = "default"):
        return Material.AssignToElement(element, material, slot=slot)

    @staticmethod
    def AssignLayerSet(element, layerSet: dict):
        return LayerSet.AssignToElement(element, layerSet)
