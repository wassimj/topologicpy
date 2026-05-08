from ..Element import BIMElement

class Slabs:
    @staticmethod
    def SlabByFace(face, thickness: float, usage: str = "Generic",
                   direction: list = [0,0,1], transferDictionaries: bool = True,
                   tolerance: float = 0.0001, silent: bool = False):
        raise NotImplementedError("Slabs.SlabByFace is a stub.")

    @staticmethod
    def AssignSlabType(slab, usage: str = "Generic"):
        slab = BIMElement.Ensure(slab, silent=True)
        slab = BIMElement.SetType(slab, "Slab")
        slab = BIMElement.SetProperty(slab, "usage", str(usage))
        return slab
