from ..Element import BIMElement

class Spaces:
    @staticmethod
    def SpaceByCell(cell, name: str = "", number: str = ""):
        space = BIMElement.Ensure(cell, silent=True)
        space = BIMElement.SetType(space, "Space")
        if name:
            space = BIMElement.SetName(space, name)
        if number:
            space = BIMElement.SetProperty(space, "number", str(number))
        return space
