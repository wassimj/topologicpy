from ..Element import BIMElement

class Levels:
    @staticmethod
    def LevelByElevation(elevation: float, name: str = ""):
        raise NotImplementedError("Levels.LevelByElevation is a stub.")

    @staticmethod
    def AssignLevel(element, level):
        element = BIMElement.Ensure(element, silent=True)
        return BIMElement.SetProperty(element, "level", level)
