from ..Element import BIMElement

class Walls:
    """
    Wall creation and wall semantics.
    In v1 you can support:
      - WallByFace(face, thickness, height=None) returning Cell or Shell/Cluster fallback.
      - Tagging + storing thickness/height in bim_properties.
    """

    @staticmethod
    def WallByFace(face, thickness: float, height: float = None, flip: bool = False,
                   transferDictionaries: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Create a wall from an input face (typically vertical) + thickness (+ optional height).
        NOTE: Implementation intentionally left for v1 geometry logic.
        """
        raise NotImplementedError("Walls.WallByFace is a stub. Implement geometry logic in v1.")

    @staticmethod
    def AssignWallType(wall, name: str = "", isStructural: bool = False):
        wall = BIMElement.Ensure(wall, silent=True)
        wall = BIMElement.SetType(wall, "Wall")
        if name:
            wall = BIMElement.SetProperty(wall, "type_name", name)
        wall = BIMElement.SetProperty(wall, "is_structural", bool(isStructural))
        return wall

    @staticmethod
    def SetThickness(wall, thickness: float):
        return BIMElement.SetProperty(wall, "thickness", float(thickness))

    @staticmethod
    def Thickness(wall, default=None):
        return BIMElement.GetProperty(wall, "thickness", default)
