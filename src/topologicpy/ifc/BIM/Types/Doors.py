from ..Element import BIMElement

class Doors:
    """
    Doors are conceptually openings hosted by a wall.
    In v1, door geometry can be an 'opening cell' + optional frame/panel cluster later.
    """

    @staticmethod
    def DoorByFace(face, thickness: float = None, transferDictionaries: bool = True,
                   tolerance: float = 0.0001, silent: bool = False):
        raise NotImplementedError("Doors.DoorByFace is a stub.")

    @staticmethod
    def AssignDoorType(door, operation: str = "SingleSwing", handing: str = "Left"):
        door = BIMElement.Ensure(door, silent=True)
        door = BIMElement.SetType(door, "Door")
        door = BIMElement.SetProperty(door, "operation", str(operation))
        door = BIMElement.SetProperty(door, "handing", str(handing))
        return door
