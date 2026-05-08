from ..Element import BIMElement

class Windows:
    @staticmethod
    def WindowByFace(face, thickness: float = None, sillHeight: float = 0.9,
                     transferDictionaries: bool = True, tolerance: float = 0.0001, silent: bool = False):
        raise NotImplementedError("Windows.WindowByFace is a stub.")

    @staticmethod
    def AssignWindowType(window, operation: str = "Fixed"):
        window = BIMElement.Ensure(window, silent=True)
        window = BIMElement.SetType(window, "Window")
        window = BIMElement.SetProperty(window, "operation", str(operation))
        return window
