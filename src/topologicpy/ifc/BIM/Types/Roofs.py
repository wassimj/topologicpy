from .Slabs import Slabs

class Roofs:
    @staticmethod
    def RoofByFace(face, thickness: float, **kwargs):
        return Slabs.SlabByFace(face, thickness, usage="Roof", **kwargs)
