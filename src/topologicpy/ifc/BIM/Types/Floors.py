from .Slabs import Slabs

class Floors:
    @staticmethod
    def FloorByFace(face, thickness: float, **kwargs):
        return Slabs.SlabByFace(face, thickness, usage="Floor", **kwargs)
