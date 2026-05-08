class Layer:
    @staticmethod
    def ByMaterialThickness(material: dict, thickness: float) -> dict:
        return {"material": material, "thickness": float(thickness)}
