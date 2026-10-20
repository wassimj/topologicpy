import importlib

import pytest

pytest.importorskip("OCC.Core.TopoDS")
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_CompSolid

backend = importlib.import_module("topologicpy.pythonocc_backend.cell_complex")
CellComplex = backend.CellComplex


@pytest.mark.parametrize("container_type,make_method", [
    (TopoDS_Compound, "MakeCompound"),
    (TopoDS_CompSolid, "MakeCompSolid"),
])
def test_empty_volume_result_returns_none(monkeypatch, container_type, make_method):
    shape = container_type()
    getattr(BRep_Builder(), make_method)(shape)

    class EmptyMaker:
        def SetArguments(self, args): pass
        def SetIntersect(self, value): pass
        def SetFuzzyValue(self, value): pass
        def Perform(self): pass
        def HasErrors(self): return False
        def Shape(self): return shape

    monkeypatch.setattr(backend, "BOPAlgo_MakerVolume", EmptyMaker)
    assert CellComplex._build_from_shapes([BRepPrimAPI_MakeBox(1, 1, 1).Shape()]) is None
    assert backend._as_compsolid(shape) is None
    assert CellComplex.ByCells([backend.Cell(shape=shape)]) is None


def test_failed_build_does_not_return_shapeless_complex(monkeypatch):
    cells = [backend.Cell(shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape()) for _ in range(2)]
    monkeypatch.setattr(backend, "BOPAlgo_CellsBuilder", None)
    monkeypatch.setattr(CellComplex, "_build_from_shapes", staticmethod(lambda *args: None))
    assert CellComplex.ByCells(cells) is None


def test_single_solid_remains_supported():
    cell = backend.Cell(shape=BRepPrimAPI_MakeBox(1, 1, 1).Shape())
    result = CellComplex.ByCells([cell])
    assert isinstance(result, CellComplex)
    assert len(result.Cells()) == 1


@pytest.mark.parametrize("origins,expected", [
    ([(0, 0, 0), (10, 0, 0)], False),  # disjoint
    ([(0, 0, 0), (1, 1, 1)], False),  # vertex only
    ([(0, 0, 0), (1, 1, 0)], False),  # edge only
    ([(0, 0, 0), (1, 0, 0)], True),   # shared face
    ([(0, 0, 0), (1, 0, 0), (10, 0, 0)], False),  # isolated cell
    ([(0, 0, 0), (1, 0, 0), (2, 0, 0)], True),   # face-connected chain
])
@pytest.mark.parametrize("maker_only", [False, True])
def test_bycells_requires_shared_faces(monkeypatch, origins, expected, maker_only):
    from OCC.Core.gp import gp_Pnt
    from topologicpy.CellComplex import CellComplex as PublicCellComplex

    if maker_only:
        monkeypatch.setattr(backend, "BOPAlgo_CellsBuilder", None)
    cells = [backend.Cell(shape=BRepPrimAPI_MakeBox(gp_Pnt(*p), 1, 1, 1).Shape())
             for p in origins]
    result = CellComplex.ByCells(cells)
    if expected:
        assert isinstance(result, CellComplex)
        assert len(result.Cells()) == len(cells)
    else:
        assert result is None
        # The public face-based fallback must not resurrect invalid results.
        assert PublicCellComplex.ByCells(cells, silent=True) is None
