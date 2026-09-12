from topologicpy.Cell import Cell
from topologicpy.Topology import Topology


def test_brep_string_is_raw_occt_brep():
    cell = Cell.Prism()
    brep = Topology.BREPString(cell)

    assert isinstance(brep, str)
    assert brep.strip()
    assert not brep.lstrip().startswith("{")
    assert brep.lstrip().startswith("DBRep_DrawableShape")

    rebuilt = Topology.ByBREPString(brep)
    assert Topology.IsInstance(rebuilt, "Cell")


def test_export_to_brep_writes_raw_occt_brep(tmp_path):
    cell = Cell.Prism()
    path = tmp_path / "cell.brep"

    result = Topology.ExportToBREP(cell, str(path), overwrite=True)
    assert result is True

    text = path.read_text(encoding="utf-8")
    assert text.strip()
    assert not text.lstrip().startswith("{")
    assert text.lstrip().startswith("DBRep_DrawableShape")
