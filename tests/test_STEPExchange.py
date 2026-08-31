"""Focused PythonOCC STEP BREP exchange regression tests."""

import math
import re

import pytest

from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Core import Core
from topologicpy.Dictionary import Dictionary
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


def _require_pythonocc():
    backend_name = Core.Backend().__class__.__name__.lower()
    if "pythonocc" not in backend_name:
        pytest.skip("STEP BREP codec currently requires the PythonOCC backend")
    pytest.importorskip("OCC.Core.STEPControl")


def _value(topology, key):
    dictionary = Topology.Dictionary(topology, silent=True)
    return Dictionary.ValueAtKey(dictionary, key, None, silent=True)


def test_step_cell_roundtrip_preserves_type_volume_and_drops_topologic_semantics(tmp_path):
    _require_pythonocc()

    cell = Cell.Prism(width=7.0, length=3.0, height=2.0, silent=True)
    cell = Topology.SetDictionary(
        cell,
        {"semantic_only": {"must_not_be_claimed_by_step": True}},
        silent=True,
    )
    expected_volume = Cell.Volume(cell, mantissa=None, silent=True)

    path = tmp_path / "cell.step"
    assert Topology.ExportToSTEP(
        cell,
        path,
        overwrite=True,
        schema="AP242DIS",
        unit="MM",
        silent=True,
    )

    imported = Topology.BySTEPPath(path, unit="MM", silent=True)
    assert Topology.IsInstance(imported, "Cell")
    assert Cell.Volume(imported, mantissa=None, silent=True) == pytest.approx(
        expected_volume,
        rel=1.0e-8,
        abs=1.0e-8,
    )
    assert _value(imported, "semantic_only") is None


def test_step_nurbs_face_stays_bspline_surface(tmp_path):
    _require_pythonocc()

    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface

    z_values = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    control_points = [
        [Vertex.ByCoordinates(float(i), float(j), z_values[i][j]) for j in range(4)]
        for i in range(4)
    ]
    face = Face.ByNurbsParameters(
        controlPoints=control_points,
        uDegree=3,
        vDegree=3,
        tolerance=1.0e-4,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")

    path = tmp_path / "nurbs.stp"
    assert Topology.ExportToSTEP(face, path, overwrite=True, unit="MM", silent=True)
    imported = Topology.BySTEPPath(path, unit="MM", silent=True)
    assert Topology.IsInstance(imported, "Face")

    before = BRepAdaptor_Surface(Topology.OCCTShape(face, silent=True), True)
    after = BRepAdaptor_Surface(Topology.OCCTShape(imported, silent=True), True)
    assert before.GetType() == GeomAbs_BSplineSurface
    assert after.GetType() == GeomAbs_BSplineSurface

    # STEP exchange must not silently introduce a tessellated representation.
    step_text = path.read_text(encoding="utf-8", errors="ignore").upper()
    assert "TESSELLATED_" not in step_text
    assert "TOPOLOGICPY_STEP_ROOT_TYPE=FACE" in step_text

    assert Face.Area(imported, mantissa=None, silent=True) == pytest.approx(
        Face.Area(face, mantissa=None, silent=True), rel=1.0e-8, abs=1.0e-8
    )


def test_step_face_with_hole_preserves_trimmed_area_and_internal_boundary(tmp_path):
    _require_pythonocc()

    outer = Wire.Rectangle(width=10.0, length=8.0, silent=True)
    inner = Wire.Rectangle(width=2.0, length=2.0, silent=True)
    face = Face.ByWires(outer, [inner], silent=True)
    assert Topology.IsInstance(face, "Face")

    path = tmp_path / "holed_face.step"
    assert Topology.ExportToSTEP(face, path, overwrite=True, silent=True)
    imported = Topology.BySTEPPath(path, silent=True)
    assert Topology.IsInstance(imported, "Face")

    assert Face.Area(imported, mantissa=None, silent=True) == pytest.approx(
        Face.Area(face, mantissa=None, silent=True), rel=1.0e-8, abs=1.0e-8
    )
    assert len(Face.InternalBoundaries(imported, silent=True) or []) == 1


def test_step_unit_declaration_roundtrip_preserves_numeric_coordinates(tmp_path):
    _require_pythonocc()

    cell = Cell.Prism(width=2.0, length=3.0, height=4.0, silent=True)
    path = tmp_path / "meters.step"
    assert Topology.ExportToSTEP(
        cell,
        path,
        overwrite=True,
        unit="M",
        silent=True,
    )
    imported = Topology.BySTEPPath(path, unit="M", silent=True)
    assert Topology.IsInstance(imported, "Cell")
    assert Cell.Volume(imported, mantissa=None, silent=True) == pytest.approx(
        Cell.Volume(cell, mantissa=None, silent=True), rel=1.0e-8, abs=1.0e-8
    )


def test_generic_save_load_dispatches_step_extensions(tmp_path):
    _require_pythonocc()

    cell = Cell.Prism(width=2.0, length=2.0, height=2.0, silent=True)
    path = tmp_path / "generic.stp"
    assert Topology.Save(cell, path, overwrite=True, silent=True)
    imported = Topology.Load(path, silent=True)
    assert Topology.IsInstance(imported, "Cell")


def test_step_rejects_overwrite_and_invalid_settings(tmp_path):
    _require_pythonocc()

    cell = Cell.Prism(silent=True)
    path = tmp_path / "overwrite.step"
    assert Topology.ExportToSTEP(cell, path, overwrite=True, silent=True)
    assert not Topology.ExportToSTEP(cell, path, overwrite=False, silent=True)
    assert not Topology.ExportToSTEP(cell, tmp_path / "bad.step", schema="AP999", silent=True)
    assert not Topology.ExportToSTEP(cell, tmp_path / "bad2.step", unit="PARSEC", silent=True)


def test_step_single_child_transport_compounds_are_unwrapped():
    _require_pythonocc()

    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from topologicpy.io.step import _normalise_transport_shape

    face = Face.Rectangle(width=2.0, length=1.0, silent=True)
    face_shape = Topology.OCCTShape(face, silent=True)

    inner = TopoDS_Compound()
    outer = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(inner)
    builder.Add(inner, face_shape)
    builder.MakeCompound(outer)
    builder.Add(outer, inner)

    result = _normalise_transport_shape(outer)
    wrapped = Topology.ByOCCTShape(result, silent=True)
    assert Topology.IsInstance(wrapped, "Face")


def test_step_multi_child_compound_is_not_unwrapped():
    _require_pythonocc()

    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from topologicpy.io.step import _normalise_transport_shape

    face_a = Face.Rectangle(width=2.0, length=1.0, silent=True)
    face_b = Topology.Translate(face_a, x=3.0, silent=True)
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, Topology.OCCTShape(face_a, silent=True))
    builder.Add(compound, Topology.OCCTShape(face_b, silent=True))

    result = _normalise_transport_shape(compound)
    assert result.ShapeType() == compound.ShapeType()



def test_step_root_type_hint_distinguishes_face_from_one_face_shell(tmp_path):
    _require_pythonocc()

    face = Face.Rectangle(width=3.0, length=2.0, silent=True)
    assert Topology.IsInstance(face, "Face")

    face_path = tmp_path / "root_face.step"
    assert Topology.ExportToSTEP(face, face_path, overwrite=True, silent=True)
    imported_face = Topology.BySTEPPath(face_path, silent=True)
    assert Topology.IsInstance(imported_face, "Face")

    shell = Shell.ByFaces([face], silent=True)
    assert Topology.IsInstance(shell, "Shell")
    shell_path = tmp_path / "one_face_shell.step"
    assert Topology.ExportToSTEP(shell, shell_path, overwrite=True, silent=True)
    imported_shell = Topology.BySTEPPath(shell_path, silent=True)
    assert Topology.IsInstance(imported_shell, "Shell")
    assert len(Topology.Faces(imported_shell, silent=True) or []) == 1

    assert "TOPOLOGICPY_STEP_ROOT_TYPE=FACE" in face_path.read_text(
        encoding="utf-8", errors="ignore"
    ).upper()
    assert "TOPOLOGICPY_STEP_ROOT_TYPE=SHELL" in shell_path.read_text(
        encoding="utf-8", errors="ignore"
    ).upper()


def test_step_root_type_hint_is_optional_and_third_party_shells_remain_shells(tmp_path):
    _require_pythonocc()

    face = Face.Rectangle(width=3.0, length=2.0, silent=True)
    path = tmp_path / "externalized.step"
    assert Topology.ExportToSTEP(face, path, overwrite=True, silent=True)

    # Simulate a third-party STEP processor that strips TopologicPy's standard
    # Part 21 comment while leaving the STEP BREP untouched.
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(
        r"/\*\s*TOPOLOGICPY_STEP_ROOT_TYPE\s*=\s*[A-Z]+\s*\*/\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    path.write_text(text, encoding="utf-8")

    imported = Topology.BySTEPPath(path, silent=True)
    # Without provenance information, preserving OCCT's Shell is the only
    # defensible interpretation: a one-face Shell may be genuine.
    assert Topology.IsInstance(imported, "Shell")
    assert len(Topology.Faces(imported, silent=True) or []) == 1



def test_step_topologicpy_cellcomplex_reconstructs_shared_topology(tmp_path):
    _require_pythonocc()

    source = CellComplex.Prism(
        width=4.0,
        length=4.0,
        height=3.0,
        uSides=2,
        vSides=1,
        wSides=1,
        silent=True,
    )
    assert Topology.IsInstance(source, "CellComplex")

    source_cells = Topology.Cells(source, silent=True) or []
    source_internal = CellComplex.InternalFaces(source) or []
    source_volume = sum(
        Cell.Volume(cell, mantissa=None, silent=True) for cell in source_cells
    )
    assert len(source_cells) == 2
    assert len(source_internal) == 1

    path = tmp_path / "cellcomplex.step"
    assert Topology.ExportToSTEP(source, path, overwrite=True, silent=True)
    step_text = path.read_text(encoding="utf-8", errors="ignore").upper()
    assert "TOPOLOGICPY_STEP_ROOT_TYPE=CELLCOMPLEX" in step_text

    imported = Topology.BySTEPPath(path, silent=True)
    assert Topology.IsInstance(imported, "CellComplex")

    imported_cells = Topology.Cells(imported, silent=True) or []
    imported_internal = CellComplex.InternalFaces(imported) or []
    imported_volume = sum(
        Cell.Volume(cell, mantissa=None, silent=True) for cell in imported_cells
    )

    assert len(imported_cells) == len(source_cells)
    assert len(imported_internal) == len(source_internal)
    assert imported_volume == pytest.approx(source_volume, rel=1.0e-7, abs=1.0e-8)


def test_step_unhinted_multisolid_model_remains_cluster(tmp_path):
    _require_pythonocc()

    source = CellComplex.Prism(
        width=4.0,
        length=4.0,
        height=3.0,
        uSides=2,
        vSides=1,
        wSides=1,
        silent=True,
    )
    assert Topology.IsInstance(source, "CellComplex")

    path = tmp_path / "third_party_multisolid.step"
    assert Topology.ExportToSTEP(source, path, overwrite=True, silent=True)

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(
        r"/\*\s*TOPOLOGICPY_STEP_ROOT_TYPE\s*=\s*[A-Z]+\s*\*/\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    path.write_text(text, encoding="utf-8")

    imported = Topology.BySTEPPath(path, silent=True)
    assert Topology.IsInstance(imported, "Cluster")
    assert len(Topology.Cells(imported, silent=True) or []) == 2
