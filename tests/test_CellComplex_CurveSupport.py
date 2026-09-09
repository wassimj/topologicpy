import math
import os
import pytest

from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Face import Face
from topologicpy.Topology import Topology

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "topologic_core").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _circle_wire(z=0.0, radius=1.0):
    edge = Edge.Circle(origin=Vertex.ByCoordinates(0,0,z), radius=radius, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def test_cellcomplex_volume_sums_each_cell_once_and_none_is_unrounded():
    c0 = Cell.Prism(origin=Vertex.ByCoordinates(0,0,0), width=1, length=1, height=1, placement="lowerleft", silent=True)
    c1 = Cell.Prism(origin=Vertex.ByCoordinates(1,0,0), width=1, length=1, height=1, placement="lowerleft", silent=True)
    assert Topology.IsInstance(c0,"Cell") and Topology.IsInstance(c1,"Cell")
    cc = CellComplex.ByCells([c0,c1], tolerance=1e-4, silent=True)
    assert Topology.IsInstance(cc,"CellComplex")
    value = CellComplex.Volume(cc, mantissa=None, silent=True)
    assert isinstance(value, float)
    assert math.isclose(value, 2.0, rel_tol=1e-7, abs_tol=1e-7)


def test_existing_faceted_bywires_still_constructs():
    wires=[Wire.Rectangle(origin=Vertex.ByCoordinates(0,0,z), width=2, length=2, placement="center") for z in (0,1,2)]
    cc=CellComplex.ByWires(wires, triangulate=False, tolerance=1e-4, silent=True)
    assert Topology.IsInstance(cc,"CellComplex")
    assert len(CellComplex.Cells(cc)) >= 1


@pytest.mark.skipif(IS_PYTHONOCC, reason="TopologicCore capability guard is backend-specific.")
def test_topologiccore_exact_cellcomplex_loft_is_explicitly_unsupported():
    wires=[_circle_wire(z) for z in (0.0,1.0,2.0)]
    assert CellComplex.ByWires(wires, polyhedron=False, silent=True) is None
    assert CellComplex.Torus(majorRadius=2, minorRadius=0.5, uSides=8, polyhedron=False, silent=True) is None


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact curved CellComplex loft is PythonOCC-specific.")
def test_pythonocc_exact_circle_loft_has_two_cells_shared_internal_face_and_exact_volume():
    r=1.0
    wires=[_circle_wire(z,r) for z in (0.0,1.0,2.0)]
    cc=CellComplex.ByWires(wires, polyhedron=False, silent=True)
    assert Topology.IsInstance(cc,"CellComplex")
    cells=CellComplex.Cells(cc)
    assert len(cells)==2
    volume=CellComplex.Volume(cc,mantissa=None,silent=True)
    assert math.isclose(volume, 2.0*math.pi*r*r, rel_tol=1e-6, abs_tol=1e-6)
    internal=CellComplex.InternalFaces(cc)
    assert isinstance(internal,list)
    assert len(internal)==1
    faces=CellComplex.Faces(cc)
    assert any(Face.IsPlanar(f,silent=True) is False for f in faces)


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact toroidal CellComplex is PythonOCC-specific.")
def test_pythonocc_exact_torus_cellcomplex_preserves_curvature_cells_and_volume():
    R=2.0; r=0.5; n=8
    cc=CellComplex.Torus(majorRadius=R, minorRadius=r, uSides=n, polyhedron=False, silent=True)
    assert Topology.IsInstance(cc,"CellComplex")
    cells=CellComplex.Cells(cc)
    assert len(cells)==n
    volume=CellComplex.Volume(cc,mantissa=None,silent=True)
    expected=2.0*math.pi*math.pi*R*r*r
    assert math.isclose(volume, expected, rel_tol=1e-6, abs_tol=1e-6)
    faces=CellComplex.Faces(cc)
    assert any(Face.IsPlanar(f,silent=True) is False for f in faces)
    internal=CellComplex.InternalFaces(cc)
    assert isinstance(internal,list)
    assert len(internal)==n
