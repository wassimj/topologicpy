import math
import pytest

from topologicpy.Cell import Cell
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire

try:
    IS_PYTHONOCC = not Topology._IsTopologicCoreBackend()
except Exception:
    IS_PYTHONOCC = False


def _curved_face_exists(cell):
    faces = Topology.Faces(cell) or []
    return any(Face.IsPlanar(f, silent=True) is False for f in faces)


def _circle_wire(z, radius=1.0):
    edge = Edge.Circle(origin=Vertex.ByCoordinates(0,0,z), radius=radius, silent=True)
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def test_existing_polyhedral_cylinder_path_remains_available():
    cell = Cell.Cylinder(radius=1.0, height=2.0)
    assert Topology.IsInstance(cell, "Cell")
    assert Cell.Volume(cell, mantissa=5) > 0


def test_cell_volume_none_returns_unrounded_float():
    cell = Cell.Cylinder(radius=1.0, height=2.0)
    value = Cell.Volume(cell, mantissa=None, silent=True)
    assert isinstance(value, float)
    assert value > 0.0


@pytest.mark.topologiccore_only
def test_topologiccore_exact_cell_paths_are_explicitly_unsupported():
    assert Cell.Cylinder(radius=1.0, height=2.0, polyhedron=False, silent=True) is None
    assert Cell.Sphere(radius=1.0, polyhedron=False, silent=True) is None
    w0 = _circle_wire(0.0)
    w1 = _circle_wire(2.0)
    assert Cell.ByWires([w0,w1], polyhedron=False, silent=True) is None


@pytest.mark.pythonocc_only
def test_pythonocc_exact_cylinder_volume_and_curved_face():
    r,h=1.25,3.0
    cell=Cell.Cylinder(radius=r,height=h,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    assert _curved_face_exists(cell)
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True), math.pi*r*r*h, rel_tol=1e-8, abs_tol=1e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_exact_cone_frustum_volume():
    r1,r2,h=1.5,0.5,2.0
    cell=Cell.Cone(baseRadius=r1,topRadius=r2,height=h,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    expected=math.pi*h*(r1*r1+r1*r2+r2*r2)/3.0
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),expected,rel_tol=1e-8,abs_tol=1e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_exact_sphere_volume_and_curved_face():
    r=1.3
    cell=Cell.Sphere(radius=r,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    assert _curved_face_exists(cell)
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),4.0*math.pi*r**3/3.0,rel_tol=1e-8,abs_tol=1e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_exact_torus_volume():
    R,r=2.0,0.5
    cell=Cell.Torus(majorRadius=R,minorRadius=r,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    assert _curved_face_exists(cell)
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),2.0*math.pi**2*R*r*r,rel_tol=1e-8,abs_tol=1e-8)


@pytest.mark.pythonocc_only
def test_pythonocc_exact_capsule_volume():
    r,h=0.5,3.0
    cell=Cell.Capsule(radius=r,height=h,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    expected=math.pi*r*r*(h-2*r)+4.0*math.pi*r**3/3.0
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),expected,rel_tol=1e-7,abs_tol=1e-7)


@pytest.mark.pythonocc_only
def test_pythonocc_curve_preserving_cell_loft():
    r,h=1.0,2.0
    w0=_circle_wire(0.0,r)
    w1=_circle_wire(h,r)
    cell=Cell.ByWires([w0,w1],polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    assert _curved_face_exists(cell)
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),math.pi*r*r*h,rel_tol=1e-7,abs_tol=1e-7)


@pytest.mark.pythonocc_only
def test_pythonocc_native_thickened_circular_face_preserves_curvature():
    r,t=1.0,0.75
    wire=_circle_wire(0.0,r)
    face=Face.ByWire(wire,silent=True)
    assert Topology.IsInstance(face,"Face")
    cell=Cell.ByThickenedFace(face,thickness=t,bothSides=False,polyhedron=False,silent=True)
    assert Topology.IsInstance(cell,"Cell")
    assert _curved_face_exists(cell)
    assert math.isclose(Cell.Volume(cell,mantissa=None,silent=True),math.pi*r*r*t,rel_tol=1e-6,abs_tol=1e-6)
