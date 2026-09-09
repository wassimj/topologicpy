import math
import os
import pytest

from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology

BACKEND = os.environ.get("TOPOLOGICPY_CORE_BACKEND", "topologic_core").lower()
IS_PYTHONOCC = "pythonocc" in BACKEND


def _square_wire(z=0.0, size=2.0):
    h = size * 0.5
    verts = [
        Vertex.ByCoordinates(-h, -h, z),
        Vertex.ByCoordinates(h, -h, z),
        Vertex.ByCoordinates(h, h, z),
        Vertex.ByCoordinates(-h, h, z),
    ]
    return Wire.ByVertices(verts, close=True, silent=True)


def _circle_wire(z=0.0, radius=1.0):
    origin = Vertex.ByCoordinates(0.0, 0.0, z)
    edge = Edge.Circle(origin=origin, radius=radius, placement="center", silent=True)
    assert Topology.IsInstance(edge, "Edge")
    wire = Wire.ByEdges([edge], silent=True)
    assert Topology.IsInstance(wire, "Wire")
    return wire


def _faces(shell):
    result = Topology.Faces(shell, silent=True)
    return result if isinstance(result, list) else []


def _pythonocc_surface_area(topology):
    """Return exact OCCT surface area for a PythonOCC topology."""
    if not IS_PYTHONOCC or topology is None:
        return None
    shape = getattr(topology, "shape", None)
    if shape is None:
        return None

    from OCC.Core.GProp import GProp_GProps
    props = GProp_GProps()

    # pythonocc-core has exposed BRepGProp in two forms across releases.
    try:
        from OCC.Core.BRepGProp import brepgprop
        brepgprop.SurfaceProperties(shape, props)
    except (ImportError, AttributeError):
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
        brepgprop_SurfaceProperties(shape, props)

    return float(props.Mass())


def test_polyhedral_bywires_default_behavior_remains_available():
    w0 = _square_wire(z=0.0)
    w1 = _square_wire(z=3.0)
    shell = Shell.ByWires([w0, w1], triangulate=True, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(_faces(shell)) == 8


def test_polyhedral_bywires_nontriangulated_behavior_remains_available():
    w0 = _square_wire(z=0.0)
    w1 = _square_wire(z=3.0)
    shell = Shell.ByWires([w0, w1], triangulate=False, polyhedron=True, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(_faces(shell)) == 4


def test_bywirescluster_preserves_existing_polyhedral_behavior():
    w0 = _square_wire(z=0.0)
    w1 = _square_wire(z=2.0)
    cluster = Cluster.ByTopologies([w0, w1])
    shell = Shell.ByWiresCluster(cluster, triangulate=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(_faces(shell)) == 4


def test_topologiccore_exact_curve_preserving_loft_is_explicitly_unsupported():
    if IS_PYTHONOCC:
        pytest.skip("TopologicCore-specific exact-loft capability guard.")
    w0 = _circle_wire(z=0.0, radius=1.0)
    w1 = _circle_wire(z=2.0, radius=1.0)
    assert Shell.ByWires([w0, w1], polyhedron=False, silent=True) is None


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact ruled curve-preserving loft is PythonOCC-specific.")
def test_pythonocc_curve_preserving_circle_loft_is_nonplanar_and_exact_area():
    radius = 1.0
    height = 2.0
    w0 = _circle_wire(z=0.0, radius=radius)
    w1 = _circle_wire(z=height, radius=radius)
    shell = Shell.ByWires([w0, w1], polyhedron=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")

    faces = _faces(shell)
    assert len(faces) >= 1
    assert any(Face.IsPlanar(face, silent=True) is False for face in faces)

    # Measure the loft itself with OCCT. Face.Area in the current public
    # PythonOCC Face implementation is still polygon-boundary based and is
    # therefore not a valid oracle for genuinely curved faces.
    area = _pythonocc_surface_area(shell)
    expected = 2.0 * math.pi * radius * height
    assert isinstance(area, float)
    assert math.isclose(area, expected, rel_tol=1.0e-6, abs_tol=1.0e-6)

    edges = Topology.Edges(shell, silent=True) or []
    curved = [edge for edge in edges if Topology.IsInstance(edge, "Edge") and Edge.IsLinear(edge, silent=True) is False]
    assert len(curved) >= 2


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact ruled curve-preserving loft is PythonOCC-specific.")
def test_pythonocc_curve_preserving_loft_retains_radial_surface_geometry():
    radius = 1.5
    height = 3.0
    shell = Shell.ByWires([_circle_wire(0.0, radius), _circle_wire(height, radius)], polyhedron=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    face = _faces(shell)[0]
    v = Face.VertexByParameters(face, 0.37, 0.5, silent=True)
    assert Topology.IsInstance(v, "Vertex")
    x = Vertex.X(v, mantissa=9)
    y = Vertex.Y(v, mantissa=9)
    z = Vertex.Z(v, mantissa=9)
    assert math.isclose(math.hypot(x, y), radius, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert 0.0 <= z <= height


@pytest.mark.skipif(not IS_PYTHONOCC, reason="Exact ruled curve-preserving loft is PythonOCC-specific.")
def test_pythonocc_bywirescluster_forwards_curve_preserving_mode():
    w0 = _circle_wire(z=0.0, radius=1.0)
    w1 = _circle_wire(z=1.0, radius=1.0)
    cluster = Cluster.ByTopologies([w0, w1])
    shell = Shell.ByWiresCluster(cluster, polyhedron=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert any(Face.IsPlanar(face, silent=True) is False for face in _faces(shell))
