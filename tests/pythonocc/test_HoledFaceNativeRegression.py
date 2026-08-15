import math

import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Topology import Topology

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


TOLERANCE = 0.0001


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _native_area(topology):
    shape = getattr(topology, "shape", None)
    assert shape is not None

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return abs(float(props.Mass()))


def _surface_area(topology):
    return sum(
        abs(float(Face.Area(face)))
        for face in (Topology.Faces(topology, silent=True) or [])
    )


def _wire_orientations(face):
    shape = getattr(face, "shape", None)
    assert shape is not None

    result = []
    explorer = TopExp_Explorer(shape, TopAbs_WIRE)

    while explorer.More():
        wire = topods.Wire(explorer.Current())
        result.append(wire.Orientation())
        explorer.Next()

    return result


def _rectangle_face_with_hole():
    outer = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.5),
        width=4.0,
        length=3.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    inner = Wire.Rectangle(
        origin=_vertex(0.35, -0.2, 0.5),
        width=1.1,
        length=0.8,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    face = Face.ByWires(
        outer,
        [inner],
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(face, "Face")
    return face


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend"


def test_native_holed_face_area_is_material_area():
    face = _rectangle_face_with_hole()

    assert math.isclose(
        _native_area(face),
        11.12,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def test_native_holed_face_wire_orientations_are_opposite():
    face = _rectangle_face_with_hole()

    orientations = _wire_orientations(face)

    assert len(orientations) == 2
    assert orientations[0] in (TopAbs_FORWARD, TopAbs_REVERSED)
    assert orientations[1] in (TopAbs_FORWARD, TopAbs_REVERSED)
    assert orientations[0] != orientations[1]


def test_native_triangulation_of_holed_face_preserves_area():
    face = _rectangle_face_with_hole()

    triangles = Face.Triangulate(
        face,
        mode=0,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert isinstance(triangles, list)
    assert len(triangles) > 2
    assert all(Topology.IsInstance(t, "Face") for t in triangles)

    assert math.isclose(
        sum(abs(float(Face.Area(t))) for t in triangles),
        11.12,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


def test_topology_triangulate_holed_face_preserves_area():
    face = _rectangle_face_with_hole()

    result = Topology.Triangulate(
        face,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")

    assert math.isclose(
        _surface_area(result),
        11.12,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )
