from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Topology import Topology

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_WIRE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


TOLERANCE = 0.0001


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _face_with_hole():
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


def _native_area(shape):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return abs(float(props.Mass()))


def _orientation_name(value):
    names = {
        0: "FORWARD",
        1: "REVERSED",
        2: "INTERNAL",
        3: "EXTERNAL",
    }
    try:
        return names.get(int(value), str(value))
    except Exception:
        return str(value)


def _shape(obj):
    if hasattr(obj, "shape"):
        return obj.shape
    if hasattr(obj, "GetOcctShape"):
        return obj.GetOcctShape()
    return None


def test_native_holed_face_diagnostic():
    face = _face_with_hole()

    external = Face.ExternalBoundary(face, silent=True)
    internals = Face.InternalBoundaries(face) or []

    face_shape = _shape(face)
    external_shape = _shape(external)

    print("\nWRAPPER VALUES")
    print("Face.Area:", Face.Area(face))
    print("external Face area:", Face.Area(Face.ByWire(external, silent=True)))
    print(
        "internal Face areas:",
        [Face.Area(Face.ByWire(w, silent=True)) for w in internals],
    )

    print("\nNATIVE SHAPE VALUES")
    print("native face shape type:", face_shape.ShapeType())
    print("native face orientation:", _orientation_name(face_shape.Orientation()))
    print("native face area:", _native_area(face_shape))
    print("native external wire orientation:", _orientation_name(external_shape.Orientation()))

    for i, wire in enumerate(internals):
        wire_shape = _shape(wire)
        print(
            f"native internal wire {i} orientation:",
            _orientation_name(wire_shape.Orientation()),
        )

    print("\nWIRES ACTUALLY STORED IN native face.shape")
    explorer = TopExp_Explorer(face_shape, TopAbs_WIRE)
    i = 0
    while explorer.More():
        wire_shape = topods.Wire(explorer.Current())
        print(
            f"shape wire {i}: orientation={_orientation_name(wire_shape.Orientation())}"
        )
        i += 1
        explorer.Next()

    print("native face wire count:", i)

    # Rebuild a control face directly using OCCT and explicit reversed hole
    # orientation. This is diagnostic only.
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    maker = BRepBuilderAPI_MakeFace(topods.Wire(external_shape), True)
    for wire in internals:
        hole_shape = topods.Wire(_shape(wire))
        hole_shape.Reverse()
        maker.Add(hole_shape)

    assert maker.IsDone()
    control = maker.Face()

    print("\nCONTROL OCCT FACE WITH REVERSED HOLE")
    print("control native area:", _native_area(control))
    print("control face orientation:", _orientation_name(control.Orientation()))

    explorer = TopExp_Explorer(control, TopAbs_WIRE)
    i = 0
    while explorer.More():
        wire_shape = topods.Wire(explorer.Current())
        print(
            f"control wire {i}: orientation={_orientation_name(wire_shape.Orientation())}"
        )
        i += 1
        explorer.Next()

    print("control wire count:", i)

    assert round(float(Face.Area(face)), 6) == 11.12
