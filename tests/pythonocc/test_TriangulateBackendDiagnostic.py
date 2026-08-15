import inspect

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Topology import Topology


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


def _area(face):
    return abs(float(Face.Area(face)))


def _surface_area(topology_or_faces):
    if isinstance(topology_or_faces, list):
        faces = topology_or_faces
    else:
        faces = Topology.Faces(topology_or_faces, silent=True) or []

    return sum(
        _area(face)
        for face in faces
        if Topology.IsInstance(face, "Face")
    )


def test_face_with_hole_triangulation_diagnostic():
    backend = Core.Backend()
    print("\nBACKEND")
    print("class:", backend.__class__.__name__)
    print("module:", backend.__class__.__module__)

    utility = getattr(backend, "FaceUtility", None)
    assert utility is not None

    triangulate_method = getattr(utility, "Triangulate", None)
    assert callable(triangulate_method)

    print("\nACTIVE FaceUtility.Triangulate")
    try:
        print("file:", inspect.getsourcefile(triangulate_method))
    except Exception as error:
        print("file: <unavailable>", repr(error))

    try:
        print("module:", triangulate_method.__module__)
    except Exception:
        pass

    try:
        source = inspect.getsource(triangulate_method)
        print(source)
    except Exception as error:
        print("source: <unavailable>", repr(error))

    face = _face_with_hole()

    print("\nORIGINAL FACE")
    print("type:", Topology.TypeAsString(face))
    print("area:", _area(face))

    external = Face.ExternalBoundary(face, silent=True)
    internals = Face.InternalBoundaries(face) or []

    external_face = Face.ByWire(
        external,
        tolerance=TOLERANCE,
        silent=True,
    )
    print("external boundary face area:", _area(external_face))

    internal_areas = []
    for wire in internals:
        hole_face = Face.ByWire(
            wire,
            tolerance=TOLERANCE,
            silent=True,
        )
        internal_areas.append(_area(hole_face))

    print("internal boundary face areas:", internal_areas)

    origin = Topology.Centroid(face)
    normal = Face.Normal(face)

    flat_face = Topology.Flatten(
        face,
        origin=origin,
        direction=normal,
        transferDictionaries=False,
        silent=True,
    )

    print("\nFLATTENED FACE")
    print("valid:", Topology.IsInstance(flat_face, "Face"))
    print("type:", Topology.TypeAsString(flat_face) if flat_face else None)
    print("area:", _area(flat_face) if Topology.IsInstance(flat_face, "Face") else None)

    if Topology.IsInstance(flat_face, "Face"):
        flat_external = Face.ExternalBoundary(flat_face, silent=True)
        flat_internals = Face.InternalBoundaries(flat_face) or []

        flat_external_face = Face.ByWire(
            flat_external,
            tolerance=TOLERANCE,
            silent=True,
        )
        print("flat external boundary face area:", _area(flat_external_face))

        flat_internal_areas = []
        for wire in flat_internals:
            hole_face = Face.ByWire(
                wire,
                tolerance=TOLERANCE,
                silent=True,
            )
            flat_internal_areas.append(_area(hole_face))

        print("flat internal boundary face areas:", flat_internal_areas)

    print("\nDIRECT BACKEND TRIANGULATION")

    for deflection in (0.0, 1.0e-6, 0.01, 0.1, 0.4):
        raw_triangles = []

        try:
            status = Core.FaceUtility.Triangulate(
                flat_face,
                deflection,
                raw_triangles,
            )
            print(
                f"deflection={deflection}: "
                f"status={status}, "
                f"triangles={len(raw_triangles)}, "
                f"area={_surface_area(raw_triangles)}"
            )
        except Exception as error:
            print(
                f"deflection={deflection}: "
                f"ERROR {type(error).__name__}: {error}"
            )

    print("\nPUBLIC Face.Triangulate")

    public_triangles = Face.Triangulate(
        face,
        mode=0,
        tolerance=TOLERANCE,
        silent=True,
    )

    print(
        "triangles:",
        len(public_triangles) if isinstance(public_triangles, list) else None
    )
    print(
        "area:",
        _surface_area(public_triangles)
        if isinstance(public_triangles, list)
        else None
    )

    # This is intentionally the only semantic assertion.
    assert _area(face) == 11.12
