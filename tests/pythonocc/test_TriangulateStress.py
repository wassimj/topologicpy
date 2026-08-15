import math
import os
import random

import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology


TOLERANCE = 0.0001
METRIC_TOLERANCE = 1.0e-5
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_TRIANGULATE_STRESS_CASES", "75"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _edge():
    e = Edge.ByVertices(
        [_vertex(-1.2, 0.4, 0.1), _vertex(2.3, -0.7, 1.6)],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(e, "Edge")
    return e


def _wire():
    w = Wire.Rectangle(
        origin=_vertex(0.2, -0.4, 0.7),
        width=2.6,
        length=1.7,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(w, "Wire")
    return w


def _face(sides=7):
    w = Wire.Circle(
        origin=_vertex(0.3, -0.2, 0.6),
        radius=2.0,
        sides=sides,
        close=True,
        placement="center",
        tolerance=TOLERANCE,
    )
    f = Face.ByWire(w, tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(f, "Face")
    return f


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
    f = Face.ByWires(
        outer,
        [inner],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(f, "Face")
    return f


def _cell():
    c = Cell.Prism(
        origin=_vertex(0.4, -0.7, 1.1),
        width=2.4,
        length=1.7,
        height=3.2,
        uSides=1,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(c, "Cell")
    return c


def _shell():
    s = Cell.ExternalBoundary(_cell())
    assert Topology.IsInstance(s, "Shell")
    return s


def _cellcomplex():
    cc = CellComplex.Prism(
        origin=_vertex(-0.6, 0.8, -0.3),
        width=2.8,
        length=2.2,
        height=2.6,
        uSides=2,
        vSides=2,
        wSides=2,
        placement="center",
        tolerance=TOLERANCE,
    )
    assert Topology.IsInstance(cc, "CellComplex")
    return cc


def _cluster_constituents(cluster):
    for args in ((), ([],), (None, [])):
        try:
            if args == ():
                result = Core.InstanceCall(cluster, "Topologies")
                if isinstance(result, list):
                    return result
            elif len(args) == 1:
                output = args[0]
                Core.InstanceCall(cluster, "Topologies", output)
                if output:
                    return output
            else:
                output = args[1]
                Core.InstanceCall(cluster, "Topologies", args[0], output)
                if output:
                    return output
        except Exception:
            pass

    return Cluster.Topologies(cluster, tolerance=TOLERANCE, silent=True) or []


def _mixed_cluster():
    c = _cell()
    w = Topology.Translate(
        _wire(),
        x=4.5,
        y=0.0,
        z=0.3,
        transferDictionaries=False,
        silent=True,
    )
    e = Topology.Translate(
        _edge(),
        x=-4.0,
        y=1.0,
        z=-0.5,
        transferDictionaries=False,
        silent=True,
    )
    v = _vertex(0.0, 5.0, 2.0)

    cluster = Cluster.ByTopologies(
        [c, w, e, v],
        silent=True,
    )
    assert Topology.IsInstance(cluster, "Cluster")
    return cluster


def _face_area(face):
    return abs(float(Face.Area(face)))


def _surface_area(topology):
    return sum(
        _face_area(face)
        for face in (Topology.Faces(topology, silent=True) or [])
    )


def _cell_volume(cell):
    return abs(float(Cell.Volume(cell)))


def _volume(topology):
    if Topology.IsInstance(topology, "Cell"):
        return _cell_volume(topology)

    cells = Topology.Cells(topology, silent=True) or []
    return sum(_cell_volume(cell) for cell in cells)


def _all_faces_triangular(topology):
    faces = Topology.Faces(topology, silent=True) or []
    assert faces, "Expected the triangulated result to contain Faces."
    return all(
        len(Topology.Vertices(face, silent=True) or []) == 3
        for face in faces
    )


def _assert_metric_close(actual, expected, rel=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE):
    assert math.isclose(
        float(actual),
        float(expected),
        rel_tol=rel,
        abs_tol=abs_tol,
    ), f"Metric mismatch: expected {expected}, actual {actual}"


def _random_axis(rng):
    while True:
        axis = [
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        ]
        if math.sqrt(sum(v * v for v in axis)) > 1.0e-6:
            return axis


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_TriangulateStress.py must be run with the PythonOCC backend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _vertex(1.0, 2.0, 3.0),
        _edge,
        _wire,
    ],
)
def test_triangulate_no_faces_returns_original(factory):
    topology = factory()

    result = Topology.Triangulate(
        topology,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert result is topology


@pytest.mark.parametrize("sides", [4, 5, 7, 12, 24])
def test_triangulate_polygon_face_preserves_area(sides):
    face = _face(sides=sides)
    original_area = _face_area(face)

    result = Topology.Triangulate(
        face,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")
    assert _all_faces_triangular(result)
    _assert_metric_close(_surface_area(result), original_area)


def test_triangulate_face_with_hole_preserves_area():
    face = _face_with_hole()
    original_area = _face_area(face)

    result = Topology.Triangulate(
        face,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")
    assert _all_faces_triangular(result)
    _assert_metric_close(_surface_area(result), original_area)


def test_triangulate_shell_preserves_area():
    shell = _shell()
    original_area = _surface_area(shell)

    result = Topology.Triangulate(
        shell,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")
    assert _all_faces_triangular(result)
    _assert_metric_close(_surface_area(result), original_area)


def test_triangulate_cell_preserves_volume():
    cell = _cell()
    original_volume = _volume(cell)

    result = Topology.Triangulate(
        cell,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Cell")
    assert _all_faces_triangular(result)
    _assert_metric_close(_volume(result), original_volume)


def test_triangulate_cellcomplex_preserves_volume_and_cell_count():
    cc = _cellcomplex()
    original_volume = _volume(cc)
    original_cell_count = len(Topology.Cells(cc, silent=True) or [])

    result = Topology.Triangulate(
        cc,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "CellComplex")
    assert _all_faces_triangular(result)
    assert len(Topology.Cells(result, silent=True) or []) == original_cell_count
    _assert_metric_close(_volume(result), original_volume)


def test_triangulate_mixed_cluster_preserves_constituent_types():
    cluster = _mixed_cluster()

    result = Topology.Triangulate(
        cluster,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Cluster")

    constituents = _cluster_constituents(result)
    type_names = sorted(Topology.TypeAsString(t) for t in constituents)

    assert type_names == sorted(["Cell", "Wire", "Edge", "Vertex"])

    cell = next(t for t in constituents if Topology.IsInstance(t, "Cell"))
    assert _all_faces_triangular(cell)


def test_triangulate_face_transfers_dictionary_to_triangles():
    face = _face(sides=8)

    d = Dictionary.ByPythonDictionary(
        {
            "triangulate_source": "original_face",
            "triangulate_value": 37,
        }
    )
    face = Topology.SetDictionary(face, d, silent=True)

    result = Topology.Triangulate(
        face,
        transferDictionaries=True,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")

    triangles = Topology.Faces(result, silent=True) or []
    assert len(triangles) > 1

    for triangle in triangles:
        td = Topology.Dictionary(triangle, silent=True)
        assert Dictionary.ValueAtKey(td, "triangulate_source", None) == "original_face"
        assert Dictionary.ValueAtKey(td, "triangulate_value", None) == 37


def test_triangulate_already_triangular_face():
    vertices = [
        _vertex(0.0, 0.0, 0.0),
        _vertex(2.0, 0.0, 0.0),
        _vertex(0.5, 1.5, 0.0),
    ]
    face = Face.ByVertices(
        vertices,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")

    result = Topology.Triangulate(
        face,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Shell")
    assert _all_faces_triangular(result)
    _assert_metric_close(_surface_area(result), _face_area(face))


def test_triangulate_rotated_polygon_face_stress():
    rng = random.Random(f"{SEED}:face")

    for i in range(RANDOM_CASES):
        sides = rng.randint(4, 24)
        face = _face(sides=sides)
        original_area = _face_area(face)

        origin = _vertex(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
        )
        axis = _random_axis(rng)
        angle = rng.uniform(-720.0, 720.0)

        face = Topology.Rotate(
            face,
            origin=origin,
            axis=axis,
            angle=angle,
            transferDictionaries=False,
            tolerance=TOLERANCE,
            silent=True,
        )
        assert Topology.IsInstance(face, "Face"), f"Rotate failed in face stress case {i}"

        result = Topology.Triangulate(
            face,
            tolerance=TOLERANCE,
            silent=True,
        )

        assert Topology.IsInstance(result, "Shell"), f"Triangulate failed in face stress case {i}"
        assert _all_faces_triangular(result)
        _assert_metric_close(_surface_area(result), original_area, rel=5.0e-5, abs_tol=5.0e-5)


def test_triangulate_rotated_cell_stress():
    rng = random.Random(f"{SEED}:cell")

    for i in range(RANDOM_CASES):
        cell = Cell.Prism(
            origin=_vertex(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ),
            width=rng.uniform(0.5, 4.0),
            length=rng.uniform(0.5, 4.0),
            height=rng.uniform(0.5, 4.0),
            uSides=1,
            vSides=1,
            wSides=1,
            placement="center",
            tolerance=TOLERANCE,
            silent=True,
        )
        assert Topology.IsInstance(cell, "Cell")

        original_volume = _volume(cell)

        cell = Topology.Rotate(
            cell,
            origin=_vertex(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ),
            axis=_random_axis(rng),
            angle=rng.uniform(-720.0, 720.0),
            transferDictionaries=False,
            tolerance=TOLERANCE,
            silent=True,
        )
        assert Topology.IsInstance(cell, "Cell"), f"Rotate failed in cell stress case {i}"

        result = Topology.Triangulate(
            cell,
            tolerance=TOLERANCE,
            silent=True,
        )

        assert Topology.IsInstance(result, "Cell"), f"Triangulate failed in cell stress case {i}"
        assert _all_faces_triangular(result)
        _assert_metric_close(_volume(result), original_volume, rel=5.0e-5, abs_tol=5.0e-5)


def test_triangulate_rotated_cellcomplex_stress():
    rng = random.Random(f"{SEED}:cellcomplex")

    # CellComplex reconstruction is more expensive, so use a bounded subset
    # of the random stress count while still exercising many orientations.
    cases = max(10, min(RANDOM_CASES, 35))

    for i in range(cases):
        cc = CellComplex.Prism(
            origin=_vertex(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ),
            width=rng.uniform(1.5, 3.5),
            length=rng.uniform(1.5, 3.5),
            height=rng.uniform(1.5, 3.5),
            uSides=2,
            vSides=2,
            wSides=2,
            placement="center",
            tolerance=TOLERANCE,
        )
        assert Topology.IsInstance(cc, "CellComplex")

        original_volume = _volume(cc)
        original_cell_count = len(Topology.Cells(cc, silent=True) or [])

        cc = Topology.Rotate(
            cc,
            origin=_vertex(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ),
            axis=_random_axis(rng),
            angle=rng.uniform(-720.0, 720.0),
            transferDictionaries=False,
            tolerance=TOLERANCE,
            silent=True,
        )
        assert Topology.IsInstance(cc, "CellComplex"), f"Rotate failed in CellComplex stress case {i}"

        result = Topology.Triangulate(
            cc,
            tolerance=TOLERANCE,
            silent=True,
        )

        assert Topology.IsInstance(result, "CellComplex"), (
            f"Triangulate failed in CellComplex stress case {i}"
        )
        assert _all_faces_triangular(result)
        assert len(Topology.Cells(result, silent=True) or []) == original_cell_count
        _assert_metric_close(_volume(result), original_volume, rel=1.0e-4, abs_tol=1.0e-4)
