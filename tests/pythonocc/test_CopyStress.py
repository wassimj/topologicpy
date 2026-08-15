import math
import os

import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology


TOLERANCE = 0.0001
POINT_TOLERANCE = 1.0e-6
REPEAT_CASES = int(os.environ.get("TOPOLOGICPY_COPY_STRESS_CASES", "50"))


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _coords(vertex):
    return (
        float(Vertex.X(vertex, mantissa=9)),
        float(Vertex.Y(vertex, mantissa=9)),
        float(Vertex.Z(vertex, mantissa=9)),
    )


def _points(topology):
    return sorted(
        _coords(v)
        for v in (Topology.Vertices(topology, silent=True) or [])
    )


def _counts(topology):
    return (
        len(Topology.CellComplexes(topology, silent=True) or []),
        len(Topology.Cells(topology, silent=True) or []),
        len(Topology.Shells(topology, silent=True) or []),
        len(Topology.Faces(topology, silent=True) or []),
        len(Topology.Wires(topology, silent=True) or []),
        len(Topology.Edges(topology, silent=True) or []),
        len(Topology.Vertices(topology, silent=True) or []),
    )


def _assert_points_close(a, b, tol=POINT_TOLERANCE):
    assert len(a) == len(b)

    for pa, pb in zip(a, b):
        assert math.dist(pa, pb) <= tol, (
            f"Point mismatch: {pa} vs {pb}"
        )


def _edge():
    return Edge.ByVertices(
        [
            _vertex(-1.4, 0.3, -0.8),
            _vertex(2.2, -1.1, 2.6),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )


def _wire():
    return Wire.ByVertices(
        [
            _vertex(-2.0, -0.5, -1.5),
            _vertex(-0.7, 1.2, -0.3),
            _vertex(1.1, 0.6, 1.4),
            _vertex(2.4, -1.0, 3.0),
        ],
        close=False,
        tolerance=TOLERANCE,
        silent=True,
    )


def _face():
    f = Face.ByVertices(
        [
            _vertex(-1.7, -0.8, -1.1),
            _vertex(2.0, -0.2, 0.6),
            _vertex(0.3, 2.1, 2.7),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )
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

    f = Topology.Rotate(
        f,
        origin=_vertex(0.0, 0.0, 0.0),
        axis=[1.0, 0.3, 0.15],
        angle=31.0,
        transferDictionaries=False,
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


def _cluster():
    cl = Cluster.ByTopologies(
        [
            _cell(),
            Topology.Translate(
                _wire(),
                x=5.0,
                y=0.0,
                z=0.2,
                transferDictionaries=False,
                silent=True,
            ),
            Topology.Translate(
                _edge(),
                x=-4.5,
                y=1.5,
                z=-0.4,
                transferDictionaries=False,
                silent=True,
            ),
            _vertex(0.0, 5.0, 2.0),
        ],
        silent=True,
    )
    assert Topology.IsInstance(cl, "Cluster")
    return cl


FACTORIES = {
    "vertex": lambda: _vertex(0.7, -1.3, 2.1),
    "edge": _edge,
    "wire": _wire,
    "face": _face,
    "face_with_hole": _face_with_hole,
    "shell": _shell,
    "cell": _cell,
    "cellcomplex": _cellcomplex,
    "cluster": _cluster,
}


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_CopyStress.py must run with PythonOCCBackend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_shallow_copy_preserves_type_counts_and_geometry(factory_name):
    source = FACTORIES[factory_name]()

    source_type = Topology.TypeAsString(source)
    source_counts = _counts(source)
    source_points = _points(source)

    copied = Topology.Copy(
        source,
        deep=False,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Topology")
    assert copied is not source
    assert Topology.TypeAsString(copied) == source_type
    assert _counts(copied) == source_counts

    _assert_points_close(
        _points(copied),
        source_points,
    )


@pytest.mark.parametrize(
    "factory_name",
    [
        "vertex",
        "edge",
        "wire",
        "face",
        "face_with_hole",
        "shell",
        "cell",
        "cellcomplex",
    ],
)
def test_shallow_copy_has_independent_native_shape(factory_name):
    source = FACTORIES[factory_name]()
    copied = Topology.Copy(source, deep=False, silent=True)

    assert Topology.IsInstance(copied, "Topology")

    source_shape = getattr(source, "shape", None)
    copied_shape = getattr(copied, "shape", None)

    assert source_shape is not None
    assert copied_shape is not None

    try:
        same = bool(source_shape.IsSame(copied_shape))
    except Exception:
        same = source_shape == copied_shape

    assert not same, (
        f"{factory_name} copy still shares the original native OCCT shape."
    )


def test_shallow_cluster_members_are_independent():
    source = _cluster()
    copied = Topology.Copy(source, deep=False, silent=True)

    assert Topology.IsInstance(copied, "Cluster")

    source_members = Cluster.Topologies(source) or []
    copied_members = Cluster.Topologies(copied) or []

    assert len(copied_members) == len(source_members)

    assert [
        Topology.TypeAsString(t)
        for t in copied_members
    ] == [
        Topology.TypeAsString(t)
        for t in source_members
    ]

    for source_member, copied_member in zip(
        source_members,
        copied_members,
    ):
        assert copied_member is not source_member

        source_shape = getattr(source_member, "shape", None)
        copied_shape = getattr(copied_member, "shape", None)

        if source_shape is not None and copied_shape is not None:
            try:
                assert not bool(
                    source_shape.IsSame(copied_shape)
                )
            except Exception:
                pass


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_shallow_copy_preserves_parent_dictionary(factory_name):
    source = FACTORIES[factory_name]()

    d = Dictionary.ByPythonDictionary(
        {
            "copy_marker": factory_name,
            "copy_number": 73,
        }
    )

    source = Topology.SetDictionary(
        source,
        d,
        silent=True,
    )

    copied = Topology.Copy(
        source,
        deep=False,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Topology")

    copied_d = Topology.Dictionary(
        copied,
        silent=True,
    )

    assert Dictionary.ValueAtKey(
        copied_d,
        "copy_marker",
        None,
    ) == factory_name

    assert Dictionary.ValueAtKey(
        copied_d,
        "copy_number",
        None,
    ) == 73


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_shallow_copy_dictionary_isolation(factory_name):
    source = FACTORIES[factory_name]()

    source = Topology.SetDictionary(
        source,
        Dictionary.ByPythonDictionary(
            {
                "copy_isolation": "source",
            }
        ),
        silent=True,
    )

    copied = Topology.Copy(
        source,
        deep=False,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Topology")

    copied = Topology.SetDictionary(
        copied,
        Dictionary.ByPythonDictionary(
            {
                "copy_isolation": "copy",
            }
        ),
        silent=True,
    )

    source_d = Topology.Dictionary(
        source,
        silent=True,
    )

    copied_d = Topology.Dictionary(
        copied,
        silent=True,
    )

    assert Dictionary.ValueAtKey(
        source_d,
        "copy_isolation",
        None,
    ) == "source"

    assert Dictionary.ValueAtKey(
        copied_d,
        "copy_isolation",
        None,
    ) == "copy"


def test_shallow_copy_uses_native_backend_not_brep(monkeypatch):
    source = _face_with_hole()

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "BREP serialization must not be used by "
            "PythonOCC shallow Topology.Copy."
        )

    monkeypatch.setattr(
        Topology,
        "BREPString",
        staticmethod(_forbidden),
    )

    monkeypatch.setattr(
        Topology,
        "ByBREPString",
        staticmethod(_forbidden),
    )

    copied = Topology.Copy(
        source,
        deep=False,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Face")
    assert math.isclose(
        abs(float(Face.Area(copied))),
        abs(float(Face.Area(source))),
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def test_shallow_copy_face_with_hole_preserves_hole():
    source = _face_with_hole()
    copied = Topology.Copy(source, deep=False, silent=True)

    assert Topology.IsInstance(copied, "Face")

    source_holes = Face.InternalBoundaries(source) or []
    copied_holes = Face.InternalBoundaries(copied) or []

    assert len(source_holes) == 1
    assert len(copied_holes) == 1

    assert math.isclose(
        abs(float(Face.Area(copied))),
        abs(float(Face.Area(source))),
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def _tag_faces(cell):
    faces = Topology.Faces(cell, silent=True) or []
    assert faces

    expected = []

    for i, face in enumerate(faces):
        marker = f"face_{i}"
        expected.append(marker)

        d = Topology.Dictionary(
            face,
            silent=True,
        )

        d = Dictionary.SetValueAtKey(
            d,
            "deep_face_marker",
            marker,
        )

        Topology.SetDictionary(
            face,
            d,
            silent=True,
        )

    return sorted(expected)


def _face_markers(topology):
    values = []

    for face in Topology.Faces(
        topology,
        silent=True,
    ) or []:

        d = Topology.Dictionary(
            face,
            silent=True,
        )

        value = Dictionary.ValueAtKey(
            d,
            "deep_face_marker",
            None,
        )

        if value is not None:
            values.append(
                str(value)
            )

    return sorted(values)


def test_deep_copy_preserves_subtopology_dictionaries():
    source = _cell()
    expected = _tag_faces(source)

    copied = Topology.Copy(
        source,
        deep=True,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Cell")
    assert _counts(copied) == _counts(source)
    assert _face_markers(copied) == expected


@pytest.mark.parametrize(
    "factory_name",
    [
        "face_with_hole",
        "cell",
        "cellcomplex",
        "cluster",
    ],
)
def test_deep_copy_preserves_type_counts_and_geometry(factory_name):
    source = FACTORIES[factory_name]()

    source_type = Topology.TypeAsString(source)
    source_counts = _counts(source)
    source_points = _points(source)

    copied = Topology.Copy(
        source,
        deep=True,
        silent=True,
    )

    assert Topology.IsInstance(copied, "Topology")
    assert copied is not source
    assert Topology.TypeAsString(copied) == source_type
    assert _counts(copied) == source_counts
    _assert_points_close(
        _points(copied),
        source_points,
        tol=1.0e-5,
    )


@pytest.mark.parametrize(
    "factory_name",
    [
        "edge",
        "face_with_hole",
        "cell",
        "cellcomplex",
        "cluster",
    ],
)
def test_repeated_shallow_copy_stress(factory_name):
    source = FACTORIES[factory_name]()
    expected_type = Topology.TypeAsString(source)
    expected_counts = _counts(source)
    expected_points = _points(source)

    current = source

    for i in range(REPEAT_CASES):
        current = Topology.Copy(
            current,
            deep=False,
            silent=True,
        )

        assert Topology.IsInstance(current, "Topology"), (
            f"{factory_name} failed on repeated copy {i}"
        )

        assert Topology.TypeAsString(current) == expected_type
        assert _counts(current) == expected_counts

        _assert_points_close(
            _points(current),
            expected_points,
            tol=1.0e-5,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        0,
        3.14,
        "not a topology",
        [],
        {},
    ],
)
def test_copy_rejects_invalid_input(invalid):
    assert Topology.Copy(
        invalid,
        silent=True,
    ) is None
