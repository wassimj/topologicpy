import math
import os
import random

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
POINT_TOLERANCE = 5.0e-6
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_TRANSLATE_STRESS_CASES", "100"))


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
    return [_coords(v) for v in (Topology.Vertices(topology, silent=True) or [])]


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


def _dedupe_points(points, tol=POINT_TOLERANCE):
    result = []
    for point in points:
        if not any(math.dist(point, other) <= tol for other in result):
            result.append(point)
    return result


def _assert_point_sets_close(expected, actual, tol=POINT_TOLERANCE):
    expected = _dedupe_points(expected, tol)
    actual = _dedupe_points(actual, tol)

    assert len(actual) == len(expected), (
        f"Unique point count mismatch: expected {len(expected)}, actual {len(actual)}.\n"
        f"Expected={expected}\nActual={actual}"
    )

    unmatched = list(actual)

    for point in expected:
        best_index = None
        best_distance = None

        for i, candidate in enumerate(unmatched):
            distance = math.dist(point, candidate)
            if best_distance is None or distance < best_distance:
                best_index = i
                best_distance = distance

        assert best_distance is not None
        assert best_distance <= tol, (
            f"Expected point {point} has no result point within {tol}; "
            f"nearest distance={best_distance}; actual={actual}"
        )
        unmatched.pop(best_index)


def _expected_points(topology, vector):
    dx, dy, dz = vector
    return [
        (x + dx, y + dy, z + dz)
        for x, y, z in _points(topology)
    ]


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
    face = Face.ByVertices(
        [
            _vertex(-1.7, -0.8, -1.1),
            _vertex(2.0, -0.2, 0.6),
            _vertex(0.3, 2.1, 2.7),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


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

    face = Topology.Rotate(
        face,
        origin=_vertex(0.0, 0.0, 0.0),
        axis=[1.0, 0.3, 0.15],
        angle=31.0,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")

    return face


def _cell():
    cell = Cell.Prism(
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
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _shell():
    shell = Cell.ExternalBoundary(_cell())
    assert Topology.IsInstance(shell, "Shell")
    return shell


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
    cluster = Cluster.ByTopologies(
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
    assert Topology.IsInstance(cluster, "Cluster")
    return cluster


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


TRANSLATIONS = {
    "zero": (0.0, 0.0, 0.0),
    "small": (0.001, -0.002, 0.003),
    "mixed": (2.5, -1.7, 3.2),
    "negative": (-7.0, -3.5, -0.25),
    "large": (1250.0, -875.0, 432.5),
    "fractional": (math.pi, -math.e, math.sqrt(2.0)),
}


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_TranslateStress.py must run using PythonOCCBackend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize("factory_name", list(FACTORIES))
@pytest.mark.parametrize("translation_name", list(TRANSLATIONS))
def test_translate_deterministic_coordinate_oracle(factory_name, translation_name):
    topology = FACTORIES[factory_name]()
    vector = TRANSLATIONS[translation_name]

    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    expected = _expected_points(topology, vector)

    result = Topology.Translate(
        topology,
        x=vector[0],
        y=vector[1],
        z=vector[2],
        transferDictionaries=False,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology"), (
        f"Translate returned invalid result for {factory_name}/{translation_name}"
    )
    assert Topology.TypeAsString(result) == original_type
    assert _counts(result) == original_counts

    _assert_point_sets_close(expected, _points(result))


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_translate_inverse_round_trip(factory_name):
    topology = FACTORIES[factory_name]()
    vector = (3.75, -2.25, 6.125)

    original_type = Topology.TypeAsString(topology)
    original_counts = _counts(topology)
    original_points = _points(topology)

    moved = Topology.Translate(
        topology,
        x=vector[0],
        y=vector[1],
        z=vector[2],
        transferDictionaries=False,
        silent=True,
    )
    assert Topology.IsInstance(moved, "Topology")

    restored = Topology.Translate(
        moved,
        x=-vector[0],
        y=-vector[1],
        z=-vector[2],
        transferDictionaries=False,
        silent=True,
    )

    assert Topology.IsInstance(restored, "Topology")
    assert Topology.TypeAsString(restored) == original_type
    assert _counts(restored) == original_counts
    _assert_point_sets_close(original_points, _points(restored))


@pytest.mark.parametrize(
    "factory_name",
    [
        "vertex",
        "edge",
        "wire",
        "face",
        "face_with_hole",
        "cell",
        "cellcomplex",
        "cluster",
    ],
)
def test_translate_random_stress(factory_name):
    rng = random.Random(f"{SEED}:{factory_name}")

    cases = RANDOM_CASES
    if factory_name == "cellcomplex":
        cases = min(RANDOM_CASES, 40)
    elif factory_name == "cluster":
        cases = min(RANDOM_CASES, 30)

    for i in range(cases):
        topology = FACTORIES[factory_name]()
        vector = (
            rng.uniform(-100.0, 100.0),
            rng.uniform(-100.0, 100.0),
            rng.uniform(-100.0, 100.0),
        )

        original_type = Topology.TypeAsString(topology)
        original_counts = _counts(topology)
        expected = _expected_points(topology, vector)

        result = Topology.Translate(
            topology,
            x=vector[0],
            y=vector[1],
            z=vector[2],
            transferDictionaries=False,
            silent=True,
        )

        assert Topology.IsInstance(result, "Topology"), (
            f"Translate failed for {factory_name}/random/{i}; vector={vector}"
        )
        assert Topology.TypeAsString(result) == original_type
        assert _counts(result) == original_counts
        _assert_point_sets_close(expected, _points(result))


def test_translate_face_with_hole_preserves_hole_and_area():
    source = _face_with_hole()
    source_area = abs(float(Face.Area(source)))
    source_holes = Face.InternalBoundaries(source) or []

    result = Topology.Translate(
        source,
        x=4.25,
        y=-7.5,
        z=2.75,
        transferDictionaries=False,
        silent=True,
    )

    assert Topology.IsInstance(result, "Face")
    result_holes = Face.InternalBoundaries(result) or []

    assert len(source_holes) == 1
    assert len(result_holes) == 1
    assert math.isclose(
        abs(float(Face.Area(result))),
        source_area,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def test_translate_parent_dictionary_transfer():
    source = _cell()

    d = Dictionary.ByPythonDictionary(
        {
            "translate_marker": "parent",
            "translate_value": 73,
        }
    )
    source = Topology.SetDictionary(source, d, silent=True)

    result = Topology.Translate(
        source,
        x=1.0,
        y=2.0,
        z=3.0,
        transferDictionaries=True,
        silent=True,
    )

    assert Topology.IsInstance(result, "Cell")

    rd = Topology.Dictionary(result, silent=True)
    assert Dictionary.ValueAtKey(rd, "translate_marker", None) == "parent"
    assert Dictionary.ValueAtKey(rd, "translate_value", None) == 73


def test_translate_face_subtopology_dictionaries():
    source = _cell()
    source_faces = Topology.Faces(source, silent=True) or []
    assert len(source_faces) == 6

    expected = []

    for i, face in enumerate(source_faces):
        marker = f"translate_face_{i}"
        expected.append(marker)

        d = Topology.Dictionary(face, silent=True)
        d = Dictionary.SetValueAtKey(d, "translate_face_marker", marker)
        Topology.SetDictionary(face, d, silent=True)

    result = Topology.Translate(
        source,
        x=3.0,
        y=-2.0,
        z=5.0,
        transferDictionaries=True,
        silent=True,
    )

    assert Topology.IsInstance(result, "Cell")

    found = []
    for face in Topology.Faces(result, silent=True) or []:
        d = Topology.Dictionary(face, silent=True)
        value = Dictionary.ValueAtKey(
            d,
            "translate_face_marker",
            None,
        )
        if value is not None:
            found.append(str(value))

    assert sorted(found) == sorted(expected)


def test_translate_backend_exception_returns_none(monkeypatch):
    source = _cell()

    def _raise(*args, **kwargs):
        raise RuntimeError("intentional translate failure")

    monkeypatch.setattr(
        Core.TopologyUtility,
        "Translate",
        staticmethod(_raise),
    )

    result = Topology.Translate(
        source,
        x=1.0,
        y=2.0,
        z=3.0,
        silent=True,
    )

    assert result is None


def test_translate_backend_invalid_result_returns_none(monkeypatch):
    source = _cell()

    def _return_none(*args, **kwargs):
        return None

    monkeypatch.setattr(
        Core.TopologyUtility,
        "Translate",
        staticmethod(_return_none),
    )

    result = Topology.Translate(
        source,
        x=1.0,
        y=2.0,
        z=3.0,
        silent=True,
    )

    assert result is None


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
def test_translate_rejects_invalid_topology(invalid):
    assert Topology.Translate(
        invalid,
        x=1.0,
        y=2.0,
        z=3.0,
        silent=True,
    ) is None
