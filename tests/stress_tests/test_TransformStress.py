import math
import os
import random
import pytest

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
POINT_TOLERANCE = 5.0e-5
SEED = 20260815
RANDOM_CASES = int(os.environ.get("TOPOLOGICPY_TRANSFORM_STRESS_CASES", "75"))


def _v(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _coords(v):
    return (float(Vertex.X(v, mantissa=9)), float(Vertex.Y(v, mantissa=9)), float(Vertex.Z(v, mantissa=9)))


def _points(t):
    return [_coords(v) for v in (Topology.Vertices(t, silent=True) or [])]


def _counts(t):
    return (
        len(Topology.CellComplexes(t, silent=True) or []),
        len(Topology.Cells(t, silent=True) or []),
        len(Topology.Shells(t, silent=True) or []),
        len(Topology.Faces(t, silent=True) or []),
        len(Topology.Wires(t, silent=True) or []),
        len(Topology.Edges(t, silent=True) or []),
        len(Topology.Vertices(t, silent=True) or []),
    )


def _apply(p, m):
    x, y, z = p
    return (
        m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3],
        m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3],
        m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3],
    )


def _dedupe(points, tol=POINT_TOLERANCE):
    out = []
    for p in points:
        if not any(math.dist(p, q) <= tol for q in out):
            out.append(p)
    return out


def _assert_points(expected, actual, tol=POINT_TOLERANCE):
    expected, actual = _dedupe(expected, tol), _dedupe(actual, tol)
    assert len(expected) == len(actual), (expected, actual)
    unmatched = list(actual)
    for p in expected:
        i, d = min(enumerate(unmatched), key=lambda item: math.dist(p, item[1]))
        assert math.dist(p, d) <= tol, f"No match for {p}; actual={actual}"
        unmatched.pop(i)


def _det(m):
    return (
        m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
        - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
        + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
    )


def _inverse(m):
    a = [row[:3] for row in m[:3]]
    d = _det(m)
    assert abs(d) > 1e-12
    r = [
        [(a[1][1]*a[2][2]-a[1][2]*a[2][1])/d, (a[0][2]*a[2][1]-a[0][1]*a[2][2])/d, (a[0][1]*a[1][2]-a[0][2]*a[1][1])/d],
        [(a[1][2]*a[2][0]-a[1][0]*a[2][2])/d, (a[0][0]*a[2][2]-a[0][2]*a[2][0])/d, (a[0][2]*a[1][0]-a[0][0]*a[1][2])/d],
        [(a[1][0]*a[2][1]-a[1][1]*a[2][0])/d, (a[0][1]*a[2][0]-a[0][0]*a[2][1])/d, (a[0][0]*a[1][1]-a[0][1]*a[1][0])/d],
    ]
    t = [m[0][3], m[1][3], m[2][3]]
    it = [-sum(r[i][j]*t[j] for j in range(3)) for i in range(3)]
    return [r[0]+[it[0]], r[1]+[it[1]], r[2]+[it[2]], [0.0, 0.0, 0.0, 1.0]]


def _edge():
    return Edge.ByVertices([_v(-1.4, 0.3, -0.8), _v(2.2, -1.1, 2.6)], tolerance=TOLERANCE, silent=True)


def _wire():
    return Wire.ByVertices([_v(-2,-.5,-1.5), _v(-.7,1.2,-.3), _v(1.1,.6,1.4), _v(2.4,-1,3)], close=False, tolerance=TOLERANCE, silent=True)


def _face():
    f = Face.ByVertices([_v(-1.7,-.8,-1.1), _v(2,-.2,.6), _v(.3,2.1,2.7)], tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(f, "Face")
    return f


def _face_hole():
    outer = Wire.Rectangle(origin=_v(0,0,.5), width=4, length=3, placement="center", tolerance=TOLERANCE, silent=True)
    inner = Wire.Rectangle(origin=_v(.35,-.2,.5), width=1.1, length=.8, placement="center", tolerance=TOLERANCE, silent=True)
    f = Face.ByWires(outer, [inner], tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(f, "Face")
    f = Topology.Rotate(f, origin=_v(0,0,0), axis=[1,.3,.15], angle=31, transferDictionaries=False, tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(f, "Face")
    return f


def _cell():
    c = Cell.Prism(origin=_v(.4,-.7,1.1), width=2.4, length=1.7, height=3.2, uSides=1, vSides=1, wSides=1, placement="center", tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(c, "Cell")
    return c


def _shell():
    return Cell.ExternalBoundary(_cell())


def _cc():
    cc = CellComplex.Prism(origin=_v(-.6,.8,-.3), width=2.8, length=2.2, height=2.6, uSides=2, vSides=2, wSides=2, placement="center", tolerance=TOLERANCE)
    assert Topology.IsInstance(cc, "CellComplex")
    return cc


def _cluster():
    c = Cluster.ByTopologies([
        _cell(),
        Topology.Translate(_wire(), x=5, z=.2, transferDictionaries=False, silent=True),
        Topology.Translate(_edge(), x=-4.5, y=1.5, z=-.4, transferDictionaries=False, silent=True),
        _v(0,5,2),
    ], silent=True)
    assert Topology.IsInstance(c, "Cluster")
    return c


FACTORIES = {
    "vertex": lambda: _v(.7,-1.3,2.1), "edge": _edge, "wire": _wire,
    "face": _face, "face_with_hole": _face_hole, "shell": _shell,
    "cell": _cell, "cellcomplex": _cc, "cluster": _cluster,
}

MATRICES = {
    "translation": [[1,0,0,2.5],[0,1,0,-1.7],[0,0,1,3.2],[0,0,0,1]],
    "scale_translation": [[1.8,0,0,.6],[0,.7,0,-1.2],[0,0,1.35,2.1],[0,0,0,1]],
    "shear_xy": [[1,.45,0,.8],[0,1,0,-.4],[0,0,1,1.1],[0,0,0,1]],
    "general_shear": [[1,.30,-.15,1.2],[.20,1,.25,-.7],[-.10,.35,1,2],[0,0,0,1]],
    "reflection": [[-1,0,0,1.3],[0,1,0,-.8],[0,0,1,.4],[0,0,0,1]],
    "mixed": [[0,-1.4,.25,.9],[.8,.2,.1,-1.4],[.15,-.05,1.3,2.2],[0,0,0,1]],
}



@pytest.mark.parametrize("factory_name", list(FACTORIES))
@pytest.mark.parametrize("matrix_name", list(MATRICES))
def test_transform_deterministic_affine_oracle(factory_name, matrix_name):
    t = FACTORIES[factory_name]()
    m = MATRICES[matrix_name]
    typ, counts = Topology.TypeAsString(t), _counts(t)
    expected = [_apply(p, m) for p in _points(t)]
    r = Topology.Transform(t, m, transferDictionaries=False, tolerance=TOLERANCE, silent=True)
    assert Topology.IsInstance(r, "Topology"), f"{factory_name}/{matrix_name}"
    assert Topology.TypeAsString(r) == typ
    assert _counts(r) == counts
    _assert_points(expected, _points(r))


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_transform_identity_returns_same_object(factory_name):
    t = FACTORIES[factory_name]()
    m = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    assert Topology.Transform(t, m, silent=True) is t


@pytest.mark.parametrize("factory_name", list(FACTORIES))
def test_transform_general_shear_round_trip(factory_name):
    t = FACTORIES[factory_name]()
    m = MATRICES["general_shear"]
    inv = _inverse(m)
    typ, counts, pts = Topology.TypeAsString(t), _counts(t), _points(t)
    r = Topology.Transform(t, m, transferDictionaries=False, silent=True)
    assert Topology.IsInstance(r, "Topology")
    rr = Topology.Transform(r, inv, transferDictionaries=False, silent=True)
    assert Topology.IsInstance(rr, "Topology")
    assert Topology.TypeAsString(rr) == typ
    assert _counts(rr) == counts
    _assert_points(pts, _points(rr), tol=1e-4)


def _random_matrix(rng):
    while True:
        m = [
            [rng.uniform(.55,1.65), rng.uniform(-.45,.45), rng.uniform(-.35,.35), rng.uniform(-5,5)],
            [rng.uniform(-.45,.45), rng.uniform(.55,1.65), rng.uniform(-.35,.35), rng.uniform(-5,5)],
            [rng.uniform(-.35,.35), rng.uniform(-.35,.35), rng.uniform(.55,1.65), rng.uniform(-5,5)],
            [0,0,0,1],
        ]
        if abs(_det(m)) > .20:
            return m


@pytest.mark.parametrize("factory_name", ["vertex","edge","wire","face","face_with_hole","cell","cellcomplex","cluster"])
def test_transform_random_general_affine(factory_name):
    rng = random.Random(f"{SEED}:{factory_name}")
    cases = RANDOM_CASES
    if factory_name == "cellcomplex": cases = min(cases, 30)
    if factory_name == "cluster": cases = min(cases, 25)
    for i in range(cases):
        t = FACTORIES[factory_name]()
        m = _random_matrix(rng)
        typ, counts = Topology.TypeAsString(t), _counts(t)
        expected = [_apply(p, m) for p in _points(t)]
        r = Topology.Transform(t, m, transferDictionaries=False, tolerance=TOLERANCE, silent=True)
        assert Topology.IsInstance(r, "Topology"), f"{factory_name}/random/{i}: {m}"
        assert Topology.TypeAsString(r) == typ
        assert _counts(r) == counts
        _assert_points(expected, _points(r))


def test_transform_parent_dictionary_transfer():
    t = _cell()
    d = Dictionary.ByPythonDictionary({"transform_marker":"preserve_me", "transform_value":73})
    t = Topology.SetDictionary(t, d, silent=True)
    r = Topology.Transform(t, MATRICES["general_shear"], transferDictionaries=True, silent=True)
    assert Topology.IsInstance(r, "Cell")
    rd = Topology.Dictionary(r, silent=True)
    assert Dictionary.ValueAtKey(rd, "transform_marker", None) == "preserve_me"
    assert Dictionary.ValueAtKey(rd, "transform_value", None) == 73


@pytest.mark.parametrize("matrix", [
    None, [], [[1]], [[1,0,0],[0,1,0],[0,0,1]],
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,"bad"]],
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]],
])
def test_transform_rejects_invalid_matrix(matrix):
    assert Topology.Transform(_edge(), matrix, silent=True) is None


def test_transform_general_shear_coordinate_oracle():
    t = _cell()
    m = MATRICES["general_shear"]
    expected = [_apply(p, m) for p in _points(t)]
    r = Topology.Transform(t, m, transferDictionaries=False, silent=True)
    assert Topology.IsInstance(r, "Cell")
    _assert_points(expected, _points(r))
