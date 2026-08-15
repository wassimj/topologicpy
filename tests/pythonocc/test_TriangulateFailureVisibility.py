import pytest

from topologicpy.Core import Core
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology


TOLERANCE = 0.0001


def _vertex(x=0.0, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _edge():
    e = Edge.ByVertices(
        [_vertex(0, 0, 0), _vertex(1, 0, 0)],
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(e, "Edge")
    return e


def _wire():
    w = Wire.ByVertices(
        [
            _vertex(0, 0, 0),
            _vertex(1, 0, 0),
            _vertex(1, 1, 0),
            _vertex(0, 1, 0),
        ],
        close=True,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(w, "Wire")
    return w


def _face():
    f = Face.ByWire(
        _wire(),
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(f, "Face")
    return f


def _cell():
    c = Cell.Prism(
        origin=_vertex(0, 0, 0),
        width=2,
        length=2,
        height=2,
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
        origin=_vertex(0, 0, 0),
        width=2,
        length=2,
        height=2,
        uSides=2,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
    )
    assert Topology.IsInstance(cc, "CellComplex")
    return cc


@pytest.fixture(scope="session", autouse=True)
def _pythonocc_backend_only():
    backend = Core.Backend()
    assert backend is not None
    assert backend.__class__.__name__ == "PythonOCCBackend", (
        "test_TriangulateFailureVisibility.py must run using PythonOCCBackend. "
        f"Active backend: {backend.__class__.__name__}"
    )


@pytest.mark.parametrize(
    "factory",
    [_vertex, _edge, _wire],
)
def test_no_face_types_remain_legitimate_noops(factory):
    source = factory()
    result = Topology.Triangulate(source, silent=True)

    assert result is source


def test_cluster_constituent_query_failure_returns_none(monkeypatch):
    cluster = Cluster.ByTopologies(
        [_face(), _edge()],
        silent=True,
    )
    assert Topology.IsInstance(cluster, "Cluster")

    original_instance_call = Core.InstanceCall

    def _instance_call(topology, method_name, *args, **kwargs):
        if method_name == "Topologies":
            raise RuntimeError("intentional Topologies query failure")
        return original_instance_call(topology, method_name, *args, **kwargs)

    monkeypatch.setattr(
        Core,
        "InstanceCall",
        staticmethod(_instance_call),
    )

    monkeypatch.setattr(
        Cluster,
        "Topologies",
        staticmethod(lambda *args, **kwargs: []),
    )

    result = Topology.Triangulate(
        cluster,
        silent=True,
    )

    assert result is None


@pytest.mark.parametrize(
    "factory",
    [_face, _shell, _cell, _cellcomplex],
)
def test_higher_dimensional_face_query_failure_returns_none(monkeypatch, factory):
    source = factory()

    monkeypatch.setattr(
        Topology,
        "Faces",
        staticmethod(lambda *args, **kwargs: []),
    )

    result = Topology.Triangulate(
        source,
        silent=True,
    )

    assert result is None


def test_normal_face_triangulation_still_works():
    source = Face.ByVertices(
        [
            _vertex(0, 0, 0),
            _vertex(2, 0, 0),
            _vertex(2, 1, 0),
            _vertex(1, 2, 0),
            _vertex(0, 1, 0),
        ],
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(source, "Face")

    result = Topology.Triangulate(
        source,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")
    assert len(Topology.Faces(result, silent=True) or []) >= 2


def test_normal_cellcomplex_triangulation_preserves_cell_count():
    source = _cellcomplex()
    expected = len(Topology.Cells(source, silent=True) or [])

    result = Topology.Triangulate(
        source,
        silent=True,
    )

    assert Topology.IsInstance(result, "CellComplex")
    assert len(Topology.Cells(result, silent=True) or []) == expected
