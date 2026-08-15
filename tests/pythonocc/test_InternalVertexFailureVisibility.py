import math

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


def _vertex(x, y=0.0, z=0.0):
    v = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(v, "Vertex")
    return v


def _coords(v):
    return (
        float(Vertex.X(v, mantissa=9)),
        float(Vertex.Y(v, mantissa=9)),
        float(Vertex.Z(v, mantissa=9)),
    )


def _point_in_polygon_xy(point, polygon):
    """Independent strict 2D point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        dx = x2 - x1
        dy = y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx

        if abs(cross) <= 1.0e-9:
            dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
            if dot <= 1.0e-9:
                return False

        if (y1 > y) != (y2 > y):
            xinters = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xinters > x:
                inside = not inside

    return inside


def _concave_l_wire():
    points = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 1.0),
        (1.0, 1.0),
        (1.0, 4.0),
        (0.0, 4.0),
    ]

    wire = Wire.ByVertices(
        [_vertex(x, y, 0.0) for x, y in points],
        close=True,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(wire, "Wire")
    return wire, points


def _holed_face():
    outer = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.0),
        width=6.0,
        length=6.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    inner = Wire.Rectangle(
        origin=_vertex(0.0, 0.0, 0.0),
        width=2.0,
        length=2.0,
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


def _cell():
    cell = Cell.Prism(
        origin=_vertex(0.0, 0.0, 0.0),
        width=4.0,
        length=3.0,
        height=2.0,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(cell, "Cell")
    return cell


def _cellcomplex():
    cc = CellComplex.Prism(
        origin=_vertex(0.0, 0.0, 0.0),
        width=4.0,
        length=2.0,
        height=2.0,
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
        "test_InternalVertexFailureVisibility_REVISED.py must run using "
        "PythonOCCBackend. "
        f"Active backend: {backend.__class__.__name__}"
    )


def test_vertex_returns_same_location():
    source = _vertex(1.25, -2.5, 3.75)
    result = Topology._InternalVertex(source, silent=True)

    assert Topology.IsInstance(result, "Vertex")
    assert math.dist(_coords(result), _coords(source)) <= 1.0e-9


def test_edge_returns_interior_midpoint():
    start = _vertex(-2.0, 1.0, 0.5)
    end = _vertex(4.0, 3.0, 2.5)

    edge = Edge.ByVertices(
        [start, end],
        tolerance=TOLERANCE,
        silent=True,
    )

    result = Topology._InternalVertex(edge, silent=True)

    assert Topology.IsInstance(result, "Vertex")

    expected = (1.0, 2.0, 1.5)

    # Check against geometric/API tolerance rather than unrealistic
    # sub-micrometre equality through a kernel parameterization pipeline.
    assert math.dist(_coords(result), expected) <= 1.0e-5

    assert Topology.ShortestDistance(
        result,
        edge,
        mantissa=9,
        tolerance=TOLERANCE,
        silent=True,
    ) == 0.0

    # It must be in the edge interior, not either endpoint.
    assert math.dist(_coords(result), _coords(start)) > 0.1
    assert math.dist(_coords(result), _coords(end)) > 0.1


def test_open_wire_returns_point_on_wire():
    wire = Wire.ByVertices(
        [
            _vertex(0.0, 0.0, 0.0),
            _vertex(2.0, 0.0, 0.0),
            _vertex(2.0, 3.0, 0.0),
        ],
        close=False,
        tolerance=TOLERANCE,
        silent=True,
    )

    result = Topology._InternalVertex(wire, silent=True)

    assert Topology.IsInstance(result, "Vertex")
    assert Topology.ShortestDistance(
        result,
        wire,
        mantissa=9,
        silent=True,
    ) == 0.0


def test_concave_closed_wire_returns_strictly_internal_point():
    wire, polygon = _concave_l_wire()

    result = Topology._InternalVertex(
        wire,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(result, "Vertex")

    x, y, z = _coords(result)

    assert abs(z) <= 1.0e-7
    assert _point_in_polygon_xy((x, y), polygon)


def test_closed_wire_face_construction_failure_returns_none(monkeypatch):
    wire, _ = _concave_l_wire()

    def _raise(*args, **kwargs):
        raise RuntimeError("intentional Face construction failure")

    monkeypatch.setattr(
        Core.Face,
        "ByExternalInternalBoundaries",
        staticmethod(_raise),
    )

    result = Topology._InternalVertex(
        wire,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert result is None


def test_face_internal_vertex_is_strictly_inside_concave_face():
    wire, polygon = _concave_l_wire()

    face = Face.ByWire(
        wire,
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsInstance(face, "Face")

    result = Topology._InternalVertex(
        face,
        silent=True,
    )

    assert Topology.IsInstance(result, "Vertex")

    x, y, _ = _coords(result)
    assert _point_in_polygon_xy((x, y), polygon)


def test_holed_face_internal_vertex_avoids_hole_and_boundaries():
    face = _holed_face()

    result = Topology._InternalVertex(
        face,
        silent=True,
    )

    assert Topology.IsInstance(result, "Vertex")

    x, y, z = _coords(result)

    assert abs(z) <= 1.0e-7

    # Outer square is [-3, 3]^2.
    assert -3.0 < x < 3.0
    assert -3.0 < y < 3.0

    # Hole is [-1, 1]^2; strict internal point must not be in/on it.
    assert not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0)

    for edge in Topology.Edges(face, silent=True) or []:
        d = Topology.ShortestDistance(
            result,
            edge,
            mantissa=9,
            tolerance=TOLERANCE,
            silent=True,
        )
        assert d is not None
        assert d > TOLERANCE


def test_cell_internal_vertex_is_inside_cell():
    cell = _cell()
    result = Topology._InternalVertex(cell, silent=True)

    assert Topology.IsInstance(result, "Vertex")
    assert Vertex.IsInternal(
        result,
        cell,
        tolerance=TOLERANCE,
        silent=True,
    )


def test_cellcomplex_returns_point_inside_a_constituent_cell():
    cc = _cellcomplex()
    result = Topology._InternalVertex(cc, silent=True)

    assert Topology.IsInstance(result, "Vertex")

    cells = Topology.Cells(cc, silent=True) or []
    assert cells

    assert any(
        Vertex.IsInternal(
            result,
            cell,
            tolerance=TOLERANCE,
            silent=True,
        )
        for cell in cells
    )


def test_cluster_uses_actual_constituent_not_cluster_centroid():
    a = _vertex(-10.0, 0.0, 0.0)
    b = _vertex(10.0, 0.0, 0.0)

    cluster = Cluster.ByTopologies(
        [a, b],
        silent=True,
    )

    assert Topology.IsInstance(cluster, "Cluster")

    result = Topology._InternalVertex(
        cluster,
        silent=True,
    )

    assert Topology.IsInstance(result, "Vertex")

    p = _coords(result)

    assert (
        math.dist(p, _coords(a)) <= 1.0e-9
        or math.dist(p, _coords(b)) <= 1.0e-9
    )

    assert math.dist(p, (0.0, 0.0, 0.0)) > 1.0


def test_mixed_cluster_prefers_higher_dimensional_constituent():
    edge = Edge.ByVertices(
        [_vertex(-1.0, 0.0, 0.0), _vertex(1.0, 0.0, 0.0)],
        tolerance=TOLERANCE,
        silent=True,
    )

    far_vertex = _vertex(20.0, 0.0, 0.0)

    cluster = Cluster.ByTopologies(
        [far_vertex, edge],
        silent=True,
    )

    result = Topology._InternalVertex(
        cluster,
        silent=True,
    )

    assert Topology.IsInstance(result, "Vertex")

    assert Topology.ShortestDistance(
        result,
        edge,
        mantissa=9,
        silent=True,
    ) == 0.0


def test_cluster_query_failure_returns_none(monkeypatch):
    cluster = Cluster.ByTopologies(
        [
            _vertex(-1.0, 0.0, 0.0),
            _vertex(1.0, 0.0, 0.0),
        ],
        silent=True,
    )

    original_instance_call = Core.InstanceCall

    def _instance_call(topology, method_name, *args, **kwargs):
        if method_name == "Topologies":
            raise RuntimeError("intentional Cluster query failure")
        return original_instance_call(
            topology,
            method_name,
            *args,
            **kwargs,
        )

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

    result = Topology._InternalVertex(
        cluster,
        silent=True,
    )

    assert result is None


@pytest.mark.parametrize(
    "invalid",
    [None, 1, 3.14, "not a topology", [], {}],
)
def test_invalid_input_returns_none(invalid):
    assert Topology._InternalVertex(
        invalid,
        silent=True,
    ) is None
