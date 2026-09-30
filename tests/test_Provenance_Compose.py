import pytest

from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Provenance import Provenance
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


pytestmark = pytest.mark.pythonocc_only


def _v(x, y=0.0, z=0.0):
    return Vertex.ByCoordinates(float(x), float(y), float(z))


def _roles(graph):
    counts = {}
    for vertex in TGraph.Vertices(graph):
        dictionary = vertex.get("dictionary") or {}
        role = dictionary.get("role")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _record(source, result, operation, relation="modified"):
    return {
        "source": source,
        "result": result,
        "sourceType": "Vertex",
        "resultType": "Vertex",
        "sourceRole": "self",
        "operation": operation,
        "relation": relation,
    }


def test_compose_two_provenances_stitches_shared_boundary_state():
    a = _v(0)
    x = _v(1)
    y = _v(2)

    p1 = Provenance.ByRecords(
        [_record(a, x, "First")],
        operation="First",
        sources={"A": a},
        result=x,
    )
    p2 = Provenance.ByRecords(
        [_record(x, y, "Second")],
        operation="Second",
        sources={"A": x},
        result=y,
    )

    composed = Provenance.Compose(p1, p2)

    assert composed.supported is True
    assert composed.operation == "Compose"
    assert composed.metadata.get("composed") is True
    assert composed.metadata.get("stageCount") == 2

    records = composed.Records()
    assert len(records) == 2
    assert {record.get("provenanceStage") for record in records} == {0, 1}

    graph = composed.Graph(topologyType="Vertex")
    roles = _roles(graph)

    assert TGraph.Order(graph) == 3
    assert roles.get("source", 0) == 1
    assert roles.get("intermediate", 0) == 1
    assert roles.get("result", 0) == 1


def test_compose_preserves_unchanged_states_but_stitches_next_operation():
    x = _v(0)
    y = _v(1)

    p1 = Provenance.ByRecords(
        [_record(x, x, "First", relation="unchanged")],
        operation="First",
        sources={"A": x},
        result=x,
    )
    p2 = Provenance.ByRecords(
        [_record(x, y, "Second", relation="modified")],
        operation="Second",
        sources={"A": x},
        result=y,
    )

    composed = Provenance.Compose(p1, p2)
    graph = composed.Graph(topologyType="Vertex")
    roles = _roles(graph)

    # Boundary 0 and boundary 1 are distinct provenance states even though
    # both represent the same topology. Boundary 1 is then shared with the
    # source state of the second operation.
    assert TGraph.Order(graph) == 3
    assert roles.get("source", 0) == 1
    assert roles.get("intermediate", 0) == 1
    assert roles.get("result", 0) == 1


def test_compose_mixed_unchanged_and_modified_stage_preserves_boundary():
    """Regression: composed Records() must not semantically reduce stages twice."""
    x = _v(0)
    a = _v(10)
    z = _v(11)
    y = _v(1)

    boundary = Cluster.ByTopologies([x, z])
    assert Topology.IsInstance(boundary, "Cluster")

    p1 = Provenance.ByRecords(
        [
            _record(x, x, "First", relation="unchanged"),
            _record(a, z, "First", relation="modified"),
        ],
        operation="First",
        sources={"A": Cluster.ByTopologies([x, a])},
        result=boundary,
    )
    p2 = Provenance.ByRecords(
        [_record(x, y, "Second", relation="modified")],
        operation="Second",
        sources={"A": boundary},
        result=y,
    )

    # Direct p1 semantic reconciliation restores x -> x because x is an
    # authoritative final entity of the boundary Cluster.
    assert any(
        Topology.IsSame(record.get("source"), x)
        and Topology.IsSame(record.get("result"), x)
        for record in p1.Records(topologyType="Vertex")
    )

    composed = Provenance.Compose(p1, p2)
    assert composed.metadata.get("boundaryMatches") == [1]

    records = composed.Records(topologyType="Vertex")
    assert any(
        record.get("provenanceStage") == 0
        and Topology.IsSame(record.get("result"), x)
        for record in records
    )
    assert any(
        record.get("provenanceStage") == 1
        and Topology.IsSame(record.get("source"), x)
        for record in records
    )

    graph = composed.Graph(topologyType="Vertex")
    roles = _roles(graph)

    # Stage 0 produces two boundary entities: x and z. x continues into the
    # second operation; z is a dead-end output of the first operation. Both
    # belong to the non-final boundary and are therefore intermediate states
    # in the composed graph. Only y, on the final boundary, is a result.
    assert roles.get("source", 0) == 2
    assert roles.get("intermediate", 0) == 2
    assert roles.get("result", 0) == 1


def test_compose_origins_and_descendants_are_transitive():
    a = _v(0)
    x = _v(1)
    y = _v(2)
    z = _v(3)

    p1 = Provenance.ByRecords(
        [_record(a, x, "First", relation="unchanged")],
        operation="First",
        result=x,
    )
    p2 = Provenance.ByRecords(
        [_record(x, y, "Second", relation="modified")],
        operation="Second",
        result=y,
    )
    p3 = Provenance.ByRecords(
        [_record(y, z, "Third", relation="generated")],
        operation="Third",
        result=z,
    )

    composed = Provenance.Compose(p1, p2, p3)

    origins = composed.Origins(z)
    assert len(origins) == 1
    assert Topology.IsSame(origins[0]["source"], a)
    assert Topology.IsSame(origins[0]["result"], z)
    assert origins[0]["relation"] == "generated"

    descendants = composed.Descendants(a)
    assert len(descendants) == 1
    assert Topology.IsSame(descendants[0]["source"], a)
    assert Topology.IsSame(descendants[0]["result"], z)
    assert descendants[0]["relation"] == "generated"


def test_nested_compose_preserves_all_stages():
    a = _v(0)
    x = _v(1)
    y = _v(2)
    z = _v(3)

    p1 = Provenance.ByRecords([_record(a, x, "First")], operation="First", result=x)
    p2 = Provenance.ByRecords([_record(x, y, "Second")], operation="Second", result=y)
    p3 = Provenance.ByRecords([_record(y, z, "Third")], operation="Third", result=z)

    first_two = Provenance.Compose(p1, p2)
    composed = Provenance.Compose(first_two, p3)

    records = composed.Records()
    assert len(records) == 3
    assert {record.get("provenanceStage") for record in records} == {0, 1, 2}
    assert composed.metadata.get("stageCount") == 3

    graph = composed.Graph(topologyType="Vertex")
    roles = _roles(graph)

    assert TGraph.Order(graph) == 4
    assert roles.get("source", 0) == 1
    assert roles.get("intermediate", 0) == 2
    assert roles.get("result", 0) == 1


def test_compose_real_merge_then_difference():
    p1 = Cell.Prism(origin=Vertex.ByCoordinates(0, 0, 0))
    p2 = Cell.Prism(origin=Vertex.ByCoordinates(0.6, 0.6, 0.6))

    merged, provenance_1 = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
    )
    assert Topology.IsInstance(merged, "Topology")
    assert provenance_1.supported is True

    result, provenance_2 = Topology.Difference(
        merged,
        p2,
        returnProvenance=True,
    )
    assert Topology.IsInstance(result, "Topology")
    assert provenance_2.supported is True

    composed = Provenance.Compose(provenance_1, provenance_2)
    graph = composed.Graph(topologyType="Face")
    roles = _roles(graph)

    assert composed.metadata.get("stageCount") == 2
    assert roles.get("source", 0) > 0
    assert roles.get("intermediate", 0) > 0
    assert roles.get("result", 0) == len(Topology.Faces(result, silent=True))

    # Every final Face must resolve to at least one leaf origin across the
    # composed history, not merely to an intermediate Face from the Merge.
    for face in Topology.Faces(result, silent=True):
        origins = composed.Origins(face)
        assert origins
        assert all(record.get("source") is not None for record in origins)
        assert all(Topology.IsSame(record.get("result"), face) for record in origins)
