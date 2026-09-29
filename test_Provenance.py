import os
import pytest

from topologicpy.Cell import Cell
from topologicpy.Provenance import Provenance
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def _overlapping_prisms():
    p1 = Cell.Prism(origin=Vertex.ByCoordinates(0, 0, 0))
    p2 = Cell.Prism(origin=Vertex.ByCoordinates(0.6, 0.6, 0.6))
    return p1, p2


def test_provenance_container_accepts_materialised_records():
    p1, p2 = _overlapping_prisms()
    record = {
        "source": p1,
        "result": p2,
        "sourceType": "Cell",
        "resultType": "Cell",
        "relation": "modified",
        "sourceRole": "self",
        "operation": "Synthetic",
    }
    provenance = Provenance.ByRecords([record], operation="Synthetic")

    assert len(provenance.History()) == 1
    assert len(provenance.Records()) == 1
    assert provenance.Records()[0]["relation"] == "modified"
    assert len(provenance.Origins(p2)) == 1
    assert len(provenance.Descendants(p1)) == 1

    graph = provenance.Graph()
    assert isinstance(graph, TGraph)
    assert TGraph.Order(graph) == 2
    assert TGraph.Size(graph) == 1


@pytest.mark.skipif(
    os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower() != "pythonocc",
    reason="Exact Boolean provenance requires the pythonocc backend.",
)
def test_merge_returns_semantic_provenance_without_csg():
    pytest.importorskip("OCC.Core.BRepGraph")

    p1, p2 = _overlapping_prisms()
    p3, provenance = Topology.Merge(
        p1,
        p2,
        returnProvenance=True,
        silent=True,
    )

    assert Topology.IsInstance(p3, "Topology")
    assert isinstance(provenance, Provenance)
    assert provenance.supported is True
    assert len(provenance.History()) > 0

    faces = Topology.Faces(p3, silent=True) or []
    assert faces
    assert any(provenance.Origins(face) for face in faces)

    face_records = provenance.Records(topologyType="Face")
    assert face_records
    assert {
        str(record.get("relation", "")).lower()
        for record in face_records
    } & {"unchanged", "modified", "generated"}

    graph = provenance.Graph(topologyType="Face")
    assert isinstance(graph, TGraph)
    assert TGraph.Order(graph) > 0
    assert TGraph.Size(graph) > 0


@pytest.mark.skipif(
    os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower() != "pythonocc",
    reason="Exact Boolean provenance requires the pythonocc backend.",
)
def test_provenance_capture_does_not_require_dictionary_transfer():
    pytest.importorskip("OCC.Core.BRepGraph")

    p1, p2 = _overlapping_prisms()
    result, provenance = Topology.Difference(
        p1,
        p2,
        tranDict=False,
        returnProvenance=True,
        silent=True,
    )

    assert Topology.IsInstance(result, "Topology")
    assert provenance.History()
    assert provenance.Records()


@pytest.mark.skipif(
    os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower() != "pythonocc",
    reason="Exact Boolean provenance requires the pythonocc backend.",
)
def test_csg_lineage_graph_is_semantic_by_default_and_exact_on_request():
    pytest.importorskip("OCC.Core.BRepGraph")
    from topologicpy.CSG import CSG

    p1, p2 = _overlapping_prisms()
    csg = CSG.Init()
    a = CSG.Source(csg, p1, name="A")
    b = CSG.Source(csg, p2, name="B")
    op = CSG.Merge(csg, a, b)
    result = CSG.Evaluate(csg, op, lineage=True, silent=True)

    assert Topology.IsInstance(result, "Topology")

    semantic = CSG.LineageGraph(csg, operation=op, topologyType="Face")
    exact = CSG.LineageGraph(csg, operation=op, detailed=True)

    assert isinstance(semantic, TGraph)
    assert isinstance(exact, TGraph)
    assert TGraph.Order(semantic) > 0
    assert TGraph.Order(exact) > 0
