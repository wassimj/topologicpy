import pytest


@pytest.fixture()
def rs():
    import topologicpy.Reasoner as reasoner_module
    return reasoner_module.Reasoner


def _rdflib_available(rs):
    report = rs.Dependencies()
    return bool(report.get("rdflib", {}).get("available", False))


def _triple_strings(graph):
    return {(str(s), str(p), str(o)) for s, p, o in graph}


def test_namespaces_qnames_and_dependency_report(rs):
    ns = rs.Namespaces()
    assert ns["rdf"].endswith("rdf-syntax-ns#")
    assert ns["rdfs"].endswith("rdf-schema#")
    assert ns["top"].startswith("http://w3id.org/topologicpy")
    assert rs.ExpandQName("top:Graph").endswith("topologicpy#Graph")
    assert rs.QName(rs.ExpandQName("top:Graph")) == "top:Graph"

    report = rs.Dependencies()
    assert set(report) == {"rdflib", "owlrl", "pyshacl"}
    assert isinstance(report["rdflib"], dict)


def test_rdf_graph_by_triples_and_inference_contract(rs):
    triples = [
        ("inst:room", "rdf:type", "top:Room"),
        ("top:Room", "rdfs:subClassOf", "top:Space"),
        ("top:Space", "rdfs:subClassOf", "top:Topology"),
        ("top:touches", "rdfs:subPropertyOf", "top:relatedTo"),
        ("inst:a", "top:touches", "inst:b"),
    ]
    before = rs.RDFGraphByTriples(triples, silent=True)

    if not _rdflib_available(rs):
        assert before is None
        return

    assert before is not None
    after = rs.Infer(before, includeOntologyAxioms=False, inplace=False, silent=True)
    assert after is not None
    assert "top:Space" in rs.Types(after, "inst:room")
    assert "top:Topology" in rs.Types(after, "inst:room")
    assert "top:Topology" in rs.SuperClasses(after, "top:Room")

    diff = rs.Difference(before, after, compact=True)
    assert ("inst:a", "top:relatedTo", "inst:b") in diff
    summary = rs.Summary(before, after)
    assert summary["output_triples"] >= summary["input_triples"]


def test_rdf_serialization_export_and_validation_contract(tmp_path, rs):
    graph = rs.RDFGraphByTriples([( "inst:a", "rdf:type", "top:Vertex")], silent=True)
    if not _rdflib_available(rs):
        assert graph is None
        assert rs.TurtleString(graph, silent=True) is None
        return

    ttl = rs.TurtleString(graph, silent=True)
    assert isinstance(ttl, str)
    assert ttl.strip()

    path = tmp_path / "graph.ttl"
    assert rs.ExportRDF(graph, str(path), overwrite=True, silent=True) == str(path)
    assert path.exists() and path.stat().st_size > 0

    validation = rs.Validate(graph, silent=True)
    assert isinstance(validation, dict)
    assert {"available", "conforms", "results_graph", "results_text"}.issubset(validation)


def test_add_ontology_axioms_public_contract(rs):
    graph = rs.RDFGraphByTriples([], silent=True)
    if not _rdflib_available(rs):
        assert graph is None
        return

    out = rs.AddOntologyAxioms(graph, includeBOT=True, silent=True)
    assert out is graph
    triples = _triple_strings(graph)
    assert any(p.endswith("subClassOf") for _, p, _ in triples)


def test_invalid_public_reasoner_inputs_fail_gracefully(rs, tmp_path):
    assert rs.RDFGraphByTopology(None, silent=True) is None
    assert rs.Infer(None, silent=True) is None
    assert rs.TurtleString(None, silent=True) is None
    assert rs.ExportRDF(None, str(tmp_path / "bad.ttl"), silent=True) is None
