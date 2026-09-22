"""End-to-end conformance tests for the RDF/ontology export layer.

These complement ``test_Ontology.py`` (which unit-tests the vocabulary helpers in isolation) by exercising *real* exports against the *shipped* ontology:
* every ``top:`` term a real export emits must be declared in ``ontology/topologicpy.ttl`` (see issue #96);
* a TTL export must round-trip through :class:`KnowledgeGraph` without losing node/relationship counts, coordinates, datatypes or attributes;
* a real export must satisfy the SHACL contract for Node/Relationship/Graph;
* the ontology must answer a set of competency questions via SPARQL.

Every test degrades: tests that need ``rdflib``, ``pyshacl``, ``ifcopenshell`` or a geometry backend are skipped (not failed) when those are unavailable, matching the project's optional-dependency convention."""

import pytest

from topologicpy.Ontology import Ontology

TOP = "http://w3id.org/topologicpy#"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _declared_terms(rdflib):
    """Local names of every ``top:`` term declared in the shipped ontology."""
    from rdflib import RDF, OWL

    text = Ontology.OntologyTTLString(silent=True)
    if not text:
        pytest.skip("Shipped ontology could not be loaded")
    g = rdflib.Graph()
    g.parse(data=text, format="turtle")
    kinds = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty)
    return {
        str(s)[len(TOP):]
        for kind in kinds
        for s in g.subjects(RDF.type, kind)
        if str(s).startswith(TOP)
    }, g


def _emitted_terms(rdflib, path):
    """Local names of every ``top:`` term used in an exported TTL file."""
    from rdflib import RDF

    g = rdflib.Graph()
    g.parse(path, format="turtle")
    used = set()
    for s, p, o in g:
        if str(p).startswith(TOP):
            used.add(str(p)[len(TOP):])
        if p == RDF.type and str(o).startswith(TOP):
            used.add(str(o)[len(TOP):])
    return used, g


def _sample_graph():
    """A small semantic adjacency graph, or skip if no backend is available."""
    try:
        from topologicpy.CellComplex import CellComplex
        from topologicpy.TGraph import TGraph
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"topologicpy graph modules unavailable: {exc}")
    try:
        cc = CellComplex.Prism(width=4, length=2, height=2, uSides=3, vSides=1, wSides=1)
        tg = TGraph.ByTopology(cc, toExteriorTopologies=True)
    except Exception as exc:
        pytest.skip(f"No geometry backend available to build a graph: {exc}")
    if tg is None:
        pytest.skip("Graph construction returned None (no backend)")
    return tg


def _export(tg, tmp_path, name="graph.ttl"):
    from topologicpy.TGraph import TGraph

    path = str(tmp_path / name)
    TGraph.ExportTTL(tg, path, silent=True)
    return path


# --------------------------------------------------------------------------- #
# Conformance: exporter emits only declared terms
# --------------------------------------------------------------------------- #
def test_graph_export_emits_only_declared_terms(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    declared, _ = _declared_terms(rdflib)
    path = _export(_sample_graph(), tmp_path)
    used, _ = _emitted_terms(rdflib, path)
    undeclared = sorted(t for t in used if t not in declared)
    assert undeclared == [], f"export uses undeclared top: terms: {undeclared}"


def test_graph_export_with_unknown_dictionary_keys_stays_clean(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    from topologicpy.Dictionary import Dictionary
    from topologicpy.Topology import Topology
    from topologicpy.TGraph import TGraph

    declared, _ = _declared_terms(rdflib)
    tg = _sample_graph()
    verts = TGraph.Vertices(tg)
    if not verts:
        pytest.skip("Graph exposed no vertices")
    d = Dictionary.ByKeysValues(
        ["name", "area", "material", "cost", "unmapped_key"],
        ["Room-A", 8.0, "concrete", 1200, "xyz"],
    )
    verts[0] = Topology.SetDictionary(verts[0], d)
    tg = TGraph.ByVerticesEdges(verts, TGraph.Edges(tg))
    path = _export(tg, tmp_path, "graph_dict.ttl")
    used, _ = _emitted_terms(rdflib, path)
    undeclared = sorted(t for t in used if t not in declared)
    assert undeclared == [], f"unknown keys leaked into top:: {undeclared}"


def test_ifc_export_emits_only_declared_terms(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from ifcopenshell.api import run
    from topologicpy.IFC import IFC
    from topologicpy.TGraph import TGraph

    declared, _ = _declared_terms(rdflib)
    f = ifcopenshell.file(schema="IFC4")
    run("root.create_entity", f, ifc_class="IfcProject", name="P")
    run("unit.assign_unit", f)
    storey = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="L0")
    for i in range(3):
        w = run("root.create_entity", f, ifc_class="IfcWall", name=f"Wall-{i}")
        run("spatial.assign_container", f, relating_structure=storey, products=[w])
    ifc_path = str(tmp_path / "mini.ifc")
    f.write(ifc_path)

    tg = IFC.TGraphByPath(ifc_path, silent=True)
    if tg is None:
        pytest.skip("IFC.TGraphByPath returned None")
    path = str(tmp_path / "ifc.ttl")
    TGraph.ExportTTL(tg, path, silent=True)
    used, _ = _emitted_terms(rdflib, path)
    undeclared = sorted(t for t in used if t not in declared)
    assert undeclared == [], f"IFC export uses undeclared top: terms: {undeclared}"


# --------------------------------------------------------------------------- #
# Round-trip fidelity through KnowledgeGraph
# --------------------------------------------------------------------------- #
def test_export_round_trips_through_knowledgegraph(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    try:
        from topologicpy.KnowledgeGraph import KnowledgeGraph
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"KnowledgeGraph unavailable: {exc}")

    path = _export(_sample_graph(), tmp_path)
    original = rdflib.Graph()
    original.parse(path, format="turtle")

    kg = KnowledgeGraph.ByTTL(path, silent=True)
    if kg is None:
        pytest.skip("KnowledgeGraph.ByTTL returned None")
    restored = KnowledgeGraph.RDFGraph(kg)

    # triple count preserved
    assert len(restored) == len(original)

    # node and relationship counts preserved
    from rdflib import RDF

    def count(graph, cls):
        return len(set(graph.subjects(RDF.type, rdflib.URIRef(TOP + cls))))

    assert count(restored, "Node") == count(original, "Node")
    assert count(restored, "Relationship") == count(original, "Relationship")

    # literal values and their datatypes preserved
    def literals(graph):
        return {(str(o), o.datatype) for _, _, o in graph if isinstance(o, rdflib.Literal)}

    assert literals(restored) == literals(original)


# --------------------------------------------------------------------------- #
# SHACL contract
# --------------------------------------------------------------------------- #
SHACL_SHAPES = """
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix top: <http://w3id.org/topologicpy#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

top:NodeShape a sh:NodeShape ;
  sh:targetClass top:Node ;
  sh:property [ sh:path top:x ; sh:datatype xsd:double ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:y ; sh:datatype xsd:double ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:z ; sh:datatype xsd:double ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:index ; sh:datatype xsd:integer ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:category ; sh:minCount 1 ] .

top:RelationshipShape a sh:NodeShape ;
  sh:targetClass top:Relationship ;
  sh:property [ sh:path top:startsAt ; sh:class top:Node ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:endsAt   ; sh:class top:Node ; sh:minCount 1 ; sh:maxCount 1 ] ;
  sh:property [ sh:path top:hasPredicate ; sh:minCount 1 ] ;
  sh:property [ sh:path top:category ; sh:minCount 1 ] .

top:GraphShape a sh:NodeShape ;
  sh:targetClass top:Graph ;
  sh:property [ sh:path top:hasNode ; sh:minCount 1 ] ;
  sh:property [ sh:path top:hasRelationship ; sh:minCount 1 ] .
"""


def test_export_satisfies_shacl_contract(tmp_path):
    pytest.importorskip("rdflib")
    pyshacl = pytest.importorskip("pyshacl")
    import rdflib

    path = _export(_sample_graph(), tmp_path)
    data = rdflib.Graph()
    data.parse(path, format="turtle")
    shapes = rdflib.Graph()
    shapes.parse(data=SHACL_SHAPES, format="turtle")

    conforms, _, report = pyshacl.validate(data, shacl_graph=shapes, inference="rdfs")
    assert conforms, report


# --------------------------------------------------------------------------- #
# Competency questions
# --------------------------------------------------------------------------- #
PREFIXES = (
    "PREFIX top: <http://w3id.org/topologicpy#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
)


def test_competency_questions_over_graph_export(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    path = _export(_sample_graph(), tmp_path)
    g = rdflib.Graph()
    g.parse(path, format="turtle")

    def ask_count(sparql):
        return int(list(g.query(PREFIXES + sparql))[0][0])

    n_nodes = ask_count("SELECT (COUNT(?n) AS ?c) WHERE {?n a top:Node}")
    n_rels = ask_count("SELECT (COUNT(?r) AS ?c) WHERE {?r a top:Relationship}")
    assert n_nodes > 0
    assert n_rels > 0

    # every relationship connects two declared nodes
    dangling = list(
        g.query(
            PREFIXES
            + "SELECT ?r WHERE { ?r a top:Relationship ; top:startsAt ?a ; top:endsAt ?b ."
            "  FILTER NOT EXISTS { ?a a top:Node } }"
        )
    )
    assert dangling == [], "relationship endpoint is not a declared Node"

    # adjacency is queryable and symmetric traversal returns neighbours
    neighbours = list(
        g.query(
            PREFIXES
            + "SELECT DISTINCT ?other WHERE {"
            "  ?r top:startsAt ?a ; top:endsAt ?b . ?a top:index 0 . ?b top:index ?other }"
        )
    )
    assert len(neighbours) > 0, "node 0 has no queryable neighbours"


def test_competency_questions_over_ifc_export(tmp_path):
    rdflib = pytest.importorskip("rdflib")
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from ifcopenshell.api import run
    from topologicpy.IFC import IFC
    from topologicpy.TGraph import TGraph

    f = ifcopenshell.file(schema="IFC4")
    run("root.create_entity", f, ifc_class="IfcProject", name="P")
    run("unit.assign_unit", f)
    for i in range(3):
        run("root.create_entity", f, ifc_class="IfcWall", name=f"Wall-{i}")
    ifc_path = str(tmp_path / "mini.ifc")
    f.write(ifc_path)

    tg = IFC.TGraphByPath(ifc_path, silent=True)
    if tg is None:
        pytest.skip("IFC.TGraphByPath returned None")
    path = str(tmp_path / "ifc.ttl")
    TGraph.ExportTTL(tg, path, silent=True)
    g = rdflib.Graph()
    g.parse(path, format="turtle")

    walls = list(g.query(PREFIXES + "SELECT ?w WHERE {?w a top:Wall}"))
    assert len(walls) == 3, f"expected 3 walls, got {len(walls)}"

    # every wall exposes a canonical IFC class and GUID
    typed = list(
        g.query(
            PREFIXES
            + "SELECT ?w WHERE { ?w a top:Wall ; top:ifcClass ?c ; top:ifcGUID ?guid }"
        )
    )
    assert len(typed) == 3


# --------------------------------------------------------------------------- #
# Ontology self-consistency (offline, no backend needed)
# --------------------------------------------------------------------------- #
def test_shipped_ontology_parses_and_generated_triples_are_declared():
    rdflib = pytest.importorskip("rdflib")
    declared, _ = _declared_terms(rdflib)
    # every top: subject the generator emits is declared in the parsed ontology
    generated_subjects = {
        s[len("top:"):]
        for s, _, _ in Ontology.OntologyTriples()
        if isinstance(s, str) and s.startswith("top:")
    }
    missing = sorted(t for t in generated_subjects if t not in declared)
    assert missing == [], f"generator emits undeclared subjects: {missing}"


def test_shipped_ontology_has_no_dangling_domain_or_range():
    """Every top: class used as an rdfs:domain/range must be declared.
    Guards against the "untyped class" pitfall for internal terms without
    needing a network call to an external pitfall scanner.
    """
    rdflib = pytest.importorskip("rdflib")
    from rdflib import RDFS, OWL

    _, g = _declared_terms(rdflib)
    classes = set(g.subjects(rdflib.RDF.type, OWL.Class))
    dangling = []
    for prop in (RDFS.domain, RDFS.range):
        for subj, obj in g.subject_objects(prop):
            if str(obj).startswith(TOP) and obj not in classes:
                dangling.append((str(subj)[len(TOP):], str(obj)[len(TOP):]))
    assert dangling == [], f"dangling top: domain/range targets: {dangling}"


def test_shipped_ontology_is_coherent():
    rdflib = pytest.importorskip("rdflib")
    owlrl = pytest.importorskip("owlrl")
    from rdflib import RDF, OWL

    text = Ontology.OntologyTTLString(silent=True)
    if not text:
        pytest.skip("Shipped ontology could not be loaded")
    g = rdflib.Graph()
    g.parse(data=text, format="turtle")

    equated_to_nothing = [s for s, p, o in g if p == OWL.equivalentClass and o == OWL.Nothing]
    assert equated_to_nothing == [], "a class is declared equivalent to owl:Nothing"

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    unsatisfiable = [s for s, p, o in g if p == RDF.type and o == OWL.Nothing]
    assert unsatisfiable == [], f"OWL-RL closure entailed owl:Nothing members: {unsatisfiable}"
