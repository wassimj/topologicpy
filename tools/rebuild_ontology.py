#!/usr/bin/env python3
"""Bootstrap and normalise the TopologicPy ontology for v0.9.70.

Run once from the repository root BEFORE replacing Ontology.py, or at any later
point. The script uses the fuller v0.9.69 website TTL as bootstrap input when it
is larger than the stale root ontology. Thereafter ontology/topologicpy.ttl is
the sole human-edited source and the other copies are generated artefacts.
"""
from pathlib import Path
import shutil

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode, RDF, RDFS, OWL, XSD
    from rdflib.collection import Collection
except Exception as exc:
    raise SystemExit("RDFLib is required: pip install rdflib>=7.0") from exc

ROOT = Path.cwd()
TOP = Namespace("http://w3id.org/topologicpy#")
DCT = Namespace("http://purl.org/dc/terms/")
VANN = Namespace("http://purl.org/vocab/vann/")

LEGACY_RENAMES = {
    TOP.hasStartVertex: TOP.startsAt,
    TOP.hasEndVertex: TOP.endsAt,
    TOP.hasX: TOP.x,
    TOP.hasY: TOP.y,
    TOP.hasZ: TOP.z,
    TOP.hasLength: TOP.length,
    TOP.hasArea: TOP.area,
    TOP.hasVolume: TOP.volume,
    TOP.hasMantissa: TOP.mantissa,
    TOP.hasUnit: TOP.unit,
    TOP.hasFeature: TOP.feature,
    TOP.hasFeatureVector: TOP.featureVector,
    TOP.hasWeight: TOP.weight,
    TOP.TGraph: TOP.Graph,
}

# These concepts have standard predicates and should not also be TopologicPy
# properties. The Python dictionary keys remain valid; Ontology.py maps them to
# the external predicates during export.
REMOVE_REDUNDANT_TOP_PROPERTIES = {
    TOP.ontologyClass,   # rdf:type
    TOP.ontologyURI,     # RDF subject URI
    TOP.label,           # rdfs:label
    TOP.description,     # dcterms:description
    TOP.source,          # dcterms:source
    TOP.derivedFrom,     # prov:wasDerivedFrom
    TOP.generatedBy,     # prov:wasGeneratedBy / top:generatedByMethod
    TOP.createdAt,       # dcterms:created
    TOP.modifiedAt,      # dcterms:modified
}


def choose_bootstrap() -> Path:
    root = ROOT / "ontology" / "topologicpy.ttl"
    docs = ROOT / "docs" / "ontology" / "topologicpy.ttl"
    candidates = [p for p in (root, docs) if p.exists()]
    if not candidates:
        raise SystemExit("Could not find ontology/topologicpy.ttl or docs/ontology/topologicpy.ttl")
    # v0.9.69 has a stale short root TTL and a fuller website TTL.
    return max(candidates, key=lambda p: p.stat().st_size)


def replace_resource(g: Graph, old: URIRef, new: URIRef):
    triples = list(g.triples((None, None, None)))
    for s, p, o in triples:
        if old not in (s, p, o):
            continue
        g.remove((s, p, o))
        ns = new if s == old else s
        np = new if p == old else p
        no = new if o == old else o
        # Alias/deprecation axioms become meaningless after replacement.
        if np in {OWL.equivalentProperty, OWL.equivalentClass, OWL.deprecated}:
            continue
        if ns == no and np in {OWL.equivalentProperty, OWL.equivalentClass}:
            continue
        g.add((ns, np, no))


def remove_resource(g: Graph, resource: URIRef):
    for t in list(g.triples((resource, None, None))): g.remove(t)
    for t in list(g.triples((None, resource, None))): g.remove(t)
    for t in list(g.triples((None, None, resource))): g.remove(t)


def define_data_property(g, p, label, comment, domain=OWL.Thing, range_=XSD.string):
    g.add((p, RDF.type, OWL.DatatypeProperty))
    g.set((p, RDFS.label, Literal(label)))
    g.add((p, RDFS.domain, domain))
    g.add((p, RDFS.range, range_))
    g.set((p, RDFS.comment, Literal(comment)))


def define_object_property(g, p, label, comment, domain=OWL.Thing, range_=OWL.Thing):
    g.add((p, RDF.type, OWL.ObjectProperty))
    g.set((p, RDFS.label, Literal(label)))
    g.add((p, RDFS.domain, domain))
    g.add((p, RDFS.range, range_))
    g.set((p, RDFS.comment, Literal(comment)))


def make_union(g: Graph, members):
    node = BNode()
    head = BNode()
    g.add((node, RDF.type, OWL.Class))
    g.add((node, OWL.unionOf, head))
    Collection(g, head, list(members))
    return node


def normalise_multiple_domains_ranges(g: Graph):
    """Multiple rdfs:domain/range triples mean intersection in RDFS.

    Existing TopologicPy declarations were generally intended as 'one of these'.
    Replace multiple constraints with an explicit owl:unionOf class.
    """
    props = set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(g.subjects(RDF.type, OWL.DatatypeProperty))
    for prop in props:
        for predicate in (RDFS.domain, RDFS.range):
            values = list(dict.fromkeys(g.objects(prop, predicate)))
            if len(values) <= 1:
                continue
            # Datatype ranges should normally not need this transformation.
            if predicate == RDFS.range and all(str(v).startswith(str(XSD)) for v in values):
                continue
            for v in values: g.remove((prop, predicate, v))
            g.add((prop, predicate, make_union(g, values)))


def set_metadata(g: Graph):
    # The ontology document URI is the single ontology identity. Remove stale
    # ontology resources inherited from bootstrap files before writing metadata.
    ontology = URIRef("http://w3id.org/topologicpy")
    for subject in list(g.subjects(RDF.type, OWL.Ontology)):
        if subject != ontology:
            remove_resource(g, subject)
    g.add((ontology, RDF.type, OWL.Ontology))
    g.set((ontology, RDFS.label, Literal("TopologicPy Ontology", lang="en")))
    g.set((ontology, RDFS.comment, Literal(
        "The canonical ontology for TopologicPy topology, spatial-semantic graphs, BIM/IFC interoperability, provenance and analysis.",
        lang="en")))
    g.set((ontology, OWL.versionInfo, Literal("0.5.0")))
    g.set((ontology, OWL.versionIRI, URIRef("http://w3id.org/topologicpy/0.5.0")))
    g.set((ontology, DCT.created, Literal("2026-09-14", datatype=XSD.date)))
    g.set((ontology, DCT.modified, Literal("2026-09-14", datatype=XSD.date)))
    g.set((ontology, DCT.license, URIRef("https://www.gnu.org/licenses/lgpl-3.0.html")))
    g.set((ontology, VANN.preferredNamespacePrefix, Literal("top")))
    g.set((ontology, VANN.preferredNamespaceUri, Literal("http://w3id.org/topologicpy#")))


def main():
    source = choose_bootstrap()
    print("Bootstrap ontology:", source)
    g = Graph().parse(source, format="turtle")

    # Clean-slate canonical names.
    for old, new in LEGACY_RENAMES.items():
        replace_resource(g, old, new)
    for r in REMOVE_REDUNDANT_TOP_PROPERTIES:
        remove_resource(g, r)

    # Remove all remaining deprecation statements for top: terms. There are no
    # compatibility aliases in the 0.5.0 ontology.
    for s, p, o in list(g.triples((None, OWL.deprecated, None))):
        if str(s).startswith(str(TOP)): g.remove((s, p, o))
    for s, p, o in list(g.triples((None, OWL.equivalentProperty, None))):
        if str(s).startswith(str(TOP)) or str(o).startswith(str(TOP)):
            g.remove((s, p, o))
    for s, p, o in list(g.triples((None, OWL.equivalentClass, None))):
        if s == TOP.Graph or o == TOP.Graph:
            g.remove((s, p, o))

    # Core round-trip protocol terms.
    define_object_property(g, TOP.hasPredicate, "hasPredicate",
        "Associates a relationship record with the RDF predicate that gives the relationship its semantic meaning.",
        TOP.Relationship, RDF.Property)
    define_object_property(g, TOP.hasInversePredicate, "hasInversePredicate",
        "Associates a relationship record with an explicitly supplied inverse RDF predicate.",
        TOP.Relationship, RDF.Property)
    define_data_property(g, TOP.directed, "directed",
        "True when a graph or relationship is directed.", OWL.Thing, XSD.boolean)
    define_data_property(g, TOP.allowsSelfLoops, "allowsSelfLoops",
        "True when a graph permits self-loop relationships.", TOP.Graph, XSD.boolean)
    define_data_property(g, TOP.allowsParallelEdges, "allowsParallelEdges",
        "True when a graph permits parallel relationships between the same endpoints.", TOP.Graph, XSD.boolean)
    define_data_property(g, TOP.generatedByMethod, "generatedByMethod",
        "The TopologicPy method, function, script, or tool name that generated a resource.", OWL.Thing, XSD.string)

    # startsAt/endsAt are canonical, not aliases.
    g.set((TOP.startsAt, RDFS.comment, Literal("Associates an edge or relationship with its start vertex or source node.")))
    g.set((TOP.endsAt, RDFS.comment, Literal("Associates an edge or relationship with its end vertex or target node.")))

    # srcId/dstId are redundant with startsAt/endsAt and are removed from the
    # semantic vocabulary. top:index remains as optional graph-local ordering.
    remove_resource(g, TOP.srcId)
    remove_resource(g, TOP.dstId)

    set_metadata(g)
    normalise_multiple_domains_ranges(g)

    # Bind stable prefixes before deterministic-ish Turtle serialization.
    prefixes = {
        "top": TOP,
        "rdf": RDF, "rdfs": RDFS, "owl": OWL, "xsd": XSD,
        "dcterms": DCT, "vann": VANN,
        "prov": Namespace("http://www.w3.org/ns/prov#"),
        "bot": Namespace("https://w3id.org/bot#"),
        "brick": Namespace("https://brickschema.org/schema/Brick#"),
        "geo": Namespace("http://www.opengis.net/ont/geosparql#"),
        "ifc": Namespace("https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#"),
        "dict": Namespace("http://w3id.org/topologicpy/dictionary#"),
        "inst": Namespace("http://w3id.org/topologicpy/instance#"),
    }
    for p, ns in prefixes.items(): g.bind(p, ns, replace=True)

    canonical = ROOT / "ontology" / "topologicpy.ttl"
    package = ROOT / "src" / "topologicpy" / "ontology" / "topologicpy.ttl"
    website = ROOT / "docs" / "ontology" / "topologicpy.ttl"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    package.parent.mkdir(parents=True, exist_ok=True)
    website.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=canonical, format="turtle")
    shutil.copy2(canonical, package)
    shutil.copy2(canonical, website)
    print("Wrote canonical ontology:", canonical)
    print("Synced package copy:", package)
    print("Synced website copy:", website)
    print("Triples:", len(g))


if __name__ == "__main__":
    main()
