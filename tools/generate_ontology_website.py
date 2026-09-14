#!/usr/bin/env python3
"""Generate/update the GitHub Pages ontology documentation from canonical TTL.

Run from the TopologicPy repository root after tools/rebuild_ontology.py.
"""
from pathlib import Path
from html import escape
import re
import shutil

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
except Exception as exc:
    raise SystemExit("RDFLib is required: pip install rdflib>=7.0") from exc

ROOT = Path.cwd()
TTL = ROOT / "ontology" / "topologicpy.ttl"
DOCS = ROOT / "docs"
TOP = Namespace("http://w3id.org/topologicpy#")
DCT = Namespace("http://purl.org/dc/terms/")


def qname(g, value):
    if value is None:
        return "—"
    try:
        return g.namespace_manager.normalizeUri(value)
    except Exception:
        return str(value)


def literals(g, subject, predicate):
    return [str(x) for x in g.objects(subject, predicate)]


def first(g, subject, predicate, default=""):
    return next((str(x) for x in g.objects(subject, predicate)), default)


def term_local(uri):
    text = str(uri)
    return text[len(str(TOP)):] if text.startswith(str(TOP)) else text


def term_card(g, term, kind):
    local = term_local(term)
    label = first(g, term, RDFS.label, local)
    comment = first(g, term, RDFS.comment, "")
    domains = [qname(g, x) for x in g.objects(term, RDFS.domain)]
    ranges = [qname(g, x) for x in g.objects(term, RDFS.range)]
    supers = [qname(g, x) for x in g.objects(term, RDFS.subClassOf)] if kind == "Class" else []
    subprops = [qname(g, x) for x in g.objects(term, RDFS.subPropertyOf)] if kind != "Class" else []
    rows = []
    if supers: rows.append(("Subclass of", ", ".join(supers)))
    if subprops: rows.append(("Subproperty of", ", ".join(subprops)))
    if domains: rows.append(("Domain", ", ".join(domains)))
    if ranges: rows.append(("Range", ", ".join(ranges)))
    rows_html = "".join(f"<tr><th>{escape(k)}</th><td><code>{escape(v)}</code></td></tr>" for k,v in rows)
    if not rows_html:
        rows_html = "<tr><th>Semantics</th><td>Defined by its label, comment, and ontology axioms.</td></tr>"
    return f'''<article class="term-card" id="{escape(local)}">
      <div class="term-head"><div><h3>{escape(label)}</h3><code>top:{escape(local)}</code></div><span class="badge">{escape(kind)}</span></div>
      <p class="term-comment">{escape(comment) if comment else "No comment supplied."}</p>
      <table class="spec-table compact">{rows_html}</table>
    </article>'''


def generate_specification(g):
    ontology = next(g.subjects(RDF.type, OWL.Ontology), TOP.TopologicPyOntology)
    version = first(g, ontology, OWL.versionInfo, "0.5.0")
    modified = first(g, ontology, DCT.modified, "")
    classes = sorted({s for s in g.subjects(RDF.type, OWL.Class) if str(s).startswith(str(TOP))}, key=term_local)
    obj = sorted({s for s in g.subjects(RDF.type, OWL.ObjectProperty) if str(s).startswith(str(TOP))}, key=term_local)
    data = sorted({s for s in g.subjects(RDF.type, OWL.DatatypeProperty) if str(s).startswith(str(TOP))}, key=term_local)
    ann = sorted({s for s in g.subjects(RDF.type, OWL.AnnotationProperty) if str(s).startswith(str(TOP))}, key=term_local)

    cards = {
        "classes": "\n".join(term_card(g, t, "Class") for t in classes),
        "object": "\n".join(term_card(g, t, "Object property") for t in obj),
        "data": "\n".join(term_card(g, t, "Datatype property") for t in data),
        "annotation": "\n".join(term_card(g, t, "Annotation property") for t in ann),
    }
    css = r'''
    :root{--bg:#f5f5f7;--paper:#fbfbfd;--ink:#1d1d1f;--muted:#6e6e73;--line:rgba(29,29,31,.09);--blue:#0066cc;--max:1180px}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:linear-gradient(180deg,#fff 0%,var(--bg) 45%,#fff 100%);font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.55}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.nav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.82);backdrop-filter:blur(22px);border-bottom:1px solid var(--line)}.nav-inner,.wrap{max-width:var(--max);margin:auto;padding:0 1.35rem}.nav-inner{min-height:3.5rem;display:flex;align-items:center;justify-content:space-between;gap:1rem}.nav-links{display:flex;gap:1rem}.hero{text-align:center;padding:6rem 0 4rem}.eyebrow,.badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:.35rem .65rem;background:rgba(255,255,255,.75);color:var(--muted);font-size:.8rem;font-weight:650}h1{font-size:clamp(3rem,7vw,6.4rem);line-height:.94;letter-spacing:-.065em;margin:1rem auto;max-width:980px}.lead{max-width:840px;margin:auto;color:var(--muted);font-size:1.2rem}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:2rem}.card,.term-card{background:rgba(255,255,255,.80);border:1px solid rgba(255,255,255,.9);border-radius:24px;box-shadow:0 16px 44px rgba(0,0,0,.055)}.card{padding:1.4rem}.stat{font-size:2.6rem;font-weight:760;letter-spacing:-.05em}.section{padding:3.5rem 0}.section-head{text-align:center;max-width:840px;margin:0 auto 1.6rem}.section-head h2{font-size:clamp(2rem,4.5vw,4rem);line-height:1;letter-spacing:-.05em;margin:.2rem 0 .7rem}.section-head p,.muted{color:var(--muted)}.rules{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem}.rules .card h3{margin:.1rem 0 .45rem}.protocol{overflow:auto}.protocol pre{background:#1d1d1f;color:#f5f5f7;border-radius:18px;padding:1rem;overflow:auto}.term-grid{display:grid;gap:1rem}.term-card{padding:1.2rem;scroll-margin-top:4.5rem}.term-head{display:flex;justify-content:space-between;gap:1rem}.term-head h3{margin:0}.term-comment{color:var(--muted)}.spec-table{border-collapse:collapse;width:100%}.spec-table th,.spec-table td{text-align:left;vertical-align:top;border-top:1px solid var(--line);padding:.6rem}.spec-table th{width:22%;color:var(--muted)}.toc{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center;margin-top:1rem}.toc a{border:1px solid var(--line);border-radius:999px;padding:.35rem .65rem;background:#fff}footer{padding:3rem 0;border-top:1px solid var(--line);color:var(--muted)}@media(max-width:800px){.stats,.rules{grid-template-columns:1fr 1fr}.nav-links{display:none}}@media(max-width:520px){.stats,.rules{grid-template-columns:1fr}}
    '''
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TopologicPy Ontology Specification</title><meta name="description" content="Generated human-readable specification of the canonical TopologicPy ontology."><style>{css}</style></head>
<body><nav class="nav"><div class="nav-inner"><a href="index.html"><strong>TopologicPy Ontology</strong></a><div class="nav-links"><a href="#design">Design</a><a href="#classes">Classes</a><a href="#object-properties">Object properties</a><a href="#datatype-properties">Datatype properties</a><a href="ontology/topologicpy.ttl">TTL</a></div></div></nav>
<header class="hero"><div class="wrap"><span class="eyebrow">Canonical ontology · v{escape(version)}</span><h1>One vocabulary. One semantic contract.</h1><p class="lead">This specification is generated directly from <code>ontology/topologicpy.ttl</code>. The TTL is the single human-edited source of truth used by the Python API, serializers, reasoner, tests, and website.</p><div class="stats"><div class="card"><span class="stat">{len(classes)}</span><div>Classes</div></div><div class="card"><span class="stat">{len(obj)}</span><div>Object properties</div></div><div class="card"><span class="stat">{len(data)}</span><div>Datatype properties</div></div><div class="card"><span class="stat">{len(ann)}</span><div>Annotation properties</div></div></div><div class="toc"><a href="#design">Design rules</a><a href="#protocol">Graph protocol</a><a href="#classes">Classes</a><a href="#object-properties">Object properties</a><a href="#datatype-properties">Datatype properties</a></div></div></header>
<main>
<section class="section" id="design"><div class="wrap"><div class="section-head"><h2>Vocabulary design rules</h2><p>TopologicPy owns only terms it can define precisely; established external vocabularies keep their native terms.</p></div><div class="rules">
<div class="card"><h3>Attributes are nouns</h3><p class="muted">Datatype properties use concise names such as <code>top:x</code>, <code>top:area</code>, <code>top:volume</code>, and <code>top:index</code>. They do not receive a mechanical <code>has</code> prefix.</p></div>
<div class="card"><h3>Relationships are semantic verbs</h3><p class="muted">Object properties describe the relation: <code>top:startsAt</code>, <code>top:endsAt</code>, <code>brick:feeds</code>. <code>has...</code> is used only when possession or association is the relation.</p></div>
<div class="card"><h3>No undeclared top: terms</h3><p class="muted">The <code>top:</code> namespace is closed to the vocabulary declared here. Unknown Python dictionary keys serialize under <code>dict:</code>, never as accidental ontology terms.</p></div>
<div class="card"><h3>No compatibility aliases</h3><p class="muted">Ontology 0.5.0 is a clean-slate pre-release vocabulary. Legacy experimental names such as <code>hasX</code>, <code>hasArea</code>, and <code>hasStartVertex</code> are not part of the ontology.</p></div>
<div class="card"><h3>Identity is not a label</h3><p class="muted"><code>rdfs:label</code> is presentation metadata. RDF identity comes from an explicit URI/UUID/GUID or a stable graph-record fallback, never from a human-readable label.</p></div>
<div class="card"><h3>External vocabularies stay external</h3><p class="muted">RDF/RDFS/OWL, Dublin Core, PROV-O, BOT, Brick, GeoSPARQL, and IFC terms are reused without renaming them to match TopologicPy house style.</p></div>
</div></div></section>
<section class="section" id="protocol"><div class="wrap"><div class="section-head"><h2>Lossless graph relationship protocol</h2><p>A relationship record separates graph structure from semantic meaning.</p></div><div class="card protocol"><pre><code>inst:r17 a top:Relationship ;
    top:startsAt inst:n3 ;
    top:endsAt inst:n8 ;
    top:hasPredicate brick:feeds .

inst:n3 brick:feeds inst:n8 .</code></pre><p class="muted"><code>startsAt</code>/<code>endsAt</code> reconstruct the TGraph edge. <code>hasPredicate</code> records its semantic predicate explicitly. <code>hasInversePredicate</code> is emitted only when an inverse predicate was explicitly supplied.</p></div></div></section>
<section class="section" id="classes"><div class="wrap"><div class="section-head"><h2>Classes</h2><p>{len(classes)} TopologicPy classes.</p></div><div class="term-grid">{cards['classes']}</div></div></section>
<section class="section" id="object-properties"><div class="wrap"><div class="section-head"><h2>Object properties</h2><p>{len(obj)} resource-to-resource relationships.</p></div><div class="term-grid">{cards['object']}</div></div></section>
<section class="section" id="datatype-properties"><div class="wrap"><div class="section-head"><h2>Datatype properties</h2><p>{len(data)} literal-valued properties.</p></div><div class="term-grid">{cards['data']}</div></div></section>
<section class="section" id="annotation-properties"><div class="wrap"><div class="section-head"><h2>Annotation properties</h2><p>{len(ann)} annotation properties.</p></div><div class="term-grid">{cards['annotation']}</div></div></section>
</main><footer><div class="wrap">TopologicPy Ontology v{escape(version)}{(' · modified '+escape(modified)) if modified else ''}. Generated from the canonical TTL.</div></footer></body></html>'''
    (DOCS / "specification.html").write_text(html, encoding="utf-8")


def patch_index(version):
    path = DOCS / "index.html"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    replacements = {
        "top:hasArea": "top:area",
        "Ontology.ValidateGraph(graph)": "Ontology.Validate(graph)",
        "Ontology.GraphByTTLPath(": "Ontology.GraphByTTL(",
        "The ontology is no longer just a specification file. It now connects importers, topology,\n              graph construction, graph databases, GQL, GraphRAG, PyG, RDF export, RDF import, and validation.":
        "The ontology now has one canonical TTL source shared by importers, graph construction, reasoning, RDF export/import, tests, and this website.",
        "Ontology.py became the semantic API": "One canonical ontology became the semantic API",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace the old resource-version line(s) with the current canonical archive.
    text = re.sub(
        r'<div class="link-item"><a href="ontology/versions/[^"]+/topologicpy\.ttl">Version [^<]+</a><span class="small">TTL</span></div>',
        f'<div class="link-item"><a href="ontology/versions/{version}/topologicpy.ttl">Version {version}</a><span class="small">TTL</span></div>',
        text,
    )
    # Do not present ontology_uri as a semantic vocabulary key; it is an internal
    # identity control field. Keep the useful Python-facing keys.
    text = text.replace("<li>ontology_class</li><li>ontology_uri</li><li>category</li>",
                        "<li>ontology_class</li><li>uri</li><li>category</li>")
    text = text.replace("<li>label</li><li>uri</li><li>source</li>",
                        "<li>label</li><li>source</li>")
    path.write_text(text, encoding="utf-8")


def archive_version(version):
    target = DOCS / "ontology" / "versions" / version / "topologicpy.ttl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TTL, target)


def main():
    if not TTL.exists():
        raise SystemExit("Missing ontology/topologicpy.ttl; run tools/rebuild_ontology.py first")
    g = Graph().parse(TTL, format="turtle")
    ontology = next(g.subjects(RDF.type, OWL.Ontology), TOP.TopologicPyOntology)
    version = first(g, ontology, OWL.versionInfo, "0.5.0")
    DOCS.mkdir(parents=True, exist_ok=True)
    generate_specification(g)
    patch_index(version)
    archive_version(version)
    print("Updated docs/specification.html")
    print("Patched docs/index.html")
    print(f"Archived docs/ontology/versions/{version}/topologicpy.ttl")


if __name__ == "__main__":
    main()
