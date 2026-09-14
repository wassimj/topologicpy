# Copyright (C) 2026
# TopologicPy
#
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3.0 of the License, or
# at your option any later version.
#
# TopologicPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import re


class Reasoner:
    """
    A lightweight semantic reasoning helper for TopologicPy.

    The Reasoner class is intentionally optional and dependency-light. It uses
    RDFLib when available, can use owlrl/pyshacl when installed, and otherwise
    provides a small built-in RDFS reasoner that supports the inferences most
    useful for TopologicPy graphs:

    - rdfs:subClassOf transitivity
    - rdf:type inheritance through superclass chains
    - rdfs:subPropertyOf transitivity
    - predicate inheritance through superproperty chains
    - rdfs:domain and rdfs:range type inference

    The class does not replace Ontology.py. It treats Ontology.py as the
    canonical source of namespaces, classes, categories, BOT mappings, object
    properties, and data properties.
    """

    # ---------------------------------------------------------------------
    # Public explainability and proof-graph API
    # ---------------------------------------------------------------------

    @staticmethod
    def Explain(resultOrGraph, subject=None, predicate=None, object=None, triple=None,
                maxDepth: int = 32, compact: bool = True):
        """
        Explains why a semantic triple is asserted or inferred.

        Parameters
        ----------
        resultOrGraph : InferenceResult, rdflib.Graph, KnowledgeGraph, or iterable
            The inference result or graph to inspect.
        subject, predicate, object : Any, optional
            The triple components to explain.
        triple : tuple, optional
            A three-item triple. If supplied, it overrides subject/predicate/object.
        maxDepth : int, optional
            The maximum proof-tree depth. Default is 32.
        compact : bool, optional
            If True, returns a compact human-readable explanation. Default is True.

        Returns
        -------
        str
            The explanation string.
        """
        if triple is not None:
            subject, predicate, object = triple
        elif isinstance(subject, (list, tuple)) and len(subject) == 3 and predicate is None and object is None:
            subject, predicate, object = subject
        return _Reasoner_Explain(resultOrGraph, subject, predicate, object, maxDepth=maxDepth, compact=compact)

    @staticmethod
    def ProofTree(resultOrGraph, subject=None, predicate=None, object=None, triple=None,
                  maxDepth: int = 32):
        """
        Returns a structured proof tree for a semantic triple.
        """
        if triple is not None:
            subject, predicate, object = triple
        elif isinstance(subject, (list, tuple)) and len(subject) == 3 and predicate is None and object is None:
            subject, predicate, object = subject
        return _Reasoner_ProofTree(resultOrGraph, subject, predicate, object, maxDepth=maxDepth)

    @staticmethod
    def ProofGraph(resultOrGraph=None, subject=None, predicate=None, object=None,
                   triple=None, **kwargs):
        """
        Returns proof-graph data for a semantic triple.
        """
        return _Reasoner_ProofGraph(resultOrGraph=resultOrGraph, subject=subject,
                                    predicate=predicate, object=object, triple=triple,
                                    **kwargs)

    @staticmethod
    def ProofGraphData(resultOrGraph=None, subject=None, predicate=None, object=None,
                       triple=None, **kwargs):
        """
        Returns Plotly-ready proof-graph data for a semantic triple.
        """
        return _Reasoner_ProofGraphData(resultOrGraph=resultOrGraph, subject=subject,
                                        predicate=predicate, object=object, triple=triple,
                                        **kwargs)

    @staticmethod
    def ProofGraphFigure(resultOrGraph=None, subject=None, predicate=None, object=None,
                         triple=None, **kwargs):
        """
        Returns a Plotly figure for a proof graph.
        """
        return _Reasoner_ProofGraphFigure(resultOrGraph=resultOrGraph, subject=subject,
                                          predicate=predicate, object=object, triple=triple,
                                          **kwargs)

    @staticmethod
    def ProofGraphHTML(resultOrGraph=None, subject=None, predicate=None, object=None,
                       triple=None, path="proof_graph.html", **kwargs):
        """
        Exports a proof-graph visualisation to an HTML file.
        """
        return _Reasoner_ProofGraphHTML(resultOrGraph=resultOrGraph, subject=subject,
                                        predicate=predicate, object=object, triple=triple,
                                        path=path, **kwargs)

    @staticmethod
    def ProofTGraph(resultOrGraph=None, subject=None, predicate=None, object=None,
                    triple=None, **kwargs):
        """
        Returns a proof graph as a TGraph.
        """
        return _Reasoner_ProofTGraph(resultOrGraph=resultOrGraph, subject=subject,
                                     predicate=predicate, object=object, triple=triple,
                                     **kwargs)

    @staticmethod
    def RuleStatistics(resultOrGraph):
        """
        Returns counts grouped by inference rule.
        """
        return _Reasoner_RuleStatistics(resultOrGraph)

    @staticmethod
    def Diff(beforeGraph, afterGraph=None, compact: bool = True, limit: int = None):
        """
        Returns a semantic diff between graphs or, for an InferenceResult, the inferred triples.
        """
        return _Reasoner_Diff(beforeGraph, afterGraph=afterGraph, compact=compact, limit=limit)

    # ---------------------------------------------------------------------
    # Dependency and import helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def Dependencies() -> Dict[str, Any]:
        """
        Reports optional semantic-reasoning dependencies.

        Returns
        -------
        dict
            Dependency availability and versions.
        """
        report = {}
        for name in ["rdflib", "owlrl", "pyshacl"]:
            try:
                module = __import__(name)
                report[name] = {"available": True, "version": getattr(module, "__version__", None)}
            except Exception as exc:
                report[name] = {"available": False, "error": str(exc)}
        return report

    @staticmethod
    def _ontology_class():
        try:
            from topologicpy.Ontology import Ontology
            return Ontology
        except Exception:
            try:
                from Ontology import Ontology
                return Ontology
            except Exception:
                return None

    @staticmethod
    def _tgraph_class():
        try:
            from topologicpy.TGraph import TGraph
            return TGraph
        except Exception:
            try:
                from TGraph import TGraph
                return TGraph
            except Exception:
                return None

    @staticmethod
    def _rdflib(silent: bool = False):
        try:
            import rdflib
            from rdflib import Graph, Namespace, URIRef, Literal, BNode
            from rdflib.namespace import RDF, RDFS, OWL, XSD
            return {
                "rdflib": rdflib,
                "Graph": Graph,
                "Namespace": Namespace,
                "URIRef": URIRef,
                "Literal": Literal,
                "BNode": BNode,
                "RDF": RDF,
                "RDFS": RDFS,
                "OWL": OWL,
                "XSD": XSD,
            }
        except Exception as exc:
            if not silent:
                print("Reasoner - Error: RDFLib is required. Install with 'pip install rdflib'.")
                print("Error:", exc)
            return None

    # ---------------------------------------------------------------------
    # URI/QName helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def Namespaces() -> Dict[str, str]:
        """Returns the namespace dictionary used by TopologicPy reasoning.

        Ontology.py is the authoritative source for TopologicPy namespaces. A
        minimal standards-only fallback is retained so Reasoner can still inspect
        arbitrary RDF graphs when Ontology.py is unavailable.
        """
        Ontology = Reasoner._ontology_class()
        if Ontology is not None:
            try:
                return dict(Ontology.Namespaces())
            except Exception:
                try:
                    return dict(Ontology.NAMESPACES)
                except Exception:
                    pass

        return {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dcterms": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
            "bot": "https://w3id.org/bot#",
            "brick": "https://brickschema.org/schema/Brick#",
            "geo": "http://www.opengis.net/ont/geosparql#",
            "ifc": "https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#",
            "top": "http://w3id.org/topologicpy#",
            "dict": "http://w3id.org/topologicpy/dictionary#",
            "inst": "http://w3id.org/topologicpy/instance#",
        }

    @staticmethod
    def ExpandQName(term: Any, defaultValue: Any = None):
        """Expands a QName, bracketed URI, or absolute URI to a URI string."""
        if term is None:
            return defaultValue

        Ontology = Reasoner._ontology_class()
        if Ontology is not None:
            try:
                return Ontology.ExpandQName(term, defaultValue=defaultValue)
            except Exception:
                pass

        text = str(term).strip()
        if text == "":
            return defaultValue
        if text.startswith("<") and text.endswith(">"):
            return text[1:-1]
        if text.startswith(("http://", "https://", "urn:")):
            return text
        if ":" not in text:
            return defaultValue
        prefix, local = text.split(":", 1)
        ns = Reasoner.Namespaces().get(prefix)
        if ns is None or local == "":
            return defaultValue
        return ns + local

    @staticmethod
    def QName(uri: Any, defaultValue: Any = None) -> Any:
        """Compacts a URIRef or URI string to a QName when a namespace matches."""
        if uri is None:
            return defaultValue

        Ontology = Reasoner._ontology_class()
        if Ontology is not None:
            try:
                return Ontology.QName(uri, defaultValue=defaultValue)
            except TypeError:
                try:
                    return Ontology.QName(uri, defaultValue)
                except Exception:
                    pass
            except Exception:
                pass

        text = str(uri).strip()
        if text.startswith("<") and text.endswith(">"):
            text = text[1:-1]
        for prefix, ns in sorted(Reasoner.Namespaces().items(), key=lambda item: len(item[1]), reverse=True):
            if text.startswith(ns):
                local = text[len(ns):]
                if local:
                    return prefix + ":" + local
        return defaultValue if defaultValue is not None else text

    @staticmethod
    def _uri_ref(term: Any, URIRef=None):
        """Returns an rdflib.URIRef for a QName, bracketed URI, or absolute URI."""
        if URIRef is None:
            rd = Reasoner._rdflib(silent=True)
            if rd is None:
                return None
            URIRef = rd["URIRef"]
        if term is None:
            return None
        if isinstance(term, URIRef):
            return term
        text = str(term).strip()
        if text.startswith("<") and text.endswith(">"):
            return URIRef(text[1:-1])
        expanded = Reasoner.ExpandQName(text, defaultValue=None)
        if expanded is not None:
            return URIRef(expanded)
        return URIRef(text)

    @staticmethod
    def _bind_namespaces(rdfGraph):
        if rdfGraph is None:
            return None
        try:
            from rdflib import Namespace
            for prefix, uri in Reasoner.Namespaces().items():
                rdfGraph.bind(prefix, Namespace(uri))
        except Exception:
            pass
        return rdfGraph

    # ---------------------------------------------------------------------
    # RDF graph creation and ontology axiom injection
    # ---------------------------------------------------------------------

    @staticmethod
    def RDFGraphByTopology(
        topology: Any,
        includeGraph: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        includeOntologyAxioms: bool = True,
        namespacePrefix: str = "inst",
        instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
        silent: bool = False,
    ):
        """Returns an RDFLib graph using Ontology.py as the sole serializer."""
        if topology is None:
            return None

        Ontology = Reasoner._ontology_class()
        if Ontology is None:
            if not silent:
                print("Reasoner.RDFGraphByTopology - Error: Ontology.py is required. Returning None.")
            return None

        try:
            g = Ontology.RDFGraph(
                topology,
                includeGraph=includeGraph,
                includeDictionaries=includeDictionaries,
                includeBOT=includeBOT,
                namespacePrefix=namespacePrefix,
                instanceNamespace=instanceNamespace,
                silent=silent,
            )
        except Exception as exc:
            if not silent:
                print("Reasoner.RDFGraphByTopology - Error: Could not create the RDF graph. Returning None.")
                print("Error:", exc)
            return None

        if g is None:
            return None
        Reasoner._bind_namespaces(g)
        if includeOntologyAxioms:
            Reasoner.AddOntologyAxioms(g, includeBOT=includeBOT, silent=silent)
        return g


    @staticmethod
    def TriplesByTopology(
        topology: Any,
        includeGraph: bool = True,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
        silent: bool = False,
    ) -> List[Tuple[str, str, str]]:
        """Returns canonical triples using Ontology.py as the sole vocabulary authority."""
        if topology is None:
            return []

        Ontology = Reasoner._ontology_class()
        if Ontology is None:
            if not silent:
                print("Reasoner.TriplesByTopology - Error: Ontology.py is required. Returning an empty list.")
            return []

        try:
            is_graph_like = bool(Ontology._is_graph_like(topology))
        except Exception:
            is_graph_like = False

        try:
            if is_graph_like:
                return Ontology.GraphTriples(
                    topology,
                    includeVertices=includeGraph,
                    includeEdges=includeGraph,
                    includeDictionaries=includeDictionaries,
                    includeBOT=includeBOT,
                    namespacePrefix=namespacePrefix,
                    silent=silent,
                )
            return Ontology.Triples(
                topology,
                includeDictionaries=includeDictionaries,
                includeBOT=includeBOT,
                namespacePrefix=namespacePrefix,
                silent=silent,
            )
        except Exception as exc:
            if not silent:
                print("Reasoner.TriplesByTopology - Error: Could not create triples. Returning an empty list.")
                print("Error:", exc)
            return []

    @staticmethod
    def RDFGraphByTriples(triples: Iterable[Tuple[Any, Any, Any]], silent: bool = False):
        """Builds an RDFLib graph while preserving explicit RDF vocabulary.

        Explicit QNames and URIs retain their namespace exactly. Bare predicate
        names are treated as Python dictionary keys and are mapped through
        Ontology.PropertyQName, which places undeclared keys under ``dict:``.
        """
        rd = Reasoner._rdflib(silent=silent)
        if rd is None:
            return None

        g = rd["Graph"]()
        Reasoner._bind_namespaces(g)
        URIRef = rd["URIRef"]
        Literal = rd["Literal"]
        BNode = rd["BNode"]
        Ontology = Reasoner._ontology_class()
        namespaces = Reasoner.Namespaces()

        try:
            from rdflib.util import from_n3
        except Exception:
            from_n3 = None

        def safe_local(value):
            text = "" if value is None else str(value).strip()
            text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
            text = re.sub(r"_+", "_", text).strip("_") or "item"
            if text[0].isdigit():
                text = "id_" + text
            return text

        def explicit_resource(value):
            if value is None:
                return None
            text = str(value).strip()
            if text.startswith("_:"):
                return BNode(text[2:])
            if text.startswith("<") and text.endswith(">"):
                return URIRef(text[1:-1])
            if text.startswith(("http://", "https://", "urn:")):
                return URIRef(text)
            if ":" in text:
                prefix, local = text.split(":", 1)
                ns = namespaces.get(prefix)
                if ns is not None and local:
                    return URIRef(ns + local)
            return None

        def predicate_term(value):
            if value is None:
                return None
            if isinstance(value, URIRef):
                return value

            explicit = explicit_resource(value)
            if explicit is not None and not isinstance(explicit, BNode):
                return explicit

            if Ontology is not None:
                try:
                    qname = Ontology.PropertyQName(value, defaultPrefix="dict")
                    if qname is None:
                        return None
                    expanded = Ontology.ExpandQName(qname, defaultValue=None)
                    if expanded is not None:
                        return URIRef(expanded)
                except Exception:
                    pass

            ns = namespaces.get("dict")
            if ns is not None:
                return URIRef(ns + safe_local(value))
            return URIRef(str(value))

        def term(value, role="object"):
            if value is None:
                return None
            if isinstance(value, (URIRef, Literal, BNode)):
                return value

            # Preserve Ontology._RDFLiteral values without importing the private class.
            if all(hasattr(value, attr) for attr in ("lexical", "datatype", "language")):
                datatype = getattr(value, "datatype", None)
                language = getattr(value, "language", None)
                dt = URIRef(str(datatype)) if datatype else None
                return Literal(str(getattr(value, "lexical", "")), datatype=dt, lang=language)

            if isinstance(value, (bool, int, float)):
                return Literal(value)

            text = str(value).strip()
            if text == "":
                return None
            if role == "predicate":
                return predicate_term(value)

            explicit = explicit_resource(value)
            if explicit is not None:
                return explicit

            if role == "subject":
                ns = namespaces.get("inst", "http://w3id.org/topologicpy/instance#")
                return URIRef(ns + safe_local(text))

            if from_n3 is not None and text.startswith(("\"", "'")):
                try:
                    parsed = from_n3(text, nsm=g.namespace_manager)
                    if parsed is not None:
                        return parsed
                except Exception:
                    pass

            return Literal(value)

        for triple in triples or []:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                if not silent:
                    print("Reasoner.RDFGraphByTriples - Warning: Skipping invalid triple:", triple)
                continue
            s, p, o = triple
            ss = term(s, "subject")
            pp = term(p, "predicate")
            oo = term(o, "object")
            if ss is None or pp is None or oo is None:
                continue
            try:
                g.add((ss, pp, oo))
            except Exception as exc:
                if not silent:
                    print("Reasoner.RDFGraphByTriples - Warning: Could not add triple:", triple)
                    print("Error:", exc)
        return g

    @staticmethod
    def AddOntologyAxioms(rdfGraph, includeBOT: bool = True, silent: bool = False):
        """Adds axioms from the single canonical TopologicPy ontology graph."""
        if rdfGraph is None:
            return rdfGraph

        Ontology = Reasoner._ontology_class()
        if Ontology is None:
            if not silent:
                print("Reasoner.AddOntologyAxioms - Error: Ontology.py is required.")
            return rdfGraph

        try:
            axioms = Ontology.OntologyRDFGraph(silent=silent)
            if axioms is None:
                return rdfGraph

            namespaces = Ontology.Namespaces() if hasattr(Ontology, "Namespaces") else dict(Ontology.NAMESPACES)
            bot_ns = namespaces.get("bot", "https://w3id.org/bot#")

            for s, p, o in axioms:
                if not includeBOT and any(str(term).startswith(bot_ns) for term in (s, p, o)):
                    continue
                rdfGraph.add((s, p, o))

            Reasoner._bind_namespaces(rdfGraph)
        except Exception as exc:
            if not silent:
                print("Reasoner.AddOntologyAxioms - Error: Could not add canonical ontology axioms.")
                print("Error:", exc)
        return rdfGraph

    # ---------------------------------------------------------------------
    # Reasoning
    # ---------------------------------------------------------------------

    @staticmethod
    def Infer(
        rdfGraph,
        profile: str = "rdfs",
        includeOntologyAxioms: bool = True,
        includeBOT: bool = True,
        inplace: bool = False,
        maxIterations: int = 64,
        silent: bool = False,
    ):
        """
        Runs semantic inference over an RDFLib graph.

        Parameters
        ----------
        rdfGraph : rdflib.Graph
            The input RDFLib graph.
        profile : str , optional
            One of "rdfs", "owlrl", "shacl", or "none". "rdfs" uses the built-in
            lightweight reasoner. "owlrl" uses the optional owlrl package if
            available and falls back to built-in RDFS. "shacl" uses pyshacl if
            available and falls back to RDFS. Default is "rdfs".
        includeOntologyAxioms : bool , optional
            If True, injects TopologicPy ontology axioms before inference.
        includeBOT : bool , optional
            If True, adds BOT bridge subclass axioms.
        inplace : bool , optional
            If True, modifies the input RDF graph. Otherwise, returns a copy.
        maxIterations : int , optional
            Maximum closure iterations for the built-in RDFS reasoner.
        silent : bool , optional
            If True, suppresses warnings.

        Returns
        -------
        rdflib.Graph
            The inferred RDFLib graph.
        """
        rd = Reasoner._rdflib(silent=silent)
        if rd is None or rdfGraph is None:
            return None
        Graph = rd["Graph"]

        g = rdfGraph if inplace else Graph()
        if not inplace:
            for triple in rdfGraph:
                g.add(triple)
            for prefix, ns in getattr(rdfGraph, "namespaces", lambda: [])():
                try:
                    g.bind(prefix, ns)
                except Exception:
                    pass
        Reasoner._bind_namespaces(g)

        if includeOntologyAxioms:
            Reasoner.AddOntologyAxioms(g, includeBOT=includeBOT, silent=silent)

        profile_l = str(profile or "rdfs").strip().lower()
        if profile_l in ["none", "off", "false"]:
            return g

        if profile_l in ["owlrl", "owl-rl", "owl_rl"]:
            try:
                from owlrl import DeductiveClosure, OWLRL_Semantics
                DeductiveClosure(OWLRL_Semantics).expand(g)
                return g
            except Exception as exc:
                if not silent:
                    print("Reasoner.Infer - Warning: owlrl is not available or failed. Falling back to built-in RDFS.")
                    print("Error:", exc)

        if profile_l in ["shacl", "pyshacl"]:
            try:
                from pyshacl import validate
                # pyshacl inference mutates/returns through serialization in some modes;
                # this call is mainly useful when users pass a shapes graph externally.
                validate(g, inference="rdfs", inplace=True)
                return g
            except Exception as exc:
                if not silent:
                    print("Reasoner.Infer - Warning: pyshacl is not available or failed. Falling back to built-in RDFS.")
                    print("Error:", exc)

        return Reasoner._InferRDFS(g, maxIterations=maxIterations, silent=silent)

    @staticmethod
    def _InferRDFS(rdfGraph, maxIterations: int = 64, silent: bool = False):
        """A deterministic RDFS/OWL-alias closure implementation.

        Supports rdfs:subClassOf, rdfs:subPropertyOf, rdfs:domain,
        rdfs:range, owl:equivalentClass, owl:equivalentProperty, and
        owl:inverseOf, using the canonical ontology axioms.
        """
        rd = Reasoner._rdflib(silent=silent)
        if rd is None or rdfGraph is None:
            return rdfGraph
        RDF = rd["RDF"]
        RDFS = rd["RDFS"]
        OWL = rd["OWL"]
        URIRef = rd["URIRef"]

        subclass = RDFS.subClassOf
        subprop = RDFS.subPropertyOf
        domain = RDFS.domain
        range_ = RDFS.range
        rdf_type = RDF.type
        equiv_class = OWL.equivalentClass
        equiv_prop = OWL.equivalentProperty
        inverse_of = OWL.inverseOf

        def add_triples(triples: Iterable[Tuple[Any, Any, Any]]) -> int:
            count = 0
            for t in triples:
                if t is None or len(t) != 3 or None in t:
                    continue
                if t not in rdfGraph:
                    rdfGraph.add(t)
                    count += 1
            return count

        iterations = max(1, int(maxIterations))
        for _ in range(iterations):
            new: Set[Tuple[Any, Any, Any]] = set()

            for a, _, b in list(rdfGraph.triples((None, equiv_class, None))):
                if a != b:
                    new.add((a, subclass, b))
                    new.add((b, subclass, a))
            for a, _, b in list(rdfGraph.triples((None, equiv_prop, None))):
                if a != b:
                    new.add((a, subprop, b))
                    new.add((b, subprop, a))
            for a, _, b in list(rdfGraph.triples((None, inverse_of, None))):
                if a != b:
                    new.add((b, inverse_of, a))

            for c, _, p in list(rdfGraph.triples((None, subclass, None))):
                for _, _, gp in rdfGraph.triples((p, subclass, None)):
                    if gp != c:
                        new.add((c, subclass, gp))
            for s, _, c in list(rdfGraph.triples((None, rdf_type, None))):
                for _, _, sup in rdfGraph.triples((c, subclass, None)):
                    new.add((s, rdf_type, sup))

            for p, _, q in list(rdfGraph.triples((None, subprop, None))):
                for _, _, r in rdfGraph.triples((q, subprop, None)):
                    if r != p:
                        new.add((p, subprop, r))
            for p, _, q in list(rdfGraph.triples((None, subprop, None))):
                for s, _, o in list(rdfGraph.triples((None, p, None))):
                    new.add((s, q, o))

            for p, _, q in list(rdfGraph.triples((None, inverse_of, None))):
                for s, _, o in list(rdfGraph.triples((None, p, None))):
                    if isinstance(o, URIRef):
                        new.add((o, q, s))

            for p in set(pred for _, pred, _ in rdfGraph):
                domains = [c for _, _, c in rdfGraph.triples((p, domain, None))]
                ranges = [c for _, _, c in rdfGraph.triples((p, range_, None))]
                if not domains and not ranges:
                    continue
                for s, _, o in list(rdfGraph.triples((None, p, None))):
                    for c in domains:
                        new.add((s, rdf_type, c))
                    if isinstance(o, URIRef):
                        for c in ranges:
                            new.add((o, rdf_type, c))

            if add_triples(new) == 0:
                break
        return rdfGraph

    @staticmethod
    def Types(rdfGraph, subject: Any, compact: bool = True) -> List[str]:
        """Returns rdf:type values for a subject."""
        rd = Reasoner._rdflib(silent=True)
        if rd is None or rdfGraph is None:
            return []
        s = Reasoner._uri_ref(subject, URIRef=rd["URIRef"])
        result = []
        for _, _, o in rdfGraph.triples((s, rd["RDF"].type, None)):
            result.append(Reasoner.QName(o) if compact else str(o))
        return sorted(set(result))

    @staticmethod
    def SuperClasses(rdfGraph, ontologyClass: str, compact: bool = True) -> List[str]:
        """Returns known/inferred superclasses of the input ontology class."""
        rd = Reasoner._rdflib(silent=True)
        if rd is None or rdfGraph is None:
            return []
        c = Reasoner._uri_ref(ontologyClass, URIRef=rd["URIRef"])
        result = []
        for _, _, o in rdfGraph.triples((c, rd["RDFS"].subClassOf, None)):
            result.append(Reasoner.QName(o) if compact else str(o))
        return sorted(set(result))

    @staticmethod
    def Difference(beforeGraph, afterGraph, compact: bool = True, limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """Returns triples present in afterGraph but absent from beforeGraph."""
        if beforeGraph is None or afterGraph is None:
            return []
        diff = []
        for s, p, o in afterGraph:
            if (s, p, o) in beforeGraph:
                continue
            if compact:
                triple = _reasoner_compact_triple((s, p, o))
                if triple is not None:
                    diff.append(triple)
            else:
                diff.append((str(s), str(p), str(o)))
            if limit is not None and len(diff) >= int(limit):
                break
        return diff

    @staticmethod
    def Summary(beforeGraph, afterGraph) -> Dict[str, Any]:
        """Returns a compact inference summary."""
        before_count = len(beforeGraph) if beforeGraph is not None else 0
        after_count = len(afterGraph) if afterGraph is not None else 0
        return {
            "input_triples": before_count,
            "output_triples": after_count,
            "inferred_triples": max(0, after_count - before_count),
        }

    @staticmethod
    def TurtleString(rdfGraph, format: str = "turtle", silent: bool = False) -> Optional[str]:
        """Serializes an RDFLib graph to Turtle or another RDFLib-supported format."""
        if rdfGraph is None:
            return None
        try:
            text = rdfGraph.serialize(format=format)
            if isinstance(text, bytes):
                return text.decode("utf-8")
            return str(text)
        except Exception as exc:
            if not silent:
                print("Reasoner.TurtleString - Error: Could not serialize RDF graph.")
                print("Error:", exc)
            return None

    @staticmethod
    def ExportRDF(rdfGraph, path: str, format: str = "turtle", overwrite: bool = True, silent: bool = False) -> Optional[str]:
        """Exports an RDFLib graph to disk."""
        if rdfGraph is None or not isinstance(path, str) or path.strip() == "":
            return None
        import os
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("Reasoner.ExportRDF - Error: The output path already exists and overwrite is False.")
            return None
        try:
            rdfGraph.serialize(destination=path, format=format)
            return path
        except Exception as exc:
            if not silent:
                print("Reasoner.ExportRDF - Error: Could not export RDF graph.")
                print("Error:", exc)
            return None

    # ---------------------------------------------------------------------
    # Applying inferred facts back to TGraph dictionaries
    # ---------------------------------------------------------------------

    @staticmethod
    def _subject_for_dictionary(dictionary: Dict[str, Any], fallback: str, namespacePrefix: str = "inst") -> str:
        """Returns the RDF subject used by the canonical Ontology serializer.

        Labels are never used for identity. This method intentionally mirrors
        Ontology._identity so inferred facts are applied to the same RDF resources
        that were serialized from the TGraph.
        """
        d = dictionary if isinstance(dictionary, dict) else {}
        fallback_text = str(fallback or "resource")

        role = "resource"
        fallback_index: Any = fallback_text
        lower = fallback_text.lower()
        if lower == "graph" or lower.startswith("graph_"):
            role = "graph"
            fallback_index = fallback_text[6:] if lower.startswith("graph_") else "graph"
        elif lower.startswith("vertex_"):
            role = "node"
            fallback_index = fallback_text.split("_", 1)[1]
        elif lower.startswith("node_"):
            role = "node"
            fallback_index = fallback_text.split("_", 1)[1]
        elif lower.startswith("edge_"):
            role = "relationship"
            fallback_index = fallback_text.split("_", 1)[1]
        elif lower.startswith("relationship_"):
            role = "relationship"
            fallback_index = fallback_text.split("_", 1)[1]

        try:
            if isinstance(fallback_index, str) and fallback_index.isdigit():
                fallback_index = int(fallback_index)
        except Exception:
            pass

        Ontology = Reasoner._ontology_class()
        if Ontology is not None and hasattr(Ontology, "_identity"):
            try:
                return Ontology._identity(
                    d,
                    role=role,
                    fallbackIndex=fallback_index,
                    prefix=namespacePrefix,
                )
            except TypeError:
                try:
                    return Ontology._identity(d, role, fallback_index, namespacePrefix)
                except Exception:
                    pass
            except Exception:
                pass

        def safe(value):
            text = "" if value is None else str(value).strip()
            text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
            text = re.sub(r"_+", "_", text).strip("_") or "item"
            if text[0].isdigit():
                text = "id_" + text
            return text

        for key in ("_rdf_uri", "uri", "URI"):
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if value.startswith(("http://", "https://", "urn:", "<", "_:")) or ":" in value:
                    return value
                return f"{namespacePrefix}:{safe(value)}"

        for key in ("uuid", "UUID", "id", "ID", "ifc_guid", "ifcGUID", "GlobalId", "IFC_global_id"):
            value = d.get(key)
            if value not in (None, ""):
                return f"{namespacePrefix}:{safe(value)}"

        if d.get("index") not in (None, ""):
            return f"{namespacePrefix}:{role}_{safe(d.get('index'))}"
        return f"{namespacePrefix}:{role}_{safe(fallback_index)}"

    @staticmethod
    def _types_for_subject(rdfGraph, subjectQName: str) -> List[str]:
        return Reasoner.Types(rdfGraph, subjectQName, compact=True)

    @staticmethod
    def ApplyInferences(
        graph: Any,
        inferredGraph,
        namespacePrefix: str = "inst",
        typeKey: str = "inferred_ontology_classes",
        botTypeKey: str = "inferred_bot_classes",
        overwrite: bool = True,
        includeGraph: bool = True,
        includeVertices: bool = True,
        includeEdges: bool = True,
        silent: bool = False,
    ) -> Any:
        """Writes inferred canonical types back into TGraph dictionaries.

        Only classes declared by the canonical TopologicPy ontology are written to
        ``typeKey``. BOT classes are stored separately under ``botTypeKey``.
        """
        TGraph = Reasoner._tgraph_class()
        Ontology = Reasoner._ontology_class()
        if TGraph is None or Ontology is None or graph is None or inferredGraph is None:
            return graph

        try:
            if not isinstance(graph, TGraph):
                return graph
        except Exception:
            return graph

        def canonical_class(cls):
            if cls is None:
                return None
            try:
                return Ontology.CanonicalClass(cls, defaultValue=None)
            except Exception:
                return None

        def split_types(types: Sequence[str]) -> Tuple[List[str], List[str]]:
            top_types = []
            bot_types = []
            for value in set(types or []):
                if not isinstance(value, str):
                    continue
                if value.startswith("top:"):
                    canonical = canonical_class(value)
                    if canonical is not None:
                        top_types.append(canonical)
                elif value.startswith("bot:"):
                    bot_types.append(value)
            return sorted(set(top_types)), sorted(set(bot_types))

        def apply_to_dict(d: Dict[str, Any], fallback: str):
            if not isinstance(d, dict):
                return
            subject = Reasoner._subject_for_dictionary(d, fallback, namespacePrefix=namespacePrefix)
            types = Reasoner._types_for_subject(inferredGraph, subject)
            top_types, bot_types = split_types(types)
            asserted = canonical_class(d.get("ontology_class"))
            inferred_top = [value for value in top_types if value != asserted]

            if overwrite:
                d[typeKey] = inferred_top
                d[botTypeKey] = bot_types
            else:
                previous = d.get(typeKey, [])
                if not isinstance(previous, list):
                    previous = [previous]
                previous = [canonical_class(value) for value in previous]
                d[typeKey] = sorted(set([value for value in previous if value] + inferred_top))

                if bot_types:
                    previous_bot = d.get(botTypeKey, [])
                    if not isinstance(previous_bot, list):
                        previous_bot = [previous_bot]
                    d[botTypeKey] = sorted(set(previous_bot + bot_types))

        try:
            if includeGraph:
                apply_to_dict(graph._dictionary, "graph")
            if includeVertices:
                for vertex in graph._vertices:
                    if vertex.get("active", True):
                        apply_to_dict(vertex.get("dictionary", {}), f"vertex_{vertex.get('index')}")
            if includeEdges:
                for edge in graph._edges:
                    if edge.get("active", True):
                        apply_to_dict(edge.get("dictionary", {}), f"edge_{edge.get('index')}")
        except Exception as exc:
            if not silent:
                print("Reasoner.ApplyInferences - Warning: Could not apply all inferences.")
                print("Error:", exc)

        try:
            graph._invalidate_cache()
        except Exception:
            pass
        return graph

    # ---------------------------------------------------------------------
    # SHACL convenience validation
    # ---------------------------------------------------------------------

    @staticmethod
    def Validate(rdfGraph, shapesGraph=None, inference: str = "rdfs", silent: bool = False) -> Dict[str, Any]:
        """
        Validates an RDF graph with pyshacl when available.

        If pyshacl is not installed, this returns an informative report rather
        than failing.
        """
        try:
            from pyshacl import validate
            conforms, results_graph, results_text = validate(
                data_graph=rdfGraph,
                shacl_graph=shapesGraph,
                inference=inference,
                debug=False,
            )
            return {"available": True, "conforms": bool(conforms), "results_graph": results_graph, "results_text": results_text}
        except Exception as exc:
            if not silent:
                print("Reasoner.Validate - Warning: pyshacl is unavailable or validation failed.")
                print("Error:", exc)
            return {"available": False, "conforms": None, "results_graph": None, "results_text": str(exc)}

# -----------------------------------------------------------------------------
# Explainability and proof-graph extension methods
# -----------------------------------------------------------------------------
# These functions are attached to Reasoner at module load time to preserve
# the public Reasoner API while adding proof/explanation support
# while adding proof/explanation support used by TGraph and Plotly.

class _ReasonerFact:
    """Container for one asserted or inferred semantic fact."""

    def __init__(self, triple, asserted=False, inferred=False, rule=None, premises=None, depth=0, confidence=1.0, engine="TopologicPy Reasoner"):
        self.triple = tuple(triple) if triple is not None else None
        self.asserted = bool(asserted)
        self.inferred = bool(inferred)
        self.rule = rule
        self.premises = list(premises or [])
        self.depth = int(depth or 0)
        self.confidence = float(confidence if confidence is not None else 1.0)
        self.engine = engine

    def ToDictionary(self):
        return {
            "triple": self.triple,
            "asserted": self.asserted,
            "inferred": self.inferred,
            "rule": self.rule,
            "premises": list(self.premises),
            "depth": self.depth,
            "confidence": self.confidence,
            "engine": self.engine,
        }

    def __repr__(self):
        kind = "asserted" if self.asserted else "inferred"
        return f"Fact({self.triple!r}, {kind}, rule={self.rule!r})"


class _ReasonerInferenceResult:
    """Container for a reasoning run, including asserted/inferred facts."""

    def __init__(self, beforeGraph=None, afterGraph=None, profile="rdfs", engine="TopologicPy Reasoner", facts=None):
        self.before_graph = beforeGraph
        self.after_graph = afterGraph
        self.input_graph = beforeGraph
        self.output_graph = afterGraph
        self.profile = profile
        self.engine = engine
        self.facts = facts if isinstance(facts, dict) else {}

    def Triples(self):
        return list(self.facts.keys())

    def InferredTriples(self):
        return [t for t, f in self.facts.items() if getattr(f, "inferred", False)]

    def AssertedTriples(self):
        return [t for t, f in self.facts.items() if getattr(f, "asserted", False)]

    def Summary(self):
        inferred = len(self.InferredTriples())
        asserted = len(self.AssertedTriples())
        return {
            "profile": self.profile,
            "engine": self.engine,
            "asserted_facts": asserted,
            "inferred_facts": inferred,
            "total_facts": len(self.facts),
        }

    def __repr__(self):
        s = self.Summary()
        return f"InferenceResult(asserted={s['asserted_facts']}, inferred={s['inferred_facts']}, total={s['total_facts']})"


def _reasoner_is_literal_token(value):
    if value is None:
        return False
    text = str(value)
    return text.startswith('"') or "^^" in text


def _reasoner_compact_node(node):
    """Returns a compact token for an RDFLib node or an already compact token."""
    if node is None:
        return None
    rd = Reasoner._rdflib(silent=True)
    try:
        if rd is not None:
            if isinstance(node, rd["Literal"]):
                try:
                    return node.n3()
                except Exception:
                    return str(node)
            BNode = rd.get("BNode", None)
            if BNode is not None and isinstance(node, BNode):
                return "_:" + str(node)
            if isinstance(node, rd["URIRef"]):
                return Reasoner.QName(node)
    except Exception:
        pass
    text = str(node).strip()
    if text.startswith('"') or text.startswith("_:"):
        return text
    if text.startswith("<") and text.endswith(">"):
        return Reasoner.QName(text, defaultValue=text)
    if text.startswith("http://") or text.startswith("https://"):
        return Reasoner.QName(text, defaultValue="<" + text + ">")
    return text


def _reasoner_compact_triple(triple):
    """Returns a compact triple without rewriting ontology vocabulary."""
    if triple is None or len(triple) != 3:
        return None
    s, p, o = triple
    return (
        _reasoner_compact_node(s),
        _reasoner_compact_node(p),
        _reasoner_compact_node(o),
    )

def _reasoner_triples(obj):
    """Returns compact triples from rdflib Graph, KnowledgeGraph, InferenceResult, or iterable."""
    if obj is None:
        return []
    if isinstance(obj, _ReasonerInferenceResult):
        return list(obj.facts.keys())
    if hasattr(obj, "facts") and isinstance(getattr(obj, "facts"), dict):
        return list(getattr(obj, "facts").keys())
    try:
        # KnowledgeGraph.Triples(sort=True)
        if hasattr(obj, "Triples"):
            try:
                return [_reasoner_compact_triple(t) for t in obj.Triples(sort=True)]
            except TypeError:
                return [_reasoner_compact_triple(t) for t in obj.Triples()]
    except Exception:
        pass
    try:
        return [_reasoner_compact_triple(t) for t in list(obj)]
    except Exception:
        return []


def _reasoner_make_result(beforeGraph, afterGraph, profile="rdfs", engine="TopologicPy Reasoner"):
    before = set(t for t in _reasoner_triples(beforeGraph) if t is not None)
    after = set(t for t in _reasoner_triples(afterGraph) if t is not None)
    facts = {}
    for t in sorted(after):
        asserted = t in before
        facts[t] = _ReasonerFact(
            t,
            asserted=asserted,
            inferred=not asserted,
            rule=None if asserted else "RDFS/OWL-RL closure",
            premises=[],
            depth=0 if asserted else 1,
            confidence=1.0,
            engine=engine,
        )
    return _ReasonerInferenceResult(beforeGraph=beforeGraph, afterGraph=afterGraph, profile=profile, engine=engine, facts=facts)


def _reasoner_graph_from_result_or_graph(obj):
    if isinstance(obj, _ReasonerInferenceResult):
        return obj.after_graph
    if hasattr(obj, "after_graph"):
        return getattr(obj, "after_graph")
    return obj


def _reasoner_asserted_set(obj):
    if isinstance(obj, _ReasonerInferenceResult):
        return set(obj.AssertedTriples())
    if hasattr(obj, "facts") and isinstance(getattr(obj, "facts"), dict):
        return set(t for t, f in getattr(obj, "facts").items() if getattr(f, "asserted", False))
    return set(_reasoner_triples(obj))


def _reasoner_fact_set(obj):
    return set(_reasoner_triples(obj))


def _reasoner_normalize_target(subject=None, predicate=None, object=None):
    if predicate is None and object is None and isinstance(subject, (list, tuple)) and len(subject) == 3:
        subject, predicate, object = subject
    if subject is None or predicate is None or object is None:
        return None
    return _reasoner_compact_triple((subject, predicate, object))


def _reasoner_subclass_paths(facts, start, target, maxDepth=32):
    """Returns one superclass path from start to target using rdfs:subClassOf triples."""
    start = str(start); target = str(target)
    if start == target:
        return [start]
    adj = {}
    for s, p, o in facts:
        if str(p) == "rdfs:subClassOf":
            adj.setdefault(str(s), set()).add(str(o))
    queue = [(start, [start])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if len(path) > maxDepth:
            continue
        for nxt in sorted(adj.get(node, [])):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == target:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


def _reasoner_proof_tree(resultOrGraph, subject=None, predicate=None, object=None, maxDepth=32):
    target = _reasoner_normalize_target(subject, predicate, object)
    if target is None:
        return None
    facts = _reasoner_fact_set(resultOrGraph)
    asserted = _reasoner_asserted_set(resultOrGraph)
    is_asserted = target in asserted
    is_known = target in facts
    tree = {
        "triple": target,
        "asserted": bool(is_asserted),
        "inferred": bool(is_known and not is_asserted),
        "rule": None if is_asserted else "Unknown",
        "confidence": 1.0,
        "premises": [],
        "depth": 0 if is_asserted else 1,
    }
    if not is_known:
        tree["rule"] = "Not found"
        return tree
    if is_asserted:
        tree["rule"] = "Asserted"
        return tree

    s, p, o = target
    # RDFS type inheritance: x rdf:type C, C rdfs:subClassOf D => x rdf:type D.
    if p == "rdf:type":
        asserted_types = sorted(t[2] for t in asserted if t[0] == s and t[1] == "rdf:type")
        for base in asserted_types:
            path = _reasoner_subclass_paths(facts, base, o, maxDepth=maxDepth)
            if path and len(path) >= 2:
                premises = [(s, "rdf:type", base)]
                for a, b in zip(path[:-1], path[1:]):
                    premises.append((a, "rdfs:subClassOf", b))
                tree.update({
                    "rule": "RDFS9: type inheritance through rdfs:subClassOf",
                    "premises": premises,
                    "depth": len(premises),
                })
                return tree

        # Domain/range explanations.
        for ss, pp, oo in sorted(facts):
            if ss == s:
                dom = (pp, "rdfs:domain", o)
                if dom in facts:
                    tree.update({
                        "rule": "RDFS2: domain inference",
                        "premises": [(ss, pp, oo), dom],
                        "depth": 2,
                    })
                    return tree
            if oo == s:
                rng = (pp, "rdfs:range", o)
                if rng in facts:
                    tree.update({
                        "rule": "RDFS3: range inference",
                        "premises": [(ss, pp, oo), rng],
                        "depth": 2,
                    })
                    return tree

    # Subproperty explanation: x p y, p subPropertyOf q => x q y.
    if (s, p, o) in facts:
        for ss, pp, oo in sorted(asserted):
            if ss == s and oo == o:
                path = _reasoner_subproperty_path(facts, pp, p, maxDepth=maxDepth)
                if path and len(path) >= 2:
                    premises = [(ss, pp, oo)]
                    for a, b in zip(path[:-1], path[1:]):
                        premises.append((a, "rdfs:subPropertyOf", b))
                    tree.update({
                        "rule": "RDFS7: property inheritance through rdfs:subPropertyOf",
                        "premises": premises,
                        "depth": len(premises),
                    })
                    return tree

    # Inverse-property explanation: x p y, p owl:inverseOf q => y q x.
    for ss, pp, oo in sorted(asserted):
        if ss == o and oo == s:
            inv_a = (pp, "owl:inverseOf", p)
            inv_b = (p, "owl:inverseOf", pp)
            if inv_a in facts or inv_b in facts:
                inv = inv_a if inv_a in facts else inv_b
                tree.update({
                    "rule": "OWL inverse property inference",
                    "premises": [(ss, pp, oo), inv],
                    "depth": 2,
                })
                return tree

    return tree


def _reasoner_subproperty_path(facts, start, target, maxDepth=32):
    start = str(start); target = str(target)
    if start == target:
        return [start]
    adj = {}
    for s, p, o in facts:
        if str(p) == "rdfs:subPropertyOf":
            adj.setdefault(str(s), set()).add(str(o))
    queue = [(start, [start])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if len(path) > maxDepth:
            continue
        for nxt in sorted(adj.get(node, [])):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt == target:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


def _reasoner_format_triple(triple):
    if triple is None:
        return "None"
    return f"{triple[0]} {triple[1]} {triple[2]}"


def _Reasoner_ProofTree(resultOrGraph, subject=None, predicate=None, object=None, triple=None,
                        maxDepth=32, **kwargs):
    """Returns a structured proof tree for a target triple."""
    if triple is not None:
        subject, predicate, object = triple
    elif isinstance(subject, (list, tuple)) and len(subject) == 3 and predicate is None and object is None:
        subject, predicate, object = subject
    return _reasoner_proof_tree(resultOrGraph, subject, predicate, object, maxDepth=maxDepth)


def _Reasoner_Explain(resultOrGraph, subject=None, predicate=None, object=None, triple=None,
                      maxDepth=32, compact=True, **kwargs):
    """Returns a human-readable explanation for a target inferred or asserted triple."""
    if triple is not None:
        subject, predicate, object = triple
    elif isinstance(subject, (list, tuple)) and len(subject) == 3 and predicate is None and object is None:
        subject, predicate, object = subject
    tree = _reasoner_proof_tree(resultOrGraph, subject, predicate, object, maxDepth=maxDepth)
    if tree is None:
        return "No target triple was specified."
    target = tree.get("triple")
    if tree.get("rule") == "Not found":
        return "No proof was found because the target triple is not present: " + _reasoner_format_triple(target)
    lines = []
    if tree.get("asserted"):
        lines.append("The following fact is asserted directly:")
        lines.append("  " + _reasoner_format_triple(target))
        return "\n".join(lines)
    lines.append("The following fact is inferred:")
    lines.append("  " + _reasoner_format_triple(target))
    lines.append("")
    lines.append("Rule: " + str(tree.get("rule", "Unknown")))
    premises = tree.get("premises", []) or []
    if premises:
        lines.append("Premises:")
        for p in premises:
            lines.append("  - " + _reasoner_format_triple(p))
    else:
        lines.append("No detailed premise chain was available for this inferred fact.")
    return "\n".join(lines)


def _Reasoner_ProofGraphData(resultOrGraph=None, subject=None, predicate=None, object=None, triple=None, maxDepth=32, includePremiseClosure=True, **kwargs):
    """Returns proof graph data as {'nodes': [...], 'edges': [...]} for Plotly visualisation."""
    target = triple if triple is not None else _reasoner_normalize_target(subject, predicate, object)
    if target is None and isinstance(subject, (list, tuple)) and len(subject) == 3:
        target = tuple(subject)
    tree = _reasoner_proof_tree(resultOrGraph, target, maxDepth=maxDepth) if target is not None else None
    nodes = []
    edges = []
    node_ids = set()

    def add_node(node_id, label, kind="fact", **attrs):
        node_id = str(node_id)
        if node_id not in node_ids:
            d = {"id": node_id, "label": str(label), "kind": kind, "type": kind}
            d.update(attrs)
            nodes.append(d)
            node_ids.add(node_id)
        return node_id

    def add_edge(src, dst, label="supports", kind="supports"):
        edges.append({"src": str(src), "dst": str(dst), "label": str(label), "kind": kind, "relationship": str(label)})

    if tree is None:
        return {"nodes": nodes, "edges": edges}

    target_triple = tree.get("triple")
    target_id = "fact:" + _reasoner_format_triple(target_triple)
    add_node(target_id, _reasoner_format_triple(target_triple), kind="conclusion", triple=target_triple, asserted=tree.get("asserted"), inferred=tree.get("inferred"))

    rule = tree.get("rule") or "Unknown"
    rule_id = "rule:" + str(rule)
    add_node(rule_id, str(rule), kind="rule", rule=rule)
    add_edge(rule_id, target_id, label="derives", kind="derives")

    premises = tree.get("premises", []) or []
    for i, prem in enumerate(premises):
        pid = "fact:" + _reasoner_format_triple(prem)
        add_node(pid, _reasoner_format_triple(prem), kind="premise", triple=prem)
        add_edge(pid, rule_id, label="premise", kind="premise")
    return {"nodes": nodes, "edges": edges, "target": target_triple, "proof_tree": tree}


def _Reasoner_ProofGraph(resultOrGraph=None, subject=None, predicate=None, object=None, triple=None, **kwargs):
    """Alias for ProofGraphData."""
    return _Reasoner_ProofGraphData(resultOrGraph=resultOrGraph, subject=subject, predicate=predicate, object=object, triple=triple, **kwargs)


def _Reasoner_ProofTGraph(resultOrGraph=None, subject=None, predicate=None, object=None, triple=None, **kwargs):
    """Returns a TGraph representation of the proof graph."""
    data = _Reasoner_ProofGraphData(resultOrGraph=resultOrGraph, subject=subject, predicate=predicate, object=object, triple=triple, **kwargs)
    TGraph = Reasoner._tgraph_class()
    if TGraph is None:
        return None
    g = TGraph(directed=True, allowSelfLoops=False, allowParallelEdges=True, dictionary={"label": "Proof Graph", "ontology_class": "top:KnowledgeGraph", "category": "proof_graph"})
    id_to_index = {}
    n = max(1, len(data.get("nodes", [])))
    import math
    for i, node in enumerate(data.get("nodes", [])):
        d = dict(node)
        d.setdefault("ontology_class", "top:Node")
        d.setdefault("category", node.get("kind", "proof"))
        # Simple circular fallback positions. Plotly.DataByProofGraph can override layout.
        a = 2.0 * math.pi * i / n
        d.setdefault("x", math.cos(a))
        d.setdefault("y", math.sin(a))
        d.setdefault("z", 0.0)
        id_to_index[node.get("id")] = g.AddVertex(d)
    for edge in data.get("edges", []):
        src = id_to_index.get(edge.get("src"))
        dst = id_to_index.get(edge.get("dst"))
        if src is not None and dst is not None:
            d = dict(edge)
            d.setdefault("ontology_class", "top:Relationship")
            d.setdefault("category", "proof_relationship")
            g.AddEdge(src, dst, directed=True, dictionary=d)
    return g


def _Reasoner_ProofGraphFigure(resultOrGraph=None, subject=None, predicate=None, object=None, triple=None, **kwargs):
    """Returns a Plotly figure for a proof graph using Plotly.py when available.

    Plotly.FigureByProofGraph accepts proofGraphData/result/triple/graph, not the
    subject/predicate/object keyword triplet used by Reasoner.  Normalising the
    target here prevents those keywords from being forwarded through Plotly's
    **kwargs path and failing in Plotly.DataByProofGraph.
    """
    try:
        from topologicpy.Plotly import Plotly
    except Exception:
        try:
            from Plotly import Plotly
        except Exception:
            return None

    target = triple if triple is not None else _reasoner_normalize_target(subject, predicate, object)
    if target is None and isinstance(subject, (list, tuple)) and len(subject) == 3:
        target = tuple(subject)

    try:
        return Plotly.FigureByProofGraph(result=resultOrGraph, triple=target, **kwargs)
    except TypeError:
        try:
            return Plotly.FigureByProofGraph(proofGraphData=None, result=resultOrGraph, triple=target, **kwargs)
        except TypeError:
            return None


def _Reasoner_ProofGraphHTML(resultOrGraph=None, subject=None, predicate=None, object=None, triple=None, path="proof_graph.html", **kwargs):
    """Exports a proof graph Plotly figure to HTML.

    Subject/predicate/object are normalised to a single triple before calling
    Plotly.ProofGraphHTML for the same reason as _Reasoner_ProofGraphFigure.
    """
    try:
        from topologicpy.Plotly import Plotly
    except Exception:
        try:
            from Plotly import Plotly
        except Exception:
            return None

    target = triple if triple is not None else _reasoner_normalize_target(subject, predicate, object)
    if target is None and isinstance(subject, (list, tuple)) and len(subject) == 3:
        target = tuple(subject)

    try:
        return Plotly.ProofGraphHTML(result=resultOrGraph, triple=target, path=path, **kwargs)
    except TypeError:
        try:
            return Plotly.ProofGraphHTML(proofGraphData=None, result=resultOrGraph, triple=target, path=path, **kwargs)
        except TypeError:
            return None


def _Reasoner_RuleStatistics(resultOrGraph):
    """Returns counts grouped by inference rule."""
    stats = {}
    if isinstance(resultOrGraph, _ReasonerInferenceResult) or hasattr(resultOrGraph, "facts"):
        for f in getattr(resultOrGraph, "facts", {}).values():
            rule = getattr(f, "rule", None) or ("Asserted" if getattr(f, "asserted", False) else "Unknown")
            stats[rule] = stats.get(rule, 0) + 1
        return stats
    return {"facts": len(_reasoner_triples(resultOrGraph))}


def _Reasoner_Diff(beforeGraph, afterGraph=None, compact=True, limit=None):
    """Returns a semantic diff. If passed an InferenceResult, returns inferred triples."""
    if afterGraph is None and (isinstance(beforeGraph, _ReasonerInferenceResult) or hasattr(beforeGraph, "facts")):
        inferred = []
        for t, f in getattr(beforeGraph, "facts", {}).items():
            if getattr(f, "inferred", False):
                inferred.append(t)
                if limit is not None and len(inferred) >= int(limit):
                    break
        return {"added": inferred, "removed": [], "changed": [], "count_added": len(inferred), "count_removed": 0}

    before = set(_reasoner_triples(beforeGraph))
    after = set(_reasoner_triples(afterGraph))
    added = sorted(after - before)
    removed = sorted(before - after)
    if limit is not None:
        added = added[:int(limit)]
        removed = removed[:int(limit)]
    return {
        "added": added,
        "removed": removed,
        "changed": [],
        "count_added": len(added),
        "count_removed": len(removed),
    }


def _Reasoner_Result(beforeGraph, afterGraph, profile="rdfs", engine="TopologicPy Reasoner"):
    """Builds an explainable InferenceResult from before/after graphs."""
    return _reasoner_make_result(beforeGraph, afterGraph, profile=profile, engine=engine)


# Attach public API methods.
Reasoner.Fact = _ReasonerFact
Reasoner.InferenceResult = _ReasonerInferenceResult
Reasoner.Result = staticmethod(_Reasoner_Result)
Reasoner.Explain = staticmethod(_Reasoner_Explain)
Reasoner.ProofTree = staticmethod(_Reasoner_ProofTree)
Reasoner.ProofGraph = staticmethod(_Reasoner_ProofGraph)
Reasoner.ProofGraphData = staticmethod(_Reasoner_ProofGraphData)
Reasoner.ProofGraphFigure = staticmethod(_Reasoner_ProofGraphFigure)
Reasoner.ProofGraphHTML = staticmethod(_Reasoner_ProofGraphHTML)
Reasoner.ProofTGraph = staticmethod(_Reasoner_ProofTGraph)
Reasoner.RuleStatistics = staticmethod(_Reasoner_RuleStatistics)
Reasoner.Diff = staticmethod(_Reasoner_Diff)
