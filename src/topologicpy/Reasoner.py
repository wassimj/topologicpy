# Copyright (C) 2026
# TopologicPy
#
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.
#
# TopologicPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


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
            from rdflib import Graph, Namespace, URIRef, Literal
            from rdflib.namespace import RDF, RDFS, OWL, XSD
            return {
                "rdflib": rdflib,
                "Graph": Graph,
                "Namespace": Namespace,
                "URIRef": URIRef,
                "Literal": Literal,
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
        """Returns the namespace dictionary used by TopologicPy reasoning."""
        Ontology = Reasoner._ontology_class()
        if Ontology is not None:
            try:
                namespaces = dict(Ontology.NAMESPACES)
                namespaces.setdefault("inst", "http://w3id.org/topologicpy/instance#")
                return namespaces
            except Exception:
                pass
        return {
            "bot": "https://w3id.org/bot#",
            "brick": "https://brickschema.org/schema/Brick#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "top": "http://w3id.org/topologicpy#",
            "inst": "http://w3id.org/topologicpy/instance#",
        }

    @staticmethod
    def ExpandQName(term: Any, defaultValue: Any = None):
        """Expands a QName such as top:Room to a URI string."""
        if not isinstance(term, str):
            return defaultValue
        if term.startswith("http://") or term.startswith("https://"):
            return term
        if ":" not in term:
            return defaultValue
        prefix, local = term.split(":", 1)
        ns = Reasoner.Namespaces().get(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    @staticmethod
    def QName(uri: Any, defaultValue: Any = None) -> Any:
        """Compacts a URIRef or URI string to a QName when a namespace matches."""
        text = str(uri)
        for prefix, ns in Reasoner.Namespaces().items():
            if text.startswith(ns):
                return prefix + ":" + text[len(ns):]
        return defaultValue if defaultValue is not None else text

    @staticmethod
    def _uri_ref(term: Any, URIRef=None):
        if URIRef is None:
            rd = Reasoner._rdflib(silent=True)
            if rd is None:
                return None
            URIRef = rd["URIRef"]
        if term is None:
            return None
        if isinstance(term, URIRef):
            return term
        if isinstance(term, str):
            expanded = Reasoner.ExpandQName(term, defaultValue=None)
            if expanded is not None:
                return URIRef(expanded)
        return URIRef(str(term))

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
        """
        Returns an RDFLib graph from a TopologicPy topology, legacy graph, or TGraph.
        """
        Ontology = Reasoner._ontology_class()
        if Ontology is None:
            if not silent:
                print("Reasoner.RDFGraphByTopology - Error: Ontology.py could not be imported.")
            return None
        g = None
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
                print("Reasoner.RDFGraphByTopology - Warning: Ontology.RDFGraph failed. Trying TGraph/Ontology triples fallback.")
                print("Error:", exc)

        if g is None:
            g = Reasoner.RDFGraphByTriples(
                Reasoner.TriplesByTopology(
                    topology,
                    includeGraph=includeGraph,
                    includeDictionaries=includeDictionaries,
                    includeBOT=includeBOT,
                    namespacePrefix=namespacePrefix,
                    silent=silent,
                ),
                silent=silent,
            )
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
        """Returns RDF-like triples from a TopologicPy object using Ontology/TGraph helpers."""
        Ontology = Reasoner._ontology_class()
        TGraph = Reasoner._tgraph_class()
        if topology is None:
            return []
        if TGraph is not None:
            try:
                if isinstance(topology, TGraph):
                    return TGraph.OntologyTriples(
                        topology,
                        includeVertices=includeGraph,
                        includeEdges=includeGraph,
                        includeDictionaries=includeDictionaries,
                        includeBOT=includeBOT,
                        namespacePrefix=namespacePrefix,
                    )
            except Exception:
                pass
        if Ontology is not None:
            try:
                if includeGraph and hasattr(Ontology, "GraphTriples"):
                    return Ontology.GraphTriples(
                        topology,
                        includeVertices=True,
                        includeEdges=True,
                        includeDictionaries=includeDictionaries,
                        includeBOT=includeBOT,
                        namespacePrefix=namespacePrefix,
                        silent=silent,
                    )
            except Exception:
                pass
            try:
                return Ontology.Triples(
                    topology,
                    includeDictionaries=includeDictionaries,
                    includeBOT=includeBOT,
                    namespacePrefix=namespacePrefix,
                    silent=silent,
                )
            except Exception:
                pass
        return []

    @staticmethod
    def RDFGraphByTriples(triples: Iterable[Tuple[Any, Any, Any]], silent: bool = False):
        """Builds an RDFLib graph from TopologicPy RDF-like triples."""
        rd = Reasoner._rdflib(silent=silent)
        if rd is None:
            return None
        g = rd["Graph"]()
        Reasoner._bind_namespaces(g)
        URIRef = rd["URIRef"]
        Literal = rd["Literal"]
        RDF = rd["RDF"]
        RDFS = rd["RDFS"]
        OWL = rd["OWL"]
        XSD = rd["XSD"]

        special = {
            "rdf:type": RDF.type,
            "rdfs:label": RDFS.label,
            "rdfs:comment": RDFS.comment,
            "rdfs:subClassOf": RDFS.subClassOf,
            "rdfs:subPropertyOf": RDFS.subPropertyOf,
            "rdfs:domain": RDFS.domain,
            "rdfs:range": RDFS.range,
            "owl:Class": OWL.Class,
        }

        def node(value, predicate_position: bool = False, object_position: bool = False):
            if value is None:
                return None
            if hasattr(value, "n3"):
                return value
            if isinstance(value, str):
                text = value.strip()
                if text.startswith('"'):
                    # Minimal parser for literals produced by Ontology/TGraph helpers.
                    try:
                        body = text[1:text.rfind('"')]
                        if "^^" in text:
                            dtype = text.split("^^", 1)[1]
                            return Literal(body, datatype=Reasoner._uri_ref(dtype, URIRef=URIRef))
                        return Literal(body)
                    except Exception:
                        return Literal(text.strip('"'))
                if object_position and not predicate_position:
                    # Non-QName strings in object position are literals; QNames/URIs are resources.
                    if ":" not in text and not text.startswith("http"):
                        return Literal(text)
                expanded = Reasoner.ExpandQName(text, defaultValue=None)
                if expanded is not None:
                    return URIRef(expanded)
                if text.startswith("http://") or text.startswith("https://"):
                    return URIRef(text)
                return URIRef(text) if predicate_position else Literal(text)
            if isinstance(value, bool):
                return Literal(value, datatype=XSD.boolean)
            if isinstance(value, int):
                return Literal(value, datatype=XSD.integer)
            if isinstance(value, float):
                return Literal(value, datatype=XSD.double)
            return Literal(value)

        for s, p, o in triples or []:
            ss = node(s)
            pp = special.get(p, node(p, predicate_position=True)) if isinstance(p, str) else node(p, predicate_position=True)
            oo = special.get(o, node(o, object_position=True)) if isinstance(o, str) else node(o, object_position=True)
            if ss is not None and pp is not None and oo is not None:
                g.add((ss, pp, oo))
        return g

    @staticmethod
    def AddOntologyAxioms(rdfGraph, includeBOT: bool = True, silent: bool = False):
        """
        Adds TopologicPy ontology schema triples to an RDFLib graph.

        This method injects the subclass hierarchy, BOT bridge mappings, object
        property domain/range declarations, data property domain/range
        declarations, and property aliases as sub-property declarations.
        """
        rd = Reasoner._rdflib(silent=silent)
        Ontology = Reasoner._ontology_class()
        if rd is None or rdfGraph is None or Ontology is None:
            return rdfGraph

        URIRef = rd["URIRef"]
        Literal = rd["Literal"]
        RDF = rd["RDF"]
        RDFS = rd["RDFS"]
        OWL = rd["OWL"]

        def U(term):
            return Reasoner._uri_ref(term, URIRef=URIRef)

        Reasoner._bind_namespaces(rdfGraph)

        try:
            class_map = getattr(Ontology, "TOP_SUPERCLASSES", {}) or {}
            for cls, supers in class_map.items():
                c = U(cls)
                if c is None:
                    continue
                rdfGraph.add((c, RDF.type, OWL.Class))
                for sup in supers or []:
                    s = U(sup)
                    if s is not None:
                        rdfGraph.add((c, RDFS.subClassOf, s))

            if includeBOT:
                for top_cls, bot_cls in (getattr(Ontology, "TOP_TO_BOT", {}) or {}).items():
                    tc, bc = U(top_cls), U(bot_cls)
                    if tc is not None and bc is not None:
                        rdfGraph.add((tc, RDFS.subClassOf, bc))

            for prop, spec in (getattr(Ontology, "OBJECT_PROPERTIES", {}) or {}).items():
                domain, range_, comment = spec
                p = U(prop)
                if p is None:
                    continue
                rdfGraph.add((p, RDF.type, OWL.ObjectProperty))
                if domain:
                    rdfGraph.add((p, RDFS.domain, U(domain)))
                if range_:
                    rdfGraph.add((p, RDFS.range, U(range_)))
                if comment:
                    rdfGraph.add((p, RDFS.comment, Literal(str(comment))))

            for prop, spec in (getattr(Ontology, "DATA_PROPERTIES", {}) or {}).items():
                domain, range_, comment = spec
                p = U(prop)
                if p is None:
                    continue
                rdfGraph.add((p, RDF.type, OWL.DatatypeProperty))
                if domain:
                    rdfGraph.add((p, RDFS.domain, U(domain)))
                if range_:
                    rdfGraph.add((p, RDFS.range, U(range_)))
                if comment:
                    rdfGraph.add((p, RDFS.comment, Literal(str(comment))))

            aliases = getattr(Ontology, "PROPERTY_ALIASES", {}) or {}
            for alias, canonical in aliases.items():
                alias_q = alias if ":" in str(alias) else "top:" + str(alias)
                canonical_q = canonical if ":" in str(canonical) else "top:" + str(canonical)
                ap, cp = U(alias_q), U(canonical_q)
                if ap is not None and cp is not None:
                    rdfGraph.add((ap, RDFS.subPropertyOf, cp))
        except Exception as exc:
            if not silent:
                print("Reasoner.AddOntologyAxioms - Warning: Could not add all ontology axioms.")
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
        """A small deterministic RDFS closure implementation."""
        rd = Reasoner._rdflib(silent=silent)
        if rd is None or rdfGraph is None:
            return rdfGraph
        RDF = rd["RDF"]
        RDFS = rd["RDFS"]
        URIRef = rd["URIRef"]

        subclass = RDFS.subClassOf
        subprop = RDFS.subPropertyOf
        domain = RDFS.domain
        range_ = RDFS.range
        rdf_type = RDF.type

        def add_triples(triples: Iterable[Tuple[Any, Any, Any]]) -> int:
            count = 0
            for t in triples:
                if t not in rdfGraph:
                    rdfGraph.add(t)
                    count += 1
            return count

        iterations = max(1, int(maxIterations))
        for _ in range(iterations):
            new: Set[Tuple[Any, Any, Any]] = set()

            # rdfs:subClassOf transitivity
            for c, _, p in list(rdfGraph.triples((None, subclass, None))):
                for _, _, gp in rdfGraph.triples((p, subclass, None)):
                    if gp != c:
                        new.add((c, subclass, gp))

            # rdf:type inheritance
            for s, _, c in list(rdfGraph.triples((None, rdf_type, None))):
                for _, _, sup in rdfGraph.triples((c, subclass, None)):
                    new.add((s, rdf_type, sup))

            # rdfs:subPropertyOf transitivity
            for p, _, q in list(rdfGraph.triples((None, subprop, None))):
                for _, _, r in rdfGraph.triples((q, subprop, None)):
                    if r != p:
                        new.add((p, subprop, r))

            # Predicate inheritance
            subproperties = list(rdfGraph.triples((None, subprop, None)))
            for p, _, q in subproperties:
                for s, _, o in list(rdfGraph.triples((None, p, None))):
                    new.add((s, q, o))

            # Domain/range inference
            properties = set(p for _, p, _ in rdfGraph)
            for p in list(properties):
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

    # ---------------------------------------------------------------------
    # Query and reporting helpers
    # ---------------------------------------------------------------------

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
            if (s, p, o) not in beforeGraph:
                if compact:
                    so = Reasoner.QName(s)
                    po = Reasoner.QName(p)
                    oo = Reasoner.QName(o) if not getattr(o, "datatype", None) else str(o)
                    diff.append((so, po, oo))
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
        TGraph = Reasoner._tgraph_class()
        if TGraph is not None:
            try:
                return TGraph._OntologySubjectFromDictionary(dictionary, fallback, namespacePrefix=namespacePrefix)
            except Exception:
                pass
        uri = dictionary.get("uri") if isinstance(dictionary, dict) else None
        if isinstance(uri, str) and uri.strip() != "":
            return uri if ":" in uri else namespacePrefix + ":" + uri
        label = dictionary.get("label") if isinstance(dictionary, dict) else None
        text = str(label if label not in [None, ""] else fallback)
        safe = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in text).strip("_")
        return namespacePrefix + ":" + (safe or fallback)

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
        """
        Writes inferred rdf:type values back into TGraph dictionaries.

        The canonical asserted ontology_class is not overwritten. Inferred classes
        are stored as lists under `inferred_ontology_classes` and BOT bridge types
        under `inferred_bot_classes`.
        """
        TGraph = Reasoner._tgraph_class()
        if TGraph is None or graph is None or inferredGraph is None:
            return graph
        try:
            if not isinstance(graph, TGraph):
                return graph
        except Exception:
            return graph

        def split_types(types: Sequence[str]) -> Tuple[List[str], List[str]]:
            top_types = sorted(t for t in set(types) if isinstance(t, str) and t.startswith("top:"))
            bot_types = sorted(t for t in set(types) if isinstance(t, str) and t.startswith("bot:"))
            return top_types, bot_types

        def apply_to_dict(d: Dict[str, Any], fallback: str):
            subject = Reasoner._subject_for_dictionary(d, fallback, namespacePrefix=namespacePrefix)
            types = Reasoner._types_for_subject(inferredGraph, subject)
            top_types, bot_types = split_types(types)
            asserted = d.get("ontology_class")
            inferred_top = [t for t in top_types if t != asserted]
            if overwrite or typeKey not in d:
                d[typeKey] = inferred_top
            else:
                d[typeKey] = sorted(set(list(d.get(typeKey, [])) + inferred_top))
            if bot_types:
                if overwrite or botTypeKey not in d:
                    d[botTypeKey] = bot_types
                else:
                    d[botTypeKey] = sorted(set(list(d.get(botTypeKey, [])) + bot_types))

        try:
            if includeGraph:
                apply_to_dict(graph._dictionary, "graph")
            if includeVertices:
                for v in graph._vertices:
                    if v.get("active", True):
                        apply_to_dict(v.get("dictionary", {}), f"vertex_{v.get('index')}")
            if includeEdges:
                for e in graph._edges:
                    if e.get("active", True):
                        apply_to_dict(e.get("dictionary", {}), f"edge_{e.get('index')}")
        except Exception as exc:
            if not silent:
                print("Reasoner.ApplyInferences - Warning: Could not apply all inferences to the TGraph dictionaries.")
                print("Error:", exc)
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
