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
import os
import re
import json
import copy


class KnowledgeGraph:
    """
    A lightweight RDF/knowledge-graph bridge for TopologicPy.

    KnowledgeGraph is intentionally dependency-light. If RDFLib is installed,
    it can parse/serialise RDF, run SPARQL queries, and expose an RDFLib Graph.
    If RDFLib is not installed, it still behaves as a deterministic triple store
    and can export simple Turtle generated from QName/literal triples.

    This class is designed as the intermediate semantic layer between:

    - Ontology.py: vocabulary, namespaces, dictionary-key mappings, TTL export.
    - TGraph.py: geometric/topological graph data structures.
    - Reasoner.py: inference, proof tracking, and explanations.

    Triple representation
    ---------------------
    Internally, triples are stored as compact Turtle-like tokens:

    - resources are stored as QNames when possible, e.g. ``top:Room``;
    - absolute URIs are stored as ``<https://example.org/id>`` when they cannot
      be compacted;
    - literals are stored as legal Turtle literal tokens, e.g.
      ``"Kitchen"`` or ``"12.3"^^xsd:double``.

    This makes diffs, tests, and serialisation stable even without RDFLib.
    """

    # ---------------------------------------------------------------------
    # Construction and representation
    # ---------------------------------------------------------------------

    def __init__(
        self,
        triples: Optional[Iterable[Tuple[Any, Any, Any]]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        rdfGraph: Any = None,
        useRDFLib: bool = True,
        silent: bool = False,
    ):
        """
        Initializes a KnowledgeGraph.

        Parameters
        ----------
        triples : iterable(tuple), optional
            Input triples. Each triple must be ``(subject, predicate, object)``.
        namespaces : dict, optional
            Namespace prefix-to-URI mapping. If omitted, the mapping is taken
            from Ontology.py when available.
        rdfGraph : rdflib.Graph, optional
            Existing RDFLib graph to wrap/import.
        useRDFLib : bool, optional
            If True, RDFLib is used when available. Default is True.
        silent : bool, optional
            If True, warning/error messages are suppressed. Default is False.
        """

        self._namespaces: Dict[str, str] = KnowledgeGraph.Namespaces(namespaces)
        self._triples: Set[Tuple[str, str, str]] = set()
        self._rdflib_enabled: bool = bool(useRDFLib)
        self._rdf_graph: Any = None

        if rdfGraph is not None:
            self._rdf_graph = rdfGraph
            self._rdflib_enabled = True
            self._bind_namespaces(self._rdf_graph)
            self._sync_triples_from_rdflib(silent=silent)
        else:
            if triples is not None:
                self.AddTriples(triples, silent=silent)
            if self._rdflib_enabled and KnowledgeGraph._rdflib(silent=True) is not None:
                self._sync_rdflib_from_triples(silent=silent)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(triples={len(self._triples)}, namespaces={len(self._namespaces)})"

    def __len__(self) -> int:
        return len(self._triples)

    def __iter__(self):
        for triple in self.Triples(sort=True):
            yield triple

    def __contains__(self, triple: Tuple[Any, Any, Any]) -> bool:
        return self.HasTriple(triple)

    # ---------------------------------------------------------------------
    # Dependency helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def Dependencies() -> Dict[str, Any]:
        """
        Returns availability/version information for optional dependencies.

        Returns
        -------
        dict
            A dictionary describing optional dependency availability.
        """

        report = {}
        for name in ["rdflib"]:
            try:
                module = __import__(name)
                report[name] = {"available": True, "version": getattr(module, "__version__", None)}
            except Exception as exc:
                report[name] = {"available": False, "error": str(exc)}
        return report

    @staticmethod
    def _rdflib(silent: bool = False):
        """Returns RDFLib objects when RDFLib is available."""
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
                print("KnowledgeGraph - Warning: RDFLib is not available. Some operations are disabled.")
                print("Error:", exc)
            return None

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
    def _reasoner_class():
        try:
            from topologicpy.Reasoner import Reasoner
            return Reasoner
        except Exception:
            try:
                from Reasoner import Reasoner
                return Reasoner
            except Exception:
                return None

    # ---------------------------------------------------------------------
    # Namespace and term helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def Namespaces(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Returns the namespace dictionary used by TopologicPy knowledge graphs.

        Parameters
        ----------
        extra : dict, optional
            Additional or overriding namespaces.

        Returns
        -------
        dict
            Prefix-to-namespace-URI mapping.
        """

        Ontology = KnowledgeGraph._ontology_class()
        if Ontology is not None:
            try:
                ns = dict(Ontology.NAMESPACES)
            except Exception:
                ns = {}
        else:
            ns = {}
        ns.setdefault("bot", "https://w3id.org/bot#")
        ns.setdefault("brick", "https://brickschema.org/schema/Brick#")
        ns.setdefault("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        ns.setdefault("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        ns.setdefault("xsd", "http://www.w3.org/2001/XMLSchema#")
        ns.setdefault("owl", "http://www.w3.org/2002/07/owl#")
        ns.setdefault("top", "http://w3id.org/topologicpy#")
        ns.setdefault("inst", "http://w3id.org/topologicpy/instance#")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key is not None and value is not None:
                    ns[str(key).strip()] = str(value).strip()
        return ns

    def Namespace(self, prefix: str, defaultValue: Any = None) -> Any:
        """Returns the namespace URI for the input prefix."""
        if prefix is None:
            return defaultValue
        return self._namespaces.get(str(prefix).strip(), defaultValue)

    @staticmethod
    def ExpandQName(term: Any, namespaces: Optional[Dict[str, str]] = None, defaultValue: Any = None) -> Any:
        """
        Expands a QName such as ``top:Room`` to a full URI string.

        Parameters
        ----------
        term : any
            Input term.
        namespaces : dict, optional
            Namespace dictionary. If omitted, TopologicPy namespaces are used.
        defaultValue : any, optional
            Value returned if expansion fails.

        Returns
        -------
        any
            Expanded URI string or defaultValue.
        """

        if term is None:
            return defaultValue
        text = str(term).strip()
        if text.startswith("<") and text.endswith(">"):
            return text[1:-1]
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if ":" not in text:
            return defaultValue
        prefix, local = text.split(":", 1)
        ns = KnowledgeGraph.Namespaces(namespaces).get(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    @staticmethod
    def QName(uri: Any, namespaces: Optional[Dict[str, str]] = None, defaultValue: Any = None) -> Any:
        """
        Compacts a URI string/URIRef to a QName when possible.

        Parameters
        ----------
        uri : any
            Input URI string or URIRef.
        namespaces : dict, optional
            Namespace dictionary. If omitted, TopologicPy namespaces are used.
        defaultValue : any, optional
            Value returned if compaction fails.

        Returns
        -------
        any
            QName or defaultValue/string.
        """

        if uri is None:
            return defaultValue
        text = str(uri).strip()
        if text.startswith("<") and text.endswith(">"):
            text = text[1:-1]
        ns = KnowledgeGraph.Namespaces(namespaces)
        # Prefer longer namespaces first to avoid accidental prefix shortening.
        for prefix, namespace in sorted(ns.items(), key=lambda item: len(item[1]), reverse=True):
            if text.startswith(namespace):
                local = text[len(namespace):]
                if local:
                    return prefix + ":" + local
        return defaultValue if defaultValue is not None else text

    @staticmethod
    def _safe_local_name(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.strip()
        text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if text == "":
            text = "item"
        if text[0].isdigit():
            text = "id_" + text
        return text

    @staticmethod
    def _escape_literal(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\\", "\\\\").replace("\"", "\\\"")
        text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return text

    @staticmethod
    def _looks_like_qname(text: str, namespaces: Dict[str, str]) -> bool:
        if not isinstance(text, str) or ":" not in text:
            return False
        prefix, local = text.split(":", 1)
        return prefix in namespaces and local != ""

    @staticmethod
    def _looks_like_uri(text: str) -> bool:
        if not isinstance(text, str):
            return False
        text = text.strip()
        return text.startswith("http://") or text.startswith("https://") or (text.startswith("<") and text.endswith(">"))

    @staticmethod
    def _looks_like_literal_token(text: str) -> bool:
        if not isinstance(text, str):
            return False
        text = text.strip()
        return text.startswith('"') or text.startswith("'''") or text.startswith('"""')

    @staticmethod
    def _literal_token(value: Any, datatype: Optional[str] = None, language: Optional[str] = None) -> str:
        if isinstance(value, bool):
            return '"' + ("true" if value else "false") + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float):
            return '"' + repr(value) + '"^^xsd:double'
        literal = '"' + KnowledgeGraph._escape_literal(value) + '"'
        if language is not None:
            literal += "@" + str(language).strip()
        elif datatype is not None:
            literal += "^^" + str(datatype).strip()
        return literal

    @staticmethod
    def NormalizeTerm(
        value: Any,
        role: str = "object",
        namespaces: Optional[Dict[str, str]] = None,
        literal: bool = False,
        datatype: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Normalizes a subject, predicate, or object to a compact token.

        Parameters
        ----------
        value : any
            Input value.
        role : str, optional
            ``"subject"``, ``"predicate"``, or ``"object"``. Default is ``"object"``.
        namespaces : dict, optional
            Namespace dictionary.
        literal : bool, optional
            If True, the value is forced to be encoded as a literal. Default is False.
        datatype : str, optional
            Datatype QName for forced literals.
        language : str, optional
            Language tag for forced string literals.

        Returns
        -------
        str or None
            Normalized compact token.
        """

        if value is None:
            return None

        rd = KnowledgeGraph._rdflib(silent=True)
        if rd is not None:
            try:
                if isinstance(value, rd["URIRef"]):
                    q = KnowledgeGraph.QName(value, namespaces=namespaces, defaultValue=None)
                    return q if q is not None else "<" + str(value) + ">"
                if isinstance(value, rd["Literal"]):
                    try:
                        return value.n3()
                    except Exception:
                        return KnowledgeGraph._literal_token(str(value))
                if isinstance(value, rd["BNode"]):
                    return "_:" + str(value)
            except Exception:
                pass

        ns = KnowledgeGraph.Namespaces(namespaces)
        role = str(role or "object").lower()

        if literal or isinstance(value, (bool, int, float)):
            return KnowledgeGraph._literal_token(value, datatype=datatype, language=language)

        text = str(value).strip()
        if text == "":
            return None

        if KnowledgeGraph._looks_like_literal_token(text):
            return text

        if text.startswith("_:"):
            return text

        if text.startswith("<") and text.endswith(">"):
            return text

        if text.startswith("http://") or text.startswith("https://"):
            q = KnowledgeGraph.QName(text, namespaces=ns, defaultValue=None)
            return q if q is not None else "<" + text + ">"

        if KnowledgeGraph._looks_like_qname(text, ns):
            return text

        if role in ["subject", "predicate"]:
            prefix = "top" if role == "predicate" else "inst"
            return prefix + ":" + KnowledgeGraph._safe_local_name(text)

        return KnowledgeGraph._literal_token(text, datatype=datatype, language=language)

    @staticmethod
    def _is_literal_token(value: Any) -> bool:
        return isinstance(value, str) and KnowledgeGraph._looks_like_literal_token(value.strip())

    @staticmethod
    def _is_resource_token(value: Any, namespaces: Optional[Dict[str, str]] = None) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        ns = KnowledgeGraph.Namespaces(namespaces)
        return (
            text.startswith("<") and text.endswith(">")
            or text.startswith("http://")
            or text.startswith("https://")
            or text.startswith("_:")
            or KnowledgeGraph._looks_like_qname(text, ns)
        )

    @staticmethod
    def _strip_literal_quotes(token: str) -> str:
        text = str(token).strip()
        if not text.startswith('"'):
            return text
        # Minimal N3 literal unquoting; enough for labels/reports.
        escaped = False
        chars = []
        for ch in text[1:]:
            if escaped:
                mapping = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
                chars.append(mapping.get(ch, ch))
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                break
            else:
                chars.append(ch)
        return "".join(chars)

    # ---------------------------------------------------------------------
    # RDFLib conversion and synchronization
    # ---------------------------------------------------------------------

    def _bind_namespaces(self, rdfGraph: Any) -> Any:
        if rdfGraph is None:
            return None
        try:
            for prefix, uri in self._namespaces.items():
                if prefix and uri:
                    rdfGraph.bind(prefix, uri)
        except Exception:
            pass
        return rdfGraph

    def _term_to_rdflib(self, token: Any, role: str = "object") -> Any:
        rd = KnowledgeGraph._rdflib(silent=True)
        if rd is None or token is None:
            return None

        URIRef = rd["URIRef"]
        Literal = rd["Literal"]
        BNode = rd["BNode"]

        if isinstance(token, (URIRef, Literal, BNode)):
            return token

        text = str(token).strip()
        if text.startswith("_:"):
            return BNode(text[2:])

        if role == "object" and KnowledgeGraph._is_literal_token(text):
            return self._literal_token_to_rdflib(text)

        uri = KnowledgeGraph.ExpandQName(text, namespaces=self._namespaces, defaultValue=None)
        if uri is not None:
            return URIRef(uri)
        if text.startswith("<") and text.endswith(">"):
            return URIRef(text[1:-1])
        if text.startswith("http://") or text.startswith("https://"):
            return URIRef(text)

        if role == "object":
            return Literal(text)
        return URIRef(KnowledgeGraph.ExpandQName("inst:" + KnowledgeGraph._safe_local_name(text), namespaces=self._namespaces))

    def _literal_token_to_rdflib(self, token: str) -> Any:
        rd = KnowledgeGraph._rdflib(silent=True)
        if rd is None:
            return None
        Literal = rd["Literal"]
        URIRef = rd["URIRef"]

        text = str(token).strip()
        # Capture "lexical"@lang or "lexical"^^datatype. This is deliberately
        # conservative and avoids a mandatory N3 parser.
        match = re.match(r'^"((?:\\.|[^"\\])*)"(@[A-Za-z0-9\-]+|\^\^.+)?$', text)
        if not match:
            return Literal(KnowledgeGraph._strip_literal_quotes(text))
        lexical = bytes(match.group(1), "utf-8").decode("unicode_escape")
        suffix = match.group(2)
        if suffix is None:
            return Literal(lexical)
        if suffix.startswith("@"):
            return Literal(lexical, lang=suffix[1:])
        if suffix.startswith("^^"):
            dt_token = suffix[2:].strip()
            dt_uri = KnowledgeGraph.ExpandQName(dt_token, namespaces=self._namespaces, defaultValue=None)
            if dt_uri is None and dt_token.startswith("<") and dt_token.endswith(">"):
                dt_uri = dt_token[1:-1]
            if dt_uri is not None:
                return Literal(lexical, datatype=URIRef(dt_uri))
        return Literal(lexical)

    def _rdflib_to_token(self, term: Any) -> str:
        rd = KnowledgeGraph._rdflib(silent=True)
        if rd is None:
            return KnowledgeGraph.NormalizeTerm(term, namespaces=self._namespaces)
        try:
            if isinstance(term, rd["URIRef"]):
                q = KnowledgeGraph.QName(term, namespaces=self._namespaces, defaultValue=None)
                return q if q is not None else "<" + str(term) + ">"
            if isinstance(term, rd["BNode"]):
                return "_:" + str(term)
            if isinstance(term, rd["Literal"]):
                lexical = KnowledgeGraph._literal_token(str(term))
                if term.language:
                    return KnowledgeGraph._literal_token(str(term), language=str(term.language))
                if term.datatype:
                    dt = KnowledgeGraph.QName(term.datatype, namespaces=self._namespaces, defaultValue=None)
                    if dt is None:
                        dt = "<" + str(term.datatype) + ">"
                    return KnowledgeGraph._literal_token(str(term), datatype=dt)
                return lexical
        except Exception:
            pass
        return str(term)

    def _sync_rdflib_from_triples(self, silent: bool = False) -> Any:
        if not self._rdflib_enabled:
            return None
        rd = KnowledgeGraph._rdflib(silent=silent)
        if rd is None:
            self._rdf_graph = None
            return None
        graph = rd["Graph"]()
        self._bind_namespaces(graph)
        for s, p, o in self._triples:
            rs = self._term_to_rdflib(s, role="subject")
            rp = self._term_to_rdflib(p, role="predicate")
            ro = self._term_to_rdflib(o, role="object")
            if rs is not None and rp is not None and ro is not None:
                try:
                    graph.add((rs, rp, ro))
                except Exception as exc:
                    if not silent:
                        print("KnowledgeGraph - Warning: Could not add triple to RDFLib graph:", (s, p, o))
                        print("Error:", exc)
        self._rdf_graph = graph
        return graph

    def _sync_triples_from_rdflib(self, silent: bool = False) -> Set[Tuple[str, str, str]]:
        self._triples = set()
        if self._rdf_graph is None:
            return self._triples
        try:
            for prefix, uri in self._rdf_graph.namespaces():
                if prefix and uri:
                    self._namespaces[str(prefix)] = str(uri)
        except Exception:
            pass
        try:
            for s, p, o in self._rdf_graph:
                self._triples.add((self._rdflib_to_token(s), self._rdflib_to_token(p), self._rdflib_to_token(o)))
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph - Error: Could not import triples from RDFLib graph.")
                print("Error:", exc)
        return self._triples

    # ---------------------------------------------------------------------
    # Core triple-store API
    # ---------------------------------------------------------------------

    def AddTriple(
        self,
        subject: Any,
        predicate: Any,
        object: Any,
        objectIsLiteral: bool = False,
        datatype: Optional[str] = None,
        language: Optional[str] = None,
        silent: bool = False,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Adds one triple and returns the normalized triple.

        Parameters
        ----------
        subject : any
            Triple subject.
        predicate : any
            Triple predicate.
        object : any
            Triple object.
        objectIsLiteral : bool, optional
            If True, the object is encoded as a literal. Default is False.
        datatype : str, optional
            Datatype QName for literal objects.
        language : str, optional
            Language tag for literal objects.
        silent : bool, optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        tuple or None
            Normalized triple, or None if invalid.
        """

        s = KnowledgeGraph.NormalizeTerm(subject, role="subject", namespaces=self._namespaces)
        p = KnowledgeGraph.NormalizeTerm(predicate, role="predicate", namespaces=self._namespaces)
        o = KnowledgeGraph.NormalizeTerm(object, role="object", namespaces=self._namespaces, literal=objectIsLiteral, datatype=datatype, language=language)
        if s is None or p is None or o is None:
            if not silent:
                print("KnowledgeGraph.AddTriple - Error: Invalid subject, predicate, or object. Returning None.")
            return None
        triple = (s, p, o)
        self._triples.add(triple)
        if self._rdflib_enabled:
            if self._rdf_graph is None:
                self._sync_rdflib_from_triples(silent=True)
            else:
                try:
                    rs = self._term_to_rdflib(s, role="subject")
                    rp = self._term_to_rdflib(p, role="predicate")
                    ro = self._term_to_rdflib(o, role="object")
                    self._rdf_graph.add((rs, rp, ro))
                except Exception:
                    self._sync_rdflib_from_triples(silent=True)
        return triple

    def AddTriples(self, triples: Iterable[Tuple[Any, Any, Any]], silent: bool = False) -> int:
        """
        Adds multiple triples and returns the number of newly added triples.
        """
        before = len(self._triples)
        for triple in triples or []:
            if triple is None or len(triple) != 3:
                if not silent:
                    print("KnowledgeGraph.AddTriples - Warning: Skipping invalid triple:", triple)
                continue
            self.AddTriple(triple[0], triple[1], triple[2], silent=silent)
        return len(self._triples) - before

    def RemoveTriple(self, subject: Any, predicate: Any, object: Any, silent: bool = False) -> bool:
        """Removes a triple and returns True if it was present."""
        s = KnowledgeGraph.NormalizeTerm(subject, role="subject", namespaces=self._namespaces)
        p = KnowledgeGraph.NormalizeTerm(predicate, role="predicate", namespaces=self._namespaces)
        o = KnowledgeGraph.NormalizeTerm(object, role="object", namespaces=self._namespaces)
        triple = (s, p, o)
        existed = triple in self._triples
        if existed:
            self._triples.remove(triple)
            if self._rdf_graph is not None:
                try:
                    self._rdf_graph.remove((self._term_to_rdflib(s, "subject"), self._term_to_rdflib(p, "predicate"), self._term_to_rdflib(o, "object")))
                except Exception:
                    self._sync_rdflib_from_triples(silent=True)
        elif not silent:
            print("KnowledgeGraph.RemoveTriple - Warning: Triple was not present.")
        return existed

    def Clear(self) -> "KnowledgeGraph":
        """Removes all triples and returns self."""
        self._triples.clear()
        if self._rdf_graph is not None:
            try:
                self._rdf_graph.remove((None, None, None))
            except Exception:
                self._sync_rdflib_from_triples(silent=True)
        return self

    def Triples(
        self,
        subject: Any = None,
        predicate: Any = None,
        object: Any = None,
        sort: bool = True,
    ) -> List[Tuple[str, str, str]]:
        """
        Returns triples, optionally filtered by subject, predicate, and object.
        """
        s = KnowledgeGraph.NormalizeTerm(subject, role="subject", namespaces=self._namespaces) if subject is not None else None
        p = KnowledgeGraph.NormalizeTerm(predicate, role="predicate", namespaces=self._namespaces) if predicate is not None else None
        o = KnowledgeGraph.NormalizeTerm(object, role="object", namespaces=self._namespaces) if object is not None else None
        result = []
        for triple in self._triples:
            if s is not None and triple[0] != s:
                continue
            if p is not None and triple[1] != p:
                continue
            if o is not None and triple[2] != o:
                continue
            result.append(triple)
        return sorted(result) if sort else result

    def HasTriple(self, triple: Union[Tuple[Any, Any, Any], List[Any]]) -> bool:
        """Returns True if the normalized triple is present."""
        if triple is None or len(triple) != 3:
            return False
        s = KnowledgeGraph.NormalizeTerm(triple[0], role="subject", namespaces=self._namespaces)
        p = KnowledgeGraph.NormalizeTerm(triple[1], role="predicate", namespaces=self._namespaces)
        o = KnowledgeGraph.NormalizeTerm(triple[2], role="object", namespaces=self._namespaces)
        return (s, p, o) in self._triples

    def Subjects(self, predicate: Any = None, object: Any = None, sort: bool = True) -> List[str]:
        """Returns unique subjects, optionally filtered by predicate/object."""
        return sorted({s for s, p, o in self.Triples(predicate=predicate, object=object, sort=False)}) if sort else list({s for s, p, o in self.Triples(predicate=predicate, object=object, sort=False)})

    def Predicates(self, subject: Any = None, object: Any = None, sort: bool = True) -> List[str]:
        """Returns unique predicates, optionally filtered by subject/object."""
        return sorted({p for s, p, o in self.Triples(subject=subject, object=object, sort=False)}) if sort else list({p for s, p, o in self.Triples(subject=subject, object=object, sort=False)})

    def Objects(self, subject: Any = None, predicate: Any = None, sort: bool = True) -> List[str]:
        """Returns unique objects, optionally filtered by subject/predicate."""
        return sorted({o for s, p, o in self.Triples(subject=subject, predicate=predicate, sort=False)}) if sort else list({o for s, p, o in self.Triples(subject=subject, predicate=predicate, sort=False)})

    def Resources(self, includePredicates: bool = False, includeLiterals: bool = False, sort: bool = True) -> List[str]:
        """Returns resources used in the graph."""
        values = set()
        for s, p, o in self._triples:
            values.add(s)
            if includePredicates:
                values.add(p)
            if includeLiterals or not KnowledgeGraph._is_literal_token(o):
                values.add(o)
        return sorted(values) if sort else list(values)

    def Literals(self, sort: bool = True) -> List[str]:
        """Returns literal object tokens used in the graph."""
        result = {o for _, _, o in self._triples if KnowledgeGraph._is_literal_token(o)}
        return sorted(result) if sort else list(result)

    def Copy(self) -> "KnowledgeGraph":
        """Returns a deep copy of this KnowledgeGraph."""
        return KnowledgeGraph(triples=list(self._triples), namespaces=dict(self._namespaces), useRDFLib=self._rdflib_enabled, silent=True)

    def Dictionary(self) -> Dict[str, Any]:
        """Returns a JSON-friendly dictionary representation."""
        return {
            "type": "KnowledgeGraph",
            "namespaces": dict(self._namespaces),
            "triples": [list(t) for t in self.Triples(sort=True)],
        }

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def ByTriples(
        triples: Iterable[Tuple[Any, Any, Any]],
        namespaces: Optional[Dict[str, str]] = None,
        useRDFLib: bool = True,
        silent: bool = False,
    ) -> "KnowledgeGraph":
        """Creates a KnowledgeGraph from triples."""
        return KnowledgeGraph(triples=triples, namespaces=namespaces, useRDFLib=useRDFLib, silent=silent)

    @staticmethod
    def ByDictionary(data: Dict[str, Any], useRDFLib: bool = True, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Creates a KnowledgeGraph from a dictionary returned by ``Dictionary``."""
        if not isinstance(data, dict):
            if not silent:
                print("KnowledgeGraph.ByDictionary - Error: The input data is not a dictionary. Returning None.")
            return None
        return KnowledgeGraph(
            triples=[tuple(t) for t in data.get("triples", []) if isinstance(t, (list, tuple)) and len(t) == 3],
            namespaces=data.get("namespaces", None),
            useRDFLib=useRDFLib,
            silent=silent,
        )

    @staticmethod
    def ByJSON(jsonString: str, useRDFLib: bool = True, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Creates a KnowledgeGraph from a JSON string."""
        try:
            data = json.loads(jsonString)
            return KnowledgeGraph.ByDictionary(data, useRDFLib=useRDFLib, silent=silent)
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.ByJSON - Error: Could not parse JSON. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def ByRDFGraph(rdfGraph: Any, namespaces: Optional[Dict[str, str]] = None, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Creates a KnowledgeGraph from an RDFLib graph."""
        if rdfGraph is None:
            if not silent:
                print("KnowledgeGraph.ByRDFGraph - Error: The input RDF graph is None. Returning None.")
            return None
        return KnowledgeGraph(rdfGraph=rdfGraph, namespaces=namespaces, useRDFLib=True, silent=silent)

    @staticmethod
    def ByTurtleString(ttlString: str, format: str = "turtle", namespaces: Optional[Dict[str, str]] = None, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Creates a KnowledgeGraph by parsing a Turtle/RDF string with RDFLib."""
        rd = KnowledgeGraph._rdflib(silent=silent)
        if rd is None:
            return None
        if ttlString is None:
            if not silent:
                print("KnowledgeGraph.ByTurtleString - Error: The input string is None. Returning None.")
            return None
        try:
            g = rd["Graph"]()
            kg = KnowledgeGraph(namespaces=namespaces, useRDFLib=True, silent=True)
            kg._bind_namespaces(g)
            g.parse(data=ttlString, format=format)
            return KnowledgeGraph.ByRDFGraph(g, namespaces=kg._namespaces, silent=silent)
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.ByTurtleString - Error: Could not parse RDF string. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def ByTTLString(ttlString: str, **kwargs) -> Optional["KnowledgeGraph"]:
        """Alias for ByTurtleString."""
        return KnowledgeGraph.ByTurtleString(ttlString, **kwargs)

    @staticmethod
    def ByFile(path: str, format: Optional[str] = None, namespaces: Optional[Dict[str, str]] = None, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Creates a KnowledgeGraph by parsing an RDF/Turtle/JSON-LD/N-Triples file."""
        rd = KnowledgeGraph._rdflib(silent=silent)
        if rd is None:
            return None
        if path is None or not os.path.exists(path):
            if not silent:
                print("KnowledgeGraph.ByFile - Error: The input path does not exist. Returning None.")
            return None
        if format is None:
            ext = os.path.splitext(path)[1].lower()
            format = {
                ".ttl": "turtle",
                ".rdf": "xml",
                ".owl": "xml",
                ".xml": "xml",
                ".nt": "nt",
                ".n3": "n3",
                ".jsonld": "json-ld",
                ".json": "json-ld",
                ".trig": "trig",
            }.get(ext, "turtle")
        try:
            g = rd["Graph"]()
            kg = KnowledgeGraph(namespaces=namespaces, useRDFLib=True, silent=True)
            kg._bind_namespaces(g)
            g.parse(path, format=format)
            return KnowledgeGraph.ByRDFGraph(g, namespaces=kg._namespaces, silent=silent)
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.ByFile - Error: Could not parse RDF file. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def ByTTL(path: str, **kwargs) -> Optional["KnowledgeGraph"]:
        """Alias for ByFile."""
        kwargs.setdefault("format", "turtle")
        return KnowledgeGraph.ByFile(path, **kwargs)

    @staticmethod
    def ByTopology(
        topology: Any,
        includeOntologyAxioms: bool = False,
        includeDictionaries: bool = True,
        includeBOT: bool = True,
        namespacePrefix: str = "inst",
        useRDFLib: bool = True,
        silent: bool = False,
        **kwargs,
    ) -> Optional["KnowledgeGraph"]:
        """
        Creates a KnowledgeGraph from a TopologicPy topology, legacy Graph, or TGraph.

        The method delegates triple creation to Ontology.py whenever available so
        that dictionary keys and RDF predicates remain canonical.
        """

        Ontology = KnowledgeGraph._ontology_class()
        if topology is None:
            if not silent:
                print("KnowledgeGraph.ByTopology - Error: The input topology is None. Returning None.")
            return None
        if Ontology is None:
            TGraph = KnowledgeGraph._tgraph_class()
            if TGraph is not None and isinstance(topology, TGraph) and hasattr(TGraph, "OntologyTriples"):
                try:
                    if not silent:
                        print("KnowledgeGraph.ByTopology - Warning: Ontology.py is unavailable; using TGraph.OntologyTriples fallback.")
                    triples = TGraph.OntologyTriples(
                        topology,
                        includeDictionaries=includeDictionaries,
                        includeBOT=includeBOT,
                        namespacePrefix=namespacePrefix,
                    )
                    return KnowledgeGraph.ByTriples(triples, namespaces=KnowledgeGraph.Namespaces(), useRDFLib=useRDFLib, silent=silent)
                except Exception as exc:
                    if not silent:
                        print("KnowledgeGraph.ByTopology - Error: Could not create a knowledge graph from TGraph fallback. Returning None.")
                        print("Error:", exc)
                    return None
            if not silent:
                print("KnowledgeGraph.ByTopology - Error: Ontology.py is required for topology conversion. Returning None.")
            return None

        try:
            is_graph_like = False
            try:
                is_graph_like = bool(Ontology._is_graph_like(topology))
            except Exception:
                pass
            if is_graph_like:
                triples = Ontology.GraphTriples(
                    topology,
                    includeDictionaries=includeDictionaries,
                    includeBOT=includeBOT,
                    namespacePrefix=namespacePrefix,
                    silent=silent,
                    **kwargs,
                )
            else:
                triples = Ontology.Triples(
                    topology,
                    includeDictionaries=includeDictionaries,
                    includeBOT=includeBOT,
                    namespacePrefix=namespacePrefix,
                    silent=silent,
                    **kwargs,
                )
            if includeOntologyAxioms:
                try:
                    triples = list(triples) + list(Ontology.OntologyTriples(includeBOT=includeBOT))
                except Exception:
                    pass
            return KnowledgeGraph.ByTriples(triples, namespaces=Ontology.NAMESPACES, useRDFLib=useRDFLib, silent=silent)
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.ByTopology - Error: Could not create a knowledge graph. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def ByTGraph(graph: Any, **kwargs) -> Optional["KnowledgeGraph"]:
        """Alias for ByTopology."""
        return KnowledgeGraph.ByTopology(graph, **kwargs)

    # ---------------------------------------------------------------------
    # Serialization and file IO
    # ---------------------------------------------------------------------

    def RDFGraph(self, rebuild: bool = False, silent: bool = False) -> Any:
        """
        Returns an RDFLib graph, or None if RDFLib is unavailable.
        """
        if rebuild or self._rdf_graph is None:
            return self._sync_rdflib_from_triples(silent=silent)
        return self._rdf_graph

    def TurtleString(self, includeHeader: bool = True, useRDFLib: bool = False, format: str = "turtle", silent: bool = False) -> Optional[str]:
        """
        Returns a Turtle string representation of this KnowledgeGraph.
        """
        if useRDFLib:
            g = self.RDFGraph(silent=silent)
            if g is not None:
                try:
                    data = g.serialize(format=format)
                    return data.decode("utf-8") if isinstance(data, bytes) else str(data)
                except Exception as exc:
                    if not silent:
                        print("KnowledgeGraph.TurtleString - Warning: RDFLib serialisation failed; using fallback Turtle writer.")
                        print("Error:", exc)
        Ontology = KnowledgeGraph._ontology_class()
        if Ontology is not None:
            try:
                return Ontology.TurtleFromTriples(
                    self.Triples(sort=True),
                    namespaces=self._namespaces,
                    instanceNamespace=self._namespaces.get("inst", "http://w3id.org/topologicpy/instance#"),
                    includeHeader=includeHeader,
                )
            except Exception:
                pass
        lines = []
        if includeHeader:
            for prefix, uri in self._namespaces.items():
                lines.append(f"@prefix {prefix}: <{uri}> .")
            lines.append("")
        for s, p, o in self.Triples(sort=True):
            lines.append(f"{s} {p} {o} .")
        return "\n".join(lines) + "\n"

    def RDFString(self, format: str = "turtle", silent: bool = False) -> Optional[str]:
        """Serializes this KnowledgeGraph with RDFLib if available."""
        g = self.RDFGraph(silent=silent)
        if g is None:
            if format in ["ttl", "turtle"]:
                return self.TurtleString(silent=silent)
            if not silent:
                print("KnowledgeGraph.RDFString - Error: RDFLib is required for this format. Returning None.")
            return None
        try:
            fmt = "turtle" if format == "ttl" else format
            data = g.serialize(format=fmt)
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.RDFString - Error: Could not serialize RDF graph. Returning None.")
                print("Error:", exc)
            return None

    def JSONString(self, indent: int = 2) -> str:
        """Returns a JSON representation of this KnowledgeGraph."""
        return json.dumps(self.Dictionary(), indent=indent)

    def Export(self, path: str, format: Optional[str] = None, overwrite: bool = True, silent: bool = False) -> Optional[str]:
        """
        Exports this KnowledgeGraph to a file.
        """
        if path is None:
            if not silent:
                print("KnowledgeGraph.Export - Error: The input path is None. Returning None.")
            return None
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("KnowledgeGraph.Export - Error: File exists and overwrite is False. Returning None.")
            return None
        ext = os.path.splitext(path)[1].lower()
        if format is None:
            format = {
                ".ttl": "turtle",
                ".rdf": "xml",
                ".owl": "xml",
                ".xml": "xml",
                ".nt": "nt",
                ".n3": "n3",
                ".jsonld": "json-ld",
                ".trig": "trig",
                ".json": "json",
            }.get(ext, "turtle")
        try:
            if format == "json" or ext == ".json" and format not in ["json-ld"]:
                data = self.JSONString(indent=2)
            elif format in ["ttl", "turtle"]:
                data = self.TurtleString(silent=silent)
            else:
                data = self.RDFString(format=format, silent=silent)
            if data is None:
                return None
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            return path
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.Export - Error: Could not export file. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def TurtleStringByTriples(triples: Iterable[Tuple[Any, Any, Any]], namespaces: Optional[Dict[str, str]] = None, includeHeader: bool = True) -> str:
        """Convenience function that returns a Turtle string from triples."""
        return KnowledgeGraph.ByTriples(triples, namespaces=namespaces, useRDFLib=False).TurtleString(includeHeader=includeHeader)

    # ---------------------------------------------------------------------
    # SPARQL and RDF querying
    # ---------------------------------------------------------------------

    def Query(self, sparql: str, initBindings: Optional[Dict[str, Any]] = None, silent: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE query using RDFLib.

        For SELECT queries the method returns a list of dictionaries. For ASK it
        returns ``[{"ASK": bool}]``. For CONSTRUCT/DESCRIBE it returns a list of
        compact triples under the key ``"triples"``.
        """
        g = self.RDFGraph(silent=silent)
        if g is None:
            return None
        try:
            bindings = None
            if isinstance(initBindings, dict):
                bindings = {k: self._term_to_rdflib(v, role="object") for k, v in initBindings.items()}
            result = g.query(sparql, initBindings=bindings)
            if getattr(result, "type", None) == "ASK":
                return [{"ASK": bool(result.askAnswer)}]
            if getattr(result, "type", None) in ["CONSTRUCT", "DESCRIBE"]:
                kg = KnowledgeGraph.ByRDFGraph(result.graph, namespaces=self._namespaces, silent=True)
                return [{"triples": kg.Triples(sort=True) if kg is not None else []}]
            rows = []
            variables = [str(v) for v in getattr(result, "vars", [])]
            for row in result:
                record = {}
                for i, var in enumerate(variables):
                    record[var] = self._rdflib_to_token(row[i]) if row[i] is not None else None
                rows.append(record)
            return rows
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.Query - Error: SPARQL query failed. Returning None.")
                print("Error:", exc)
            return None

    def Update(self, sparqlUpdate: str, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """
        Executes a SPARQL UPDATE statement using RDFLib and refreshes triples.
        """
        g = self.RDFGraph(silent=silent)
        if g is None:
            return None
        try:
            g.update(sparqlUpdate)
            self._sync_triples_from_rdflib(silent=silent)
            return self
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.Update - Error: SPARQL update failed. Returning None.")
                print("Error:", exc)
            return None

    # ---------------------------------------------------------------------
    # Merge, diff, and graph algebra
    # ---------------------------------------------------------------------

    def Merge(self, other: Any, inplace: bool = False, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """
        Returns the union of this graph and another KnowledgeGraph/triple iterable.
        """
        target = self if inplace else self.Copy()
        other_kg = KnowledgeGraph._as_kg(other, namespaces=self._namespaces, silent=silent)
        if other_kg is None:
            return None
        target._namespaces.update(other_kg._namespaces)
        target.AddTriples(other_kg.Triples(sort=False), silent=silent)
        return target

    def Difference(self, other: Any, direction: str = "self_minus_other", silent: bool = False) -> Optional[List[Tuple[str, str, str]]]:
        """
        Returns set difference between this graph and another graph.

        Parameters
        ----------
        other : KnowledgeGraph or iterable(tuple)
            The graph to compare against.
        direction : str, optional
            ``"self_minus_other"`` or ``"other_minus_self"``.
        """
        other_kg = KnowledgeGraph._as_kg(other, namespaces=self._namespaces, silent=silent)
        if other_kg is None:
            return None
        if str(direction).lower() in ["other_minus_self", "added", "after_minus_before"]:
            diff = set(other_kg._triples) - set(self._triples)
        else:
            diff = set(self._triples) - set(other_kg._triples)
        return sorted(diff)

    def Diff(self, after: Any, silent: bool = False) -> Optional[Dict[str, Any]]:
        """
        Returns an added/removed/unchanged diff from this graph to ``after``.
        """
        after_kg = KnowledgeGraph._as_kg(after, namespaces=self._namespaces, silent=silent)
        if after_kg is None:
            return None
        before_set = set(self._triples)
        after_set = set(after_kg._triples)
        added = sorted(after_set - before_set)
        removed = sorted(before_set - after_set)
        unchanged = sorted(before_set & after_set)
        return {
            "before_count": len(before_set),
            "after_count": len(after_set),
            "added_count": len(added),
            "removed_count": len(removed),
            "unchanged_count": len(unchanged),
            "added": added,
            "removed": removed,
            "unchanged": unchanged,
        }

    @staticmethod
    def _as_kg(value: Any, namespaces: Optional[Dict[str, str]] = None, silent: bool = False) -> Optional["KnowledgeGraph"]:
        if isinstance(value, KnowledgeGraph):
            return value
        if value is None:
            return None
        if isinstance(value, dict) and value.get("type") == "KnowledgeGraph":
            return KnowledgeGraph.ByDictionary(value, silent=silent)
        if isinstance(value, str):
            # Treat existing paths as files; otherwise try Turtle string.
            if os.path.exists(value):
                return KnowledgeGraph.ByFile(value, namespaces=namespaces, silent=silent)
            return KnowledgeGraph.ByTurtleString(value, namespaces=namespaces, silent=silent)
        try:
            return KnowledgeGraph.ByTriples(value, namespaces=namespaces, silent=silent)
        except Exception:
            if not silent:
                print("KnowledgeGraph - Error: Could not coerce value to KnowledgeGraph. Returning None.")
            return None

    @staticmethod
    def MergeGraphs(graphs: Iterable[Any], namespaces: Optional[Dict[str, str]] = None, silent: bool = False) -> Optional["KnowledgeGraph"]:
        """Returns the union of multiple graphs/triple collections."""
        result = KnowledgeGraph(namespaces=namespaces, silent=True)
        for graph in graphs or []:
            kg = KnowledgeGraph._as_kg(graph, namespaces=result._namespaces, silent=silent)
            if kg is not None:
                result = result.Merge(kg, inplace=True, silent=silent)
        return result

    @staticmethod
    def DiffGraphs(before: Any, after: Any, silent: bool = False) -> Optional[Dict[str, Any]]:
        """Static convenience wrapper for ``before.Diff(after)``."""
        before_kg = KnowledgeGraph._as_kg(before, silent=silent)
        if before_kg is None:
            return None
        return before_kg.Diff(after, silent=silent)

    # ---------------------------------------------------------------------
    # Inference bridge
    # ---------------------------------------------------------------------

    def Infer(
        self,
        profile: str = "rdfs",
        includeOntologyAxioms: bool = True,
        includeBOT: bool = True,
        inplace: bool = False,
        silent: bool = False,
        **kwargs,
    ) -> Optional["KnowledgeGraph"]:
        """
        Runs Reasoner.Infer on this KnowledgeGraph and returns a KnowledgeGraph.

        This is a bridge only. The actual inference implementation belongs in
        Reasoner.py.
        """
        Reasoner = KnowledgeGraph._reasoner_class()
        if Reasoner is None:
            if not silent:
                print("KnowledgeGraph.Infer - Error: Reasoner.py is not available. Returning None.")
            return None
        rdf = self.RDFGraph(silent=silent)
        if rdf is None:
            return None
        try:
            inferred = Reasoner.Infer(
                rdf,
                profile=profile,
                includeOntologyAxioms=includeOntologyAxioms,
                includeBOT=includeBOT,
                silent=silent,
                **kwargs,
            )
            kg = KnowledgeGraph.ByRDFGraph(inferred, namespaces=self._namespaces, silent=silent)
            if inplace and kg is not None:
                self._triples = set(kg._triples)
                self._rdf_graph = kg._rdf_graph
                self._namespaces = dict(kg._namespaces)
                return self
            return kg
        except Exception as exc:
            if not silent:
                print("KnowledgeGraph.Infer - Error: Reasoner inference failed. Returning None.")
                print("Error:", exc)
            return None

    # ---------------------------------------------------------------------
    # Conversion to TGraph
    # ---------------------------------------------------------------------

    def ToTGraph(
        self,
        includeLiterals: bool = True,
        directed: bool = True,
        predicateKey: str = "predicate",
        relationshipKey: str = "relationship",
        uriKey: str = "uri",
        labelKey: str = "label",
        silent: bool = False,
    ) -> Any:
        """
        Converts this KnowledgeGraph to a TGraph semantic view.

        Resources and optionally literals become vertices. Triples become
        directed edges with predicate/relationship metadata.
        """
        TGraph = KnowledgeGraph._tgraph_class()
        if TGraph is None:
            if not silent:
                print("KnowledgeGraph.ToTGraph - Error: TGraph.py is not available. Returning None.")
            return None
        try:
            graph = TGraph(
                directed=directed,
                allowSelfLoops=True,
                allowParallelEdges=True,
                dictionary={
                    "ontology_class": "top:KnowledgeGraph",
                    "category": "graph",
                    "label": "KnowledgeGraph",
                    "generated_by": "KnowledgeGraph.ToTGraph",
                },
            )
        except TypeError:
            graph = TGraph(directed=directed, dictionary={"ontology_class": "top:KnowledgeGraph", "category": "graph"})

        index_by_token: Dict[str, int] = {}

        def label_from_token(token: str) -> str:
            if KnowledgeGraph._is_literal_token(token):
                return KnowledgeGraph._strip_literal_quotes(token)
            if ":" in token and not token.startswith("<"):
                return token.split(":", 1)[1]
            if token.startswith("<") and token.endswith(">"):
                return token[1:-1].rstrip("/#").split("/")[-1].split("#")[-1]
            return token

        def add_node(token: str) -> Optional[int]:
            if token in index_by_token:
                return index_by_token[token]
            is_lit = KnowledgeGraph._is_literal_token(token)
            if is_lit and not includeLiterals:
                return None
            d = {
                uriKey: token,
                labelKey: label_from_token(token),
                "ontology_class": "top:Node",
                "category": "literal" if is_lit else "resource",
            }
            try:
                idx = graph.AddVertex(dictionary=d)
            except Exception:
                idx = TGraph.AddVertex(graph, dictionary=d)
            index_by_token[token] = idx
            return idx

        for s, p, o in self.Triples(sort=True):
            sidx = add_node(s)
            oidx = add_node(o)
            if sidx is None or oidx is None:
                continue
            pred_label = label_from_token(p)
            ed = {
                predicateKey: p,
                relationshipKey: pred_label,
                "ontology_class": "top:Relationship",
                "category": "semantic",
                "label": pred_label,
            }
            try:
                graph.AddEdge(sidx, oidx, directed=True, dictionary=ed)
            except Exception:
                TGraph.AddEdge(graph, sidx, oidx, directed=True, dictionary=ed)
        return graph

    @staticmethod
    def TGraphByKnowledgeGraph(knowledgeGraph: Any, **kwargs) -> Any:
        """Static wrapper for ``ToTGraph``."""
        kg = KnowledgeGraph._as_kg(knowledgeGraph, silent=kwargs.get("silent", False))
        if kg is None:
            return None
        return kg.ToTGraph(**kwargs)

    # ---------------------------------------------------------------------
    # Summaries and validation
    # ---------------------------------------------------------------------

    def Summary(self) -> Dict[str, Any]:
        """Returns a compact summary of this KnowledgeGraph."""
        return {
            "triple_count": len(self._triples),
            "subject_count": len(self.Subjects()),
            "predicate_count": len(self.Predicates()),
            "object_count": len(self.Objects()),
            "resource_count": len(self.Resources(includePredicates=False, includeLiterals=False)),
            "literal_count": len(self.Literals()),
            "namespace_count": len(self._namespaces),
            "rdflib_available": KnowledgeGraph._rdflib(silent=True) is not None,
            "rdflib_enabled": self._rdflib_enabled,
        }

    def Validate(self, parseWithRDFLib: bool = True, silent: bool = False) -> Dict[str, Any]:
        """
        Performs lightweight structural validation and optional RDFLib parsing.
        """
        report = {"valid": True, "errors": [], "warnings": [], "triple_count": len(self._triples)}
        for triple in self._triples:
            if len(triple) != 3:
                report["valid"] = False
                report["errors"].append("Invalid triple length: " + str(triple))
                continue
            s, p, o = triple
            if not KnowledgeGraph._is_resource_token(s, self._namespaces):
                report["valid"] = False
                report["errors"].append("Subject is not a resource token: " + str(s))
            if not KnowledgeGraph._is_resource_token(p, self._namespaces):
                report["valid"] = False
                report["errors"].append("Predicate is not a resource token: " + str(p))
            if not (KnowledgeGraph._is_resource_token(o, self._namespaces) or KnowledgeGraph._is_literal_token(o)):
                report["valid"] = False
                report["errors"].append("Object is neither resource nor literal token: " + str(o))
        if parseWithRDFLib:
            rd = KnowledgeGraph._rdflib(silent=True)
            if rd is None:
                report["warnings"].append("RDFLib is not available; RDF syntax parse validation was skipped.")
            else:
                ttl = self.TurtleString(silent=True)
                try:
                    g = rd["Graph"]()
                    for prefix, uri in self._namespaces.items():
                        g.bind(prefix, uri)
                    g.parse(data=ttl, format="turtle")
                except Exception as exc:
                    report["valid"] = False
                    report["errors"].append("RDFLib Turtle parse failed: " + str(exc))
        if not silent and not report["valid"]:
            print("KnowledgeGraph.Validate - Error: KnowledgeGraph is invalid.")
            for error in report["errors"]:
                print(" -", error)
        return report

    # ---------------------------------------------------------------------
    # Compatibility aliases matching common TopologicPy naming style
    # ---------------------------------------------------------------------

    @staticmethod
    def TriplesByTopology(topology: Any, **kwargs) -> List[Tuple[str, str, str]]:
        """Returns triples for a topology/graph by constructing a KnowledgeGraph."""
        kg = KnowledgeGraph.ByTopology(topology, **kwargs)
        return [] if kg is None else kg.Triples(sort=True)

    @staticmethod
    def RDFGraphByTopology(topology: Any, **kwargs) -> Any:
        """Returns an RDFLib graph for a topology/graph when RDFLib is available."""
        kg = KnowledgeGraph.ByTopology(topology, **kwargs)
        return None if kg is None else kg.RDFGraph(silent=kwargs.get("silent", False))

    @staticmethod
    def ExportRDF(knowledgeGraph: Any, path: str, **kwargs) -> Optional[str]:
        """Static wrapper for Export."""
        kg = KnowledgeGraph._as_kg(knowledgeGraph, silent=kwargs.get("silent", False))
        if kg is None:
            return None
        return kg.Export(path, **kwargs)

    @staticmethod
    def ExportTTL(knowledgeGraph: Any, path: str, **kwargs) -> Optional[str]:
        """Static wrapper for Turtle export."""
        kwargs.setdefault("format", "turtle")
        return KnowledgeGraph.ExportRDF(knowledgeGraph, path, **kwargs)

    @staticmethod
    def Statistics(knowledgeGraph: Any, silent: bool = False) -> Optional[Dict[str, Any]]:
        """Static wrapper for Summary."""
        kg = KnowledgeGraph._as_kg(knowledgeGraph, silent=silent)
        return None if kg is None else kg.Summary()


# Public alias for code that prefers a shorter name.
KG = KnowledgeGraph
