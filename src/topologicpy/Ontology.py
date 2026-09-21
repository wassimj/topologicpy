# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Canonical ontology support for TopologicPy.

Design rules
------------
1. ``ontology/topologicpy.ttl`` is the single human-edited ontology source.
2. TopologicPy-owned datatype properties use concise attribute names such as
   ``top:x``, ``top:area`` and ``top:volume``.
3. Object properties name the relationship: ``top:startsAt``, ``top:endsAt``,
   ``top:connectsTo``. ``has...`` is used only when possession/association is
   actually the relationship (for example ``top:hasNode``).
4. Standard vocabularies are reused when they already express the semantics:
   rdf:type, rdfs:label, dcterms:*, prov:*, BOT, Brick and GeoSPARQL.
5. Unknown Python dictionary keys are exported under ``dict:``; they never
   silently become new ``top:`` terms.
6. Labels are presentation data and NEVER determine RDF identity.
7. RDF -> TGraph -> RDF preserves resource identity, namespaces, multiplicity,
   RDF types, and exact non-structural URI/BNode/literal values (including
   literal datatype and language). Structural graph statements are rebuilt
   canonically from the reconstructed TGraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import re


@dataclass(frozen=True)
class _RDFLiteral:
    lexical: str
    datatype: Optional[str] = None
    language: Optional[str] = None

class Ontology:
    # Canonical dictionary keys used internally by TopologicPy.
    ONTOLOGY_CLASS_KEY = "ontology_class"
    ONTOLOGY_URI_KEY = "ontology_uri"
    LABEL_KEY = "label"
    CATEGORY_KEY = "category"
    IFC_CLASS_KEY = "ifc_class"
    IFC_GUID_KEY = "ifc_guid"
    SOURCE_KEY = "source"
    DERIVED_FROM_KEY = "derived_from"
    GENERATED_BY_KEY = "generated_by"
    URI_KEY = "uri"

    # Lossless RDF payload keys. These are implementation metadata and are
    # deliberately never emitted as top: or dict: predicates.
    RDF_TYPES_KEY = "_rdf_types"
    RDF_PROPERTIES_KEY = "_rdf_properties"
    RDF_URI_KEY = "_rdf_uri"

    # Importer mirrors that are either emitted canonically elsewhere or are
    # processing/rendering configuration rather than graph semantics.
    _NON_RDF_DICTIONARY_KEYS = {
        "IFC_type", "IFC_global_id", "IFC_name",
        "dictionary_mode", "dictionaryMode",
        "import_mode", "importMode",
        "color",
    }

    NAMESPACES = {
        "top": "http://w3id.org/topologicpy#",
        "inst": "http://w3id.org/topologicpy/instance#",
        "dict": "http://w3id.org/topologicpy/dictionary#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "dcterms": "http://purl.org/dc/terms/",
        "prov": "http://www.w3.org/ns/prov#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "vann": "http://purl.org/vocab/vann/",
        "bot": "https://w3id.org/bot#",
        "brick": "https://brickschema.org/schema/Brick#",
        "geo": "http://www.opengis.net/ont/geosparql#",
        "ifc": "https://standards.buildingsmart.org/IFC/DEV/IFC4/ADD2_TC1/OWL#",
    }

    IFC_TO_TOP = {
        "IfcProject": "top:Project", "IfcSite": "top:Site",
        "IfcBuilding": "top:Building", "IfcBuildingStorey": "top:Storey",
        "IfcSpace": "top:Space", "IfcZone": "top:Zone",
        "IfcWall": "top:Wall", "IfcWallStandardCase": "top:Wall",
        "IfcCurtainWall": "top:CurtainWall", "IfcDoor": "top:Door",
        "IfcWindow": "top:Window", "IfcSlab": "top:Slab",
        "IfcRoof": "top:Roof", "IfcColumn": "top:Column",
        "IfcBeam": "top:Beam", "IfcMember": "top:Member",
        "IfcStair": "top:Stair", "IfcStairFlight": "top:Stair",
        "IfcRailing": "top:Railing", "IfcOpeningElement": "top:Opening",
        "IfcVirtualElement": "top:Element", "IfcFurnishingElement": "top:Furniture",
        "IfcFurniture": "top:Furniture", "IfcFlowTerminal": "top:Equipment",
        "IfcDistributionElement": "top:Equipment",
        "IfcDistributionFlowElement": "top:Equipment",
        "IfcEnergyConversionDevice": "top:Equipment",
        "IfcFlowController": "top:Equipment", "IfcFlowFitting": "top:Equipment",
        "IfcFlowMovingDevice": "top:Equipment", "IfcFlowSegment": "top:Equipment",
        "IfcFlowStorageDevice": "top:Equipment", "IfcFlowTreatmentDevice": "top:Equipment",
        "IfcSensor": "top:Sensor", "IfcBuildingElementProxy": "top:Element",
        "IfcRelSpaceBoundary": "top:Interface", "IfcMaterial": "top:Material",
        "IfcMaterialLayerSet": "top:MaterialSet", "IfcMaterialProfileSet": "top:MaterialSet",
        "IfcPropertySet": "top:PropertySet", "IfcElementQuantity": "top:Quantity",
        "IfcClassificationReference": "top:ClassificationReference",
        "IfcApproval": "top:Approval", "IfcConstraint": "top:Constraint",
        "IfcDocumentReference": "top:DocumentReference",
    }

    TOP_TO_BOT = {
        "top:Building": "bot:Building", "top:Element": "bot:Element",
        "top:Equipment": "brick:Equipment", "top:Interface": "bot:Interface",
        "top:Project": "prov:Entity", "top:Sensor": "brick:Point",
        "top:Site": "bot:Site", "top:Space": "bot:Space",
        "top:Storey": "bot:Storey", "top:Zone": "bot:Zone",
        "top:Room": "bot:Space", "top:ThermalZone": "bot:Zone",
        "top:FunctionalZone": "bot:Zone", "top:CirculationZone": "bot:Zone",
        "top:Wall": "bot:Element", "top:CurtainWall": "bot:Element",
        "top:Door": "bot:Element", "top:Window": "bot:Element",
        "top:Slab": "bot:Element", "top:Roof": "bot:Element",
        "top:Column": "bot:Element", "top:Beam": "bot:Element",
        "top:Member": "bot:Element", "top:Stair": "bot:Element",
        "top:Railing": "bot:Element", "top:Opening": "bot:Element",
        "top:Aperture": "bot:Element", "top:Furniture": "bot:Element",
    }

    # Deliberately empty. The ontology is clean-slate: no deprecated aliases.
    PROPERTY_ALIASES: Dict[str, str] = {}
    CLASS_ALIASES: Dict[str, str] = {}
    DEPRECATED_PROPERTIES = set()

    # Public vocabulary tables are populated once from the canonical TTL at
    # module import. They remain available for TGraph/KnowledgeGraph/Reasoner
    # code that needs fast membership tests without maintaining a second schema.
    TOP_SUPERCLASSES: Dict[str, List[str]] = {}
    OBJECT_PROPERTIES: Dict[str, Any] = {}
    DATA_PROPERTIES: Dict[str, Any] = {}
    ANNOTATION_PROPERTIES: Dict[str, Any] = {}

    _VOCAB_CACHE = None

    # ------------------------------------------------------------------
    # RDFLib / ontology resource
    # ------------------------------------------------------------------
    @staticmethod
    def _rdflib(silent: bool = False):
        try:
            import rdflib
            return rdflib
        except Exception:
            if not silent:
                print("Ontology - Error: RDFLib is required for ontology/RDF operations. Returning None.")
            return None

    @staticmethod
    def _ontology_resource_path() -> Optional[Path]:
        """Returns the packaged canonical TTL path when available.

        The implementation supports Python 3.8+; ``importlib.resources.files``
        is used when available and the older ``resources.path`` API is used as
        a fallback. A development checkout is also detected.
        """
        try:
            files_fn = getattr(resources, "files", None)
            if callable(files_fn):
                resource = files_fn("topologicpy").joinpath("ontology/topologicpy.ttl")
                try:
                    if resource.is_file():
                        return Path(str(resource))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            with resources.path("topologicpy", "ontology") as ontology_dir:
                candidate = Path(ontology_dir) / "topologicpy.ttl"
                if candidate.exists():
                    return candidate
        except Exception:
            pass
        here = Path(__file__).resolve()
        candidates = [
            here.parent / "ontology" / "topologicpy.ttl",
            here.parents[2] / "ontology" / "topologicpy.ttl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def OntologyTTLString(includeBOT: bool = True, silent: bool = False) -> Optional[str]:
        path = Ontology._ontology_resource_path()
        if path is not None:
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
        # Python 3.8-compatible package-data fallback.
        try:
            import pkgutil
            data = pkgutil.get_data("topologicpy", "ontology/topologicpy.ttl")
            if data is not None:
                return data.decode("utf-8")
        except Exception:
            pass
        if not silent:
            print("Ontology.OntologyTTLString - Error: Canonical ontology file was not found. Returning None.")
        return None

    @staticmethod
    def OntologyRDFGraph(silent: bool = False):
        rd = Ontology._rdflib(silent=silent)
        ttl = Ontology.OntologyTTLString(silent=silent)
        if rd is None or ttl is None:
            return None
        g = rd.Graph()
        try:
            g.parse(data=ttl, format="turtle")
            return g
        except Exception as exc:
            if not silent:
                print("Ontology.OntologyRDFGraph - Error: Could not parse ontology. Returning None.")
                print("Error:", exc)
            return None

    @staticmethod
    def _vocabulary() -> Dict[str, Any]:
        """Returns the vocabulary declared by the canonical TTL.

        RDFLib is preferred, but the fallback parser reads declarations directly
        from the same TTL file. This keeps ``ontology/topologicpy.ttl`` as the
        sole vocabulary authority while allowing lightweight helpers to work
        when RDFLib is unavailable.
        """
        if Ontology._VOCAB_CACHE is not None:
            return Ontology._VOCAB_CACHE

        result = {"classes": set(), "object": set(), "data": set(), "annotation": set(), "property": set(), "super": {}}
        rd = Ontology._rdflib(silent=True)
        g = Ontology.OntologyRDFGraph(silent=True) if rd is not None else None
        if rd is not None and g is not None:
            RDF, RDFS, OWL = rd.RDF, rd.RDFS, rd.OWL
            for kind, bucket in ((OWL.Class, "classes"), (OWL.ObjectProperty, "object"),
                                 (OWL.DatatypeProperty, "data"), (OWL.AnnotationProperty, "annotation"),
                                 (RDF.Property, "property")):
                for subject in g.subjects(RDF.type, kind):
                    result[bucket].add(Ontology.QName(str(subject), defaultValue=str(subject)))
            # Every OWL object/data/annotation property is also a property for
            # TopologicPy membership/canonicalisation purposes, even when the
            # source TTL does not repeat an explicit rdf:Property assertion.
            result["property"].update(result["object"] | result["data"] | result["annotation"])
            for subject, parent in g.subject_objects(RDFS.subClassOf):
                sq = Ontology.QName(str(subject), defaultValue=str(subject))
                pq = Ontology.QName(str(parent), defaultValue=str(parent))
                result["super"].setdefault(sq, []).append(pq)
        else:
            ttl = Ontology.OntologyTTLString(silent=True) or ""
            # Parse complete top:* subject statements. This intentionally parses
            # declarations only; it is not a general Turtle parser.
            statements = []
            current = []
            active = False
            for raw in ttl.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if re.match(r"^top:[A-Za-z_][A-Za-z0-9_.-]*\b", line):
                    if current:
                        statements.append(" ".join(current))
                    current = [line]
                    active = not line.endswith(".")
                    if not active:
                        statements.append(" ".join(current)); current = []
                    continue
                if current:
                    current.append(line)
                    if line.endswith("."):
                        statements.append(" ".join(current)); current = []; active = False
            if current:
                statements.append(" ".join(current))

            for statement in statements:
                match = re.match(r"^(top:[A-Za-z_][A-Za-z0-9_.-]*)\s+", statement)
                if not match:
                    continue
                subject = match.group(1)
                if re.search(r"(?:\ba\b|rdf:type)\s+[^.]*\bowl:Class\b", statement):
                    result["classes"].add(subject)
                if re.search(r"(?:\ba\b|rdf:type)\s+[^.]*\bowl:ObjectProperty\b", statement):
                    result["object"].add(subject)
                if re.search(r"(?:\ba\b|rdf:type)\s+[^.]*\bowl:DatatypeProperty\b", statement):
                    result["data"].add(subject)
                if re.search(r"(?:\ba\b|rdf:type)\s+[^.]*\bowl:AnnotationProperty\b", statement):
                    result["annotation"].add(subject)
                if re.search(r"(?:\ba\b|rdf:type)\s+[^.]*\brdf:Property\b", statement):
                    result["property"].add(subject)
                sm = re.search(r"rdfs:subClassOf\s+([^;]+)", statement)
                if sm:
                    parents = re.findall(r"(?:top|bot|brick|geo|prov|ifc):[A-Za-z_][A-Za-z0-9_.-]*", sm.group(1))
                    if parents:
                        result["super"].setdefault(subject, []).extend(parents)

        result["property"].update(result["object"] | result["data"] | result["annotation"])
        Ontology._VOCAB_CACHE = result
        Ontology.TOP_SUPERCLASSES = {k: list(dict.fromkeys(v)) for k, v in result["super"].items()}
        Ontology.OBJECT_PROPERTIES = {q: None for q in result["object"]}
        Ontology.DATA_PROPERTIES = {q: None for q in result["data"]}
        Ontology.ANNOTATION_PROPERTIES = {q: None for q in result["annotation"]}
        return result

    @staticmethod
    def _is_graph_like(obj: Any) -> bool:
        if obj is None:
            return False
        try:
            from topologicpy.TGraph import TGraph
            if isinstance(obj, TGraph):
                return True
        except Exception:
            pass
        name = obj.__class__.__name__
        return name in {"Graph", "TGraph"}

    @staticmethod
    def Namespaces() -> Dict[str, str]:
        return dict(Ontology.NAMESPACES)

    @staticmethod
    def Namespace(prefix: str, defaultValue: Any = None):
        return Ontology.NAMESPACES.get(str(prefix), defaultValue)

    @staticmethod
    def ExpandQName(value: Any, defaultValue: Any = None):
        if value is None:
            return defaultValue
        text = str(value).strip()
        if text.startswith("<") and text.endswith(">"):
            return text[1:-1]
        if text.startswith(("http://", "https://", "urn:")):
            return text
        if ":" in text:
            prefix, local = text.split(":", 1)
            ns = Ontology.NAMESPACES.get(prefix)
            if ns and local:
                return ns + local
        return defaultValue

    @staticmethod
    def QName(value: Any, defaultValue: Any = None):
        if value is None:
            return defaultValue
        text = str(value).strip().strip("<>")
        if ":" in text and not text.startswith(("http://", "https://", "urn:")):
            prefix = text.split(":", 1)[0]
            if prefix in Ontology.NAMESPACES:
                return text
        for prefix, ns in sorted(Ontology.NAMESPACES.items(), key=lambda kv: len(kv[1]), reverse=True):
            if text.startswith(ns):
                return prefix + ":" + text[len(ns):]
        return defaultValue if defaultValue is not None else text

    @staticmethod
    def IsQName(value: Any) -> bool:
        """Returns True when ``value`` is a QName with a registered prefix."""
        if not isinstance(value, str):
            return False
        text = value.strip()
        if ":" not in text or text.startswith(("http://", "https://", "urn:")):
            return False
        prefix, local = text.split(":", 1)
        return bool(local) and prefix in Ontology.NAMESPACES

    @staticmethod
    def IsClass(value: Any) -> bool:
        q = Ontology.QName(value, defaultValue=value)
        return isinstance(q, str) and q in Ontology._vocabulary()["classes"]

    @staticmethod
    def IsProperty(value: Any) -> bool:
        q = Ontology.QName(value, defaultValue=value)
        vocab = Ontology._vocabulary()
        return isinstance(q, str) and q in (vocab["object"] | vocab["data"] | vocab["annotation"] | vocab.get("property", set()))

    @staticmethod
    def CanonicalClass(value: Any, defaultValue: Any = None):
        """Returns a canonical class QName without inventing ``top:`` terms.

        Undeclared terms in the TopologicPy namespace are rejected.  External
        classes retain their native QName/URI because TopologicPy does not own
        their naming conventions.
        """
        if value is None:
            return defaultValue
        text = str(value).strip()
        if not text:
            return defaultValue
        q = Ontology.QName(text, defaultValue=text)
        known = Ontology._vocabulary()["classes"]
        if isinstance(q, str) and q.startswith("top:"):
            return q if q in known else defaultValue
        if isinstance(q, str) and ":" not in q:
            candidate = "top:" + q
            return candidate if candidate in known else defaultValue
        return q

    @staticmethod
    def ClassQName(value: Any, defaultValue: Any = None):
        return Ontology.CanonicalClass(value, defaultValue=defaultValue)

    @staticmethod
    def PropertyQName(key: Any, defaultPrefix: str = "dict"):
        """Returns a canonical predicate QName without inventing ``top:`` terms.

        An undeclared explicit ``top:`` predicate is invalid and returns None.
        Unknown unqualified Python dictionary keys are intentionally placed in
        the ``dict:`` namespace.  External vocabulary predicates are preserved.
        """
        if key is None:
            return None
        text = str(key).strip()
        if not text:
            return None
        # Python dictionary/control keys are adapters into canonical RDF vocabularies.
        # They are not themselves TopologicPy ontology predicates.
        adapter = {
            "label": "rdfs:label",
            "description": "dcterms:description",
            "source": "dcterms:source",
            "derived_from": "prov:wasDerivedFrom",
            "generated_by": "top:generatedByMethod",
            "ifc_class": "top:ifcClass",
            "ifc_guid": "top:ifcGUID",
        }
        if text in adapter:
            return adapter[text]
        q = Ontology.QName(text, defaultValue=text)
        vocab = Ontology._vocabulary()
        known = vocab["object"] | vocab["data"] | vocab["annotation"] | vocab.get("property", set())
        if isinstance(q, str) and q.startswith("top:"):
            return q if q in known else None
        if isinstance(q, str) and ":" in q:
            return q
        candidate = "top:" + q
        if candidate in known:
            return candidate
        return f"{defaultPrefix}:{Ontology._safe_local_name(q)}"

    @staticmethod
    def ClassByIFCClass(ifcClass: Any, defaultValue: Any = None):
        return Ontology.IFC_TO_TOP.get(str(ifcClass), defaultValue)

    @staticmethod
    def BOTClassByClass(ontologyClass: Any, defaultValue: Any = None):
        return Ontology.TOP_TO_BOT.get(Ontology.CanonicalClass(ontologyClass), defaultValue)

    @staticmethod
    def CategoryByClass(ontologyClass: Any, defaultValue: Any = None):
        q = Ontology.CanonicalClass(ontologyClass)
        if not isinstance(q, str):
            return defaultValue
        local = q.split(":", 1)[-1]
        if local.endswith("Graph") or local in {"Graph", "Path", "Relationship", "Node"}:
            return "graph"
        if local in {"Room", "Space", "Zone", "ThermalZone", "FunctionalZone", "CirculationZone"}:
            return "space"
        if local in {"Building", "Site", "Storey", "Project"}:
            return local.lower()
        if local in {"Wall", "Door", "Window", "Slab", "Roof", "Column", "Beam", "Member", "Stair", "Railing", "Opening", "Furniture", "Equipment", "Sensor", "System", "Port"}:
            return "element"
        if local in {"Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster", "Topology", "Surface", "Boundary", "Aperture", "Point"}:
            return "topology"
        return defaultValue

    # ------------------------------------------------------------------
    # Dictionary adapters
    # ------------------------------------------------------------------
    @staticmethod
    def _dictionary(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            d = obj.get("dictionary")
            return d if isinstance(d, dict) else obj
        try:
            from topologicpy.TGraph import TGraph
            if isinstance(obj, TGraph):
                return obj._dictionary
        except Exception:
            pass
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(obj)
            return dict(Dictionary.PythonDictionary(d) or {})
        except Exception:
            return {}

    @staticmethod
    def _set_dictionary(obj: Any, d: Dict[str, Any]):
        if obj is None:
            return None
        if isinstance(obj, dict):
            if "dictionary" in obj and isinstance(obj.get("dictionary"), dict):
                obj["dictionary"] = dict(d)
            else:
                obj.clear(); obj.update(d)
            return obj
        try:
            from topologicpy.TGraph import TGraph
            if isinstance(obj, TGraph):
                obj._dictionary = dict(d)
                return obj
        except Exception:
            pass
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            return Topology.SetDictionary(obj, Dictionary.ByPythonDictionary(dict(d)))
        except Exception:
            return obj

    @staticmethod
    def Class(obj: Any, defaultValue: Any = None):
        return Ontology.CanonicalClass(Ontology._dictionary(obj).get(Ontology.ONTOLOGY_CLASS_KEY), defaultValue)

    @staticmethod
    def Label(obj: Any, defaultValue: Any = None):
        return Ontology._dictionary(obj).get(Ontology.LABEL_KEY, defaultValue)

    @staticmethod
    def Category(obj: Any, defaultValue: Any = None):
        return Ontology._dictionary(obj).get(Ontology.CATEGORY_KEY, defaultValue)

    @staticmethod
    def SetClass(obj: Any, ontologyClass: str, silent: bool = False):
        q = Ontology.CanonicalClass(ontologyClass, defaultValue=None)
        if not q:
            if not silent:
                print("Ontology.SetClass - Error: Invalid or undeclared ontology class. Returning None.")
            return None
        d = dict(Ontology._dictionary(obj)); d[Ontology.ONTOLOGY_CLASS_KEY] = q
        d.setdefault(Ontology.CATEGORY_KEY, Ontology.CategoryByClass(q))
        return Ontology._set_dictionary(obj, d)

    @staticmethod
    def SetLabel(obj: Any, label: Any, silent: bool = False):
        d = dict(Ontology._dictionary(obj)); d[Ontology.LABEL_KEY] = label
        return Ontology._set_dictionary(obj, d)

    @staticmethod
    def SetCategory(obj: Any, category: Any, silent: bool = False):
        d = dict(Ontology._dictionary(obj)); d[Ontology.CATEGORY_KEY] = category
        return Ontology._set_dictionary(obj, d)

    @staticmethod
    def Value(obj: Any, key: str, defaultValue: Any = None):
        """Returns a dictionary value from a topology, TGraph item, or plain dict."""
        if not isinstance(key, str) or not key:
            return defaultValue
        return Ontology._dictionary(obj).get(key, defaultValue)

    @staticmethod
    def SetValue(obj: Any, key: str, value: Any, silent: bool = False):
        """Sets one dictionary value and preserves the object returned by the adapter."""
        if obj is None or not isinstance(key, str) or not key.strip():
            if not silent:
                print("Ontology.SetValue - Error: Invalid input. Returning None.")
            return None
        d = dict(Ontology._dictionary(obj))
        d[key.strip()] = value
        return Ontology._set_dictionary(obj, d)

    @staticmethod
    def SetURI(obj: Any, uri: str, silent: bool = False):
        """Sets the explicit RDF resource identity under the canonical ``uri`` key."""
        if obj is None or not isinstance(uri, str) or not uri.strip():
            if not silent:
                print("Ontology.SetURI - Error: Invalid URI. Returning None.")
            return None
        d = dict(Ontology._dictionary(obj))
        d[Ontology.URI_KEY] = uri.strip()
        d.pop(Ontology.ONTOLOGY_URI_KEY, None)
        return Ontology._set_dictionary(obj, d)

    @staticmethod
    def ClassByTopology(topology: Any, defaultValue: Any = None):
        if topology is None:
            return defaultValue
        name = topology.__class__.__name__
        try:
            from topologicpy.Topology import Topology
            for candidate in ("Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster"):
                try:
                    if Topology.IsInstance(topology, candidate):
                        return "top:" + candidate
                except Exception:
                    pass
        except Exception:
            pass
        candidate = "top:" + name
        return candidate if candidate in Ontology._vocabulary()["classes"] else defaultValue

    @staticmethod
    def Annotate(topology: Any, ontologyClass: str = None, category: Any = None,
                 label: Any = None, uri: str = None, source: Any = None,
                 derivedFrom: Any = None, generatedBy: Any = None,
                 inferClass: bool = False, silent: bool = False):
        if topology is None:
            return None
        d = dict(Ontology._dictionary(topology))
        cls = ontologyClass or (Ontology.ClassByTopology(topology) if inferClass else None)
        if cls is not None:
            canonical = Ontology.CanonicalClass(cls, defaultValue=None)
            if canonical is None:
                if not silent:
                    print("Ontology.Annotate - Error: Invalid or undeclared TopologicPy ontology class. Returning input unchanged.")
                return topology
            d[Ontology.ONTOLOGY_CLASS_KEY] = canonical
            cls = canonical
        if category is not None: d[Ontology.CATEGORY_KEY] = category
        elif cls is not None: d.setdefault(Ontology.CATEGORY_KEY, Ontology.CategoryByClass(cls))
        if label is not None: d[Ontology.LABEL_KEY] = label
        if uri is not None: d[Ontology.URI_KEY] = uri
        if source is not None: d[Ontology.SOURCE_KEY] = source
        if derivedFrom is not None: d[Ontology.DERIVED_FROM_KEY] = derivedFrom
        if generatedBy is not None: d[Ontology.GENERATED_BY_KEY] = generatedBy
        return Ontology._set_dictionary(topology, d)

    @staticmethod
    def AnnotateIFC(topology: Any, ifcClass: str = None, ifcGUID: str = None,
                    ifcName: str = None, source: Any = None, silent: bool = False):
        d = dict(Ontology._dictionary(topology))
        if ifcClass is not None:
            d[Ontology.IFC_CLASS_KEY] = ifcClass
            d.setdefault(Ontology.ONTOLOGY_CLASS_KEY, Ontology.ClassByIFCClass(ifcClass, "top:Element"))
        if ifcGUID is not None: d[Ontology.IFC_GUID_KEY] = ifcGUID
        if ifcName is not None: d[Ontology.LABEL_KEY] = ifcName
        if source is not None: d[Ontology.SOURCE_KEY] = source
        return Ontology._set_dictionary(topology, d)

    @staticmethod
    def NormalizeDictionary(topology: Any, inferClass: bool = True, silent: bool = False, **kwargs):
        """Normalizes a TopologicPy/IFC dictionary without legacy RDF aliases.

        IFC implementation keys are adapted to the canonical internal keys; no
        deprecated ontology predicate or class aliases are created.
        """
        if topology is None:
            return None
        d = dict(Ontology._dictionary(topology))

        # Canonical resource identity. ``ontology_uri`` is not part of the new
        # data model and is deliberately discarded.
        if Ontology.ONTOLOGY_URI_KEY in d:
            d.pop(Ontology.ONTOLOGY_URI_KEY, None)

        # IFC adapter keys used throughout TopologicPy/imported IFC dictionaries.
        if Ontology.IFC_CLASS_KEY not in d:
            for key in ("IFC_type", "ifc_type", "ifcClass", "type"):
                value = d.get(key)
                if isinstance(value, str) and value.startswith("Ifc"):
                    d[Ontology.IFC_CLASS_KEY] = value
                    break
        if Ontology.IFC_GUID_KEY not in d:
            for key in ("IFC_global_id", "GlobalId", "ifcGUID", "global_id"):
                if d.get(key) not in (None, ""):
                    d[Ontology.IFC_GUID_KEY] = d[key]
                    break
        if Ontology.LABEL_KEY not in d:
            for key in ("IFC_name", "ifc_name", "name", "Name"):
                if d.get(key) not in (None, ""):
                    d[Ontology.LABEL_KEY] = d[key]
                    break

        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        if cls is not None:
            canonical = Ontology.CanonicalClass(cls, defaultValue=None)
            if canonical is None:
                d.pop(Ontology.ONTOLOGY_CLASS_KEY, None)
            else:
                d[Ontology.ONTOLOGY_CLASS_KEY] = canonical
        elif inferClass:
            ifc_class = d.get(Ontology.IFC_CLASS_KEY)
            canonical = Ontology.ClassByIFCClass(ifc_class, defaultValue=None) if ifc_class else Ontology.ClassByTopology(topology, defaultValue=None)
            if canonical is not None:
                d[Ontology.ONTOLOGY_CLASS_KEY] = canonical

        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        if cls is not None and not d.get(Ontology.CATEGORY_KEY):
            category = Ontology.CategoryByClass(cls, defaultValue=None)
            if category is not None:
                d[Ontology.CATEGORY_KEY] = category
        return Ontology._set_dictionary(topology, d)

    @staticmethod
    def IsA(topology: Any, ontologyClass: str, transitive: bool = True) -> bool:
        actual = Ontology.Class(topology)
        target = Ontology.CanonicalClass(ontologyClass)
        if actual == target: return True
        if not transitive or not actual or not target: return False
        super_map = Ontology._vocabulary()["super"]
        stack, seen = [actual], set()
        while stack:
            c = stack.pop()
            if c in seen: continue
            seen.add(c)
            for sup in super_map.get(c, []):
                if sup == target: return True
                stack.append(sup)
        return False

    # ------------------------------------------------------------------
    # RDF term helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_local_name(value: Any) -> str:
        s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value or "unnamed").strip()).strip("_") or "unnamed"
        return "id_" + s if s[0].isdigit() else s

    @staticmethod
    def IsResourceString(value: Any) -> bool:
        if not isinstance(value, str): return False
        s = value.strip()
        if s.startswith(("http://", "https://", "urn:", "_:", "<")): return True
        return ":" in s and s.split(":", 1)[0] in Ontology.NAMESPACES

    @staticmethod
    def _identity(obj: Any, role: str = "item", fallbackIndex: Any = None, prefix: str = "inst") -> str:
        d = Ontology._dictionary(obj)
        # Explicit resource identity first.
        for key in (Ontology.RDF_URI_KEY, Ontology.URI_KEY, "uri", "URI"):
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                return value if Ontology.IsResourceString(value) else f"{prefix}:{Ontology._safe_local_name(value)}"
        # Stable identifiers. LABEL IS INTENTIONALLY ABSENT.
        for key in ("uuid", "UUID", "id", "ID", Ontology.IFC_GUID_KEY,
                    "ifc_guid", "ifcGUID", "GlobalId", "IFC_global_id"):
            value = d.get(key)
            if value not in (None, ""):
                return f"{prefix}:{Ontology._safe_local_name(value)}"
        if isinstance(obj, dict):
            value = obj.get("index")
            if value is not None:
                return f"{prefix}:{role}_{Ontology._safe_local_name(value)}"
        if fallbackIndex is not None:
            return f"{prefix}:{role}_{Ontology._safe_local_name(fallbackIndex)}"
        try:
            from topologicpy.Topology import Topology
            uid = Topology.UUID(obj, silent=True)
            if uid: return f"{prefix}:{role}_{Ontology._safe_local_name(uid)}"
        except Exception:
            pass
        return f"{prefix}:{role}_{Ontology._safe_local_name(id(obj))}"

    @staticmethod
    def _literal(value: Any, datatype: str = None, language: str = None) -> _RDFLiteral:
        if datatype is None and language is None:
            if isinstance(value, bool):
                datatype = Ontology.NAMESPACES["xsd"] + "boolean"
                value = "true" if value else "false"
            elif isinstance(value, int) and not isinstance(value, bool):
                datatype = Ontology.NAMESPACES["xsd"] + "integer"
            elif isinstance(value, float):
                datatype = Ontology.NAMESPACES["xsd"] + "double"
        return _RDFLiteral(str(value), datatype, language)

    @staticmethod
    def _encoded_rdf_object(obj: Any) -> Dict[str, Any]:
        rd = Ontology._rdflib(silent=True)
        if rd is not None:
            if isinstance(obj, rd.URIRef): return {"kind": "uri", "value": str(obj)}
            if isinstance(obj, rd.BNode): return {"kind": "bnode", "value": str(obj)}
            if isinstance(obj, rd.Literal):
                return {"kind": "literal", "value": str(obj),
                        "datatype": str(obj.datatype) if obj.datatype else None,
                        "language": obj.language}
        return {"kind": "literal", "value": str(obj), "datatype": None, "language": None}

    @staticmethod
    def _decoded_rdf_object(data: Dict[str, Any]):
        kind = data.get("kind")
        if kind == "uri": return data.get("value")
        if kind == "bnode": return "_:" + str(data.get("value"))
        return _RDFLiteral(str(data.get("value", "")), data.get("datatype"), data.get("language"))

    @staticmethod
    def _predicate_for_internal_key(key: str) -> Optional[str]:
        # Internal metadata maps to established vocabularies where appropriate.
        mapping = {
            "label": "rdfs:label",
            "source": "dcterms:source",
            "derived_from": "prov:wasDerivedFrom",
            "ifc_class": "top:ifcClass",
            "ifc_guid": "top:ifcGUID",
        }
        if key in mapping: return mapping[key]
        if key == "generated_by": return "top:generatedByMethod"
        if key in {Ontology.ONTOLOGY_CLASS_KEY, Ontology.ONTOLOGY_URI_KEY, Ontology.URI_KEY,
                   Ontology.RDF_TYPES_KEY, Ontology.RDF_PROPERTIES_KEY, Ontology.RDF_URI_KEY,
                   "active", "representation", "src", "dst",
                   "ontology_predicate", "inverse_predicate", "predicate",
                   "ontologyPredicate", "inversePredicate"} | Ontology._NON_RDF_DICTIONARY_KEYS:
            return None
        return Ontology.PropertyQName(key, defaultPrefix="dict")

    @staticmethod
    def _value_terms(value: Any) -> List[Any]:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result = []
        for v in values:
            if isinstance(v, _RDFLiteral): result.append(v)
            else: result.append(Ontology._literal(v))
        return result

    @staticmethod
    def Triples(topology: Any, subject: str = None, includeDictionaries: bool = True,
                includeBOT: bool = True, namespacePrefix: str = "inst", silent: bool = False):
        if topology is None: return []
        d = dict(Ontology._dictionary(topology))

        # Adapt common IFC importer mirrors locally so canonical RDF remains
        # complete even when callers have not explicitly normalized the source
        # dictionary. The source topology is not mutated.
        if d.get(Ontology.IFC_CLASS_KEY) in (None, "") and d.get("IFC_type") not in (None, ""):
            d[Ontology.IFC_CLASS_KEY] = d["IFC_type"]
        if d.get(Ontology.IFC_GUID_KEY) in (None, "") and d.get("IFC_global_id") not in (None, ""):
            d[Ontology.IFC_GUID_KEY] = d["IFC_global_id"]
        if d.get(Ontology.LABEL_KEY) in (None, "") and d.get("IFC_name") not in (None, ""):
            d[Ontology.LABEL_KEY] = d["IFC_name"]
        if d.get(Ontology.ONTOLOGY_CLASS_KEY) in (None, ""):
            inferred_class = Ontology.ClassByIFCClass(d.get(Ontology.IFC_CLASS_KEY), defaultValue=None)
            if inferred_class is not None:
                d[Ontology.ONTOLOGY_CLASS_KEY] = inferred_class

        subject = subject or Ontology._identity(topology, prefix=namespacePrefix)
        triples: List[Tuple[Any, Any, Any]] = []

        # Preserve every imported RDF type, then assert canonical TopologicPy type.
        for t in d.get(Ontology.RDF_TYPES_KEY, []) or []:
            triples.append((subject, "rdf:type", Ontology.QName(t, defaultValue=t)))
        cls = Ontology.CanonicalClass(d.get(Ontology.ONTOLOGY_CLASS_KEY), defaultValue=None)
        if cls:
            triples.append((subject, "rdf:type", cls))
            if includeBOT:
                ext = Ontology.BOTClassByClass(cls)
                if ext: triples.append((subject, "rdf:type", ext))

        # Re-emit exact imported non-structural properties.  Keep a set of
        # their expanded predicate URIs so convenience dictionary mirrors do
        # not emit a second, weaker copy (for example, dropping @lang or an
        # explicit datatype from rdfs:label).
        preserved_predicates = set()
        for item in d.get(Ontology.RDF_PROPERTIES_KEY, []) or []:
            raw_predicate = item.get("predicate")
            p = Ontology.QName(raw_predicate, defaultValue=raw_predicate)
            expanded = Ontology.ExpandQName(p, defaultValue=str(raw_predicate) if raw_predicate else None)
            if expanded:
                preserved_predicates.add(str(expanded))
            o = Ontology._decoded_rdf_object(item.get("object", {}))
            triples.append((subject, p, o))

        if includeDictionaries:
            for key, value in d.items():
                if key in {Ontology.RDF_TYPES_KEY, Ontology.RDF_PROPERTIES_KEY, Ontology.RDF_URI_KEY}:
                    continue
                pred = Ontology._predicate_for_internal_key(str(key))
                if not pred:
                    continue
                expanded_predicate = Ontology.ExpandQName(pred, defaultValue=pred)
                if expanded_predicate and str(expanded_predicate) in preserved_predicates:
                    continue
                # Resource-valued standard properties retain resource semantics.
                if pred in {"prov:wasDerivedFrom"} and Ontology.IsResourceString(value):
                    triples.append((subject, pred, value))
                else:
                    for term in Ontology._value_terms(value):
                        triples.append((subject, pred, term))
        return Ontology._dedupe(triples)

    @staticmethod
    def _dedupe(triples: Iterable[Tuple[Any, Any, Any]]):
        seen, result = set(), []
        for t in triples or []:
            key = repr(t)
            if key not in seen:
                seen.add(key); result.append(t)
        return result

    # ------------------------------------------------------------------
    # TGraph serialization
    # ------------------------------------------------------------------
    @staticmethod
    def GraphTriples(graph: Any, includeVertices: bool = True, includeEdges: bool = True,
                     includeDictionaries: bool = True, includeBOT: bool = True,
                     namespacePrefix: str = "inst", silent: bool = False):
        try:
            from topologicpy.TGraph import TGraph
        except Exception:
            TGraph = None
        if TGraph is None or not isinstance(graph, TGraph):
            return Ontology.Triples(graph, includeDictionaries=includeDictionaries,
                                    includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=silent)

        graph_subject = Ontology._identity(graph, "graph", "graph", namespacePrefix)
        triples = Ontology.Triples(graph, subject=graph_subject, includeDictionaries=includeDictionaries,
                                   includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True)
        triples.append((graph_subject, "rdf:type", "top:Graph"))
        triples.append((graph_subject, "top:directed", Ontology._literal(bool(getattr(graph, "_directed", False)))))
        triples.append((graph_subject, "top:allowsSelfLoops", Ontology._literal(bool(getattr(graph, "_allow_self_loops", True)))))
        triples.append((graph_subject, "top:allowsParallelEdges", Ontology._literal(bool(getattr(graph, "_allow_parallel_edges", False)))))

        vertices = TGraph.Vertices(graph, asTopologic=False, active=True, copy=False) or []
        vertex_subjects: Dict[int, str] = {}
        for i, v in enumerate(vertices):
            idx = v.get("index", i)
            subject = Ontology._identity(v, "node", idx, namespacePrefix)
            vertex_subjects[idx] = subject
            if includeVertices:
                triples.append((graph_subject, "top:hasNode", subject))
                vd = dict(v.get("dictionary", {}) or {})
                vd.setdefault(Ontology.ONTOLOGY_CLASS_KEY, "top:Node")
                v_export = dict(v)
                v_export["dictionary"] = vd
                triples.extend(Ontology.Triples(v_export, subject=subject, includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True))
                coords = None
                try:
                    coords = TGraph.Coordinates(graph, idx, default=None)
                except Exception:
                    coords = None
                if coords is not None and len(coords) >= 3:
                    triples.append((subject, "top:x", Ontology._literal(float(coords[0]))))
                    triples.append((subject, "top:y", Ontology._literal(float(coords[1]))))
                    triples.append((subject, "top:z", Ontology._literal(float(coords[2]))))
                else:
                    for axis in ("x", "y", "z"):
                        if axis in vd:
                            triples.append((subject, "top:" + axis, Ontology._literal(vd[axis])))
                triples.append((subject, "top:index", Ontology._literal(idx)))

        if includeEdges:
            edges = TGraph.Edges(graph, asTopologic=False, active=True, copy=False) or []
            for i, e in enumerate(edges):
                idx = e.get("index", i)
                subject = Ontology._identity(e, "relationship", idx, namespacePrefix)
                triples.append((graph_subject, "top:hasRelationship", subject))
                ed = dict(e.get("dictionary", {}) or {})
                ed.setdefault(Ontology.ONTOLOGY_CLASS_KEY, "top:Relationship")
                e_export = dict(e)
                e_export["dictionary"] = ed
                triples.extend(Ontology.Triples(e_export, subject=subject, includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=True))
                src, dst = e.get("src"), e.get("dst")
                s = vertex_subjects.get(src); o = vertex_subjects.get(dst)
                if s is not None: triples.append((subject, "top:startsAt", s))
                if o is not None: triples.append((subject, "top:endsAt", o))
                triples.append((subject, "top:index", Ontology._literal(idx)))
                triples.append((subject, "top:directed", Ontology._literal(bool(e.get("directed", getattr(graph, "_directed", False))))))
                pred = (
                    ed.get("ontology_predicate")
                    or ed.get("ontologyPredicate")
                    or ed.get("predicate")
                    or "top:connectsTo"
                )
                inv = ed.get("inverse_predicate") or ed.get("inversePredicate")
                if pred:
                    pq = Ontology.PropertyQName(pred)
                    if pq and not pq.startswith("dict:"):
                        triples.append((subject, "top:hasPredicate", pq))
                        if s is not None and o is not None: triples.append((s, pq, o))
                if inv:
                    iq = Ontology.PropertyQName(inv)
                    if iq and not iq.startswith("dict:"):
                        triples.append((subject, "top:hasInversePredicate", iq))
                        if s is not None and o is not None: triples.append((o, iq, s))
        return Ontology._dedupe(triples)

    # ------------------------------------------------------------------
    # RDF conversion / serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _rdflib_term(term: Any, role: str, graph):
        rd = Ontology._rdflib(silent=True)
        if rd is None: return None
        if isinstance(term, (rd.URIRef, rd.BNode, rd.Literal)): return term
        if isinstance(term, _RDFLiteral):
            dt = rd.URIRef(term.datatype) if term.datatype else None
            return rd.Literal(term.lexical, datatype=dt, lang=term.language)
        text = str(term).strip()
        if role == "object" and text.startswith('"'):
            try:
                from rdflib.util import from_n3
                parsed = from_n3(text, nsm=graph.namespace_manager)
                if parsed is not None:
                    return parsed
            except Exception:
                pass
        if role == "object" and not Ontology.IsResourceString(term):
            return rd.Literal(term)
        if text.startswith("_:"): return rd.BNode(text[2:])
        uri = Ontology.ExpandQName(text, defaultValue=None)
        if uri: return rd.URIRef(uri)
        if text.startswith(("http://", "https://")): return rd.URIRef(text)
        if role in {"subject", "predicate"}:
            prefix = "dict" if role == "predicate" else "inst"
            return rd.URIRef(Ontology.NAMESPACES[prefix] + Ontology._safe_local_name(text))
        return rd.Literal(text)

    @staticmethod
    def RDFGraph(topology: Any, includeGraph: bool = True, includeDictionaries: bool = True,
                 includeBOT: bool = True, namespacePrefix: str = "inst",
                 instanceNamespace: str = "http://w3id.org/topologicpy/instance#", silent: bool = False):
        rd = Ontology._rdflib(silent=silent)
        if rd is None: return None
        g = rd.Graph()
        namespaces = dict(Ontology.NAMESPACES); namespaces[namespacePrefix] = instanceNamespace
        for prefix, uri in namespaces.items(): g.bind(prefix, rd.Namespace(uri))
        try:
            from topologicpy.TGraph import TGraph
            is_graph = isinstance(topology, TGraph)
        except Exception:
            is_graph = False
        triples = (Ontology.GraphTriples(topology, includeVertices=includeGraph, includeEdges=includeGraph,
                                         includeDictionaries=includeDictionaries, includeBOT=includeBOT,
                                         namespacePrefix=namespacePrefix, silent=silent)
                   if is_graph else Ontology.Triples(topology, includeDictionaries=includeDictionaries,
                                                     includeBOT=includeBOT, namespacePrefix=namespacePrefix, silent=silent))
        for s, p, o in triples:
            ss = Ontology._rdflib_term(s, "subject", g)
            pp = Ontology._rdflib_term(p, "predicate", g)
            oo = Ontology._rdflib_term(o, "object", g)
            if None not in (ss, pp, oo): g.add((ss, pp, oo))
        return g

    @staticmethod
    def RDFString(topology: Any, format: str = "turtle", **kwargs):
        g = Ontology.RDFGraph(topology, **kwargs)
        if g is None: return None
        data = g.serialize(format=format)
        return data.decode("utf-8") if isinstance(data, bytes) else data

    @staticmethod
    def TTLString(topology: Any, **kwargs):
        kwargs.pop("format", None)
        rd = Ontology._rdflib(silent=True)
        if rd is not None:
            text = Ontology.RDFString(topology, format="turtle", **kwargs)
            if text is not None:
                return text
        includeGraph = kwargs.pop("includeGraph", True)
        includeDictionaries = kwargs.pop("includeDictionaries", True)
        includeBOT = kwargs.pop("includeBOT", True)
        namespacePrefix = kwargs.pop("namespacePrefix", "inst")
        instanceNamespace = kwargs.pop("instanceNamespace", Ontology.NAMESPACES["inst"])
        if Ontology._is_graph_like(topology):
            triples = Ontology.GraphTriples(topology, includeVertices=includeGraph, includeEdges=includeGraph,
                                             includeDictionaries=includeDictionaries, includeBOT=includeBOT,
                                             namespacePrefix=namespacePrefix, silent=kwargs.pop("silent", False))
        else:
            triples = Ontology.Triples(topology, includeDictionaries=includeDictionaries, includeBOT=includeBOT,
                                       namespacePrefix=namespacePrefix, silent=kwargs.pop("silent", False))
        return Ontology.TurtleFromTriples(triples, namespacePrefix=namespacePrefix,
                                          instanceNamespace=instanceNamespace, includeHeader=True)

    @staticmethod
    def ExportRDF(topology: Any, path: str, format: str = None, silent: bool = False, **kwargs):
        g = Ontology.RDFGraph(topology, silent=silent, **kwargs)
        if g is None: return None
        fmt = format or ({".ttl": "turtle", ".nt": "nt", ".rdf": "xml", ".xml": "xml", ".jsonld": "json-ld"}.get(Path(path).suffix.lower(), "turtle"))
        try:
            g.serialize(destination=str(path), format=fmt); return str(path)
        except Exception as exc:
            if not silent: print("Ontology.ExportRDF - Error:", exc)
            return None

    @staticmethod
    def ExportTTL(topology: Any, path: str, **kwargs):
        """Exports a topology/graph/dictionary as Turtle.

        Unlike :meth:`ExportRDF`, this method does not require RDFLib. It uses
        :meth:`TTLString`, whose deterministic fallback serializer preserves the
        same canonical predicates when RDFLib is unavailable.
        """
        silent = bool(kwargs.pop("silent", False))
        ttl = Ontology.TTLString(topology, silent=silent, **kwargs)
        if ttl is None:
            return None
        try:
            Path(path).write_text(ttl, encoding="utf-8")
            return str(path)
        except Exception as exc:
            if not silent:
                print("Ontology.ExportTTL - Error:", exc)
            return None

    @staticmethod
    def ExportOntologyTTL(path: str, includeBOT: bool = True, silent: bool = False):
        ttl = Ontology.OntologyTTLString(includeBOT=includeBOT, silent=silent)
        if ttl is None: return None
        try:
            Path(path).write_text(ttl, encoding="utf-8"); return str(path)
        except Exception as exc:
            if not silent: print("Ontology.ExportOntologyTTL - Error:", exc)
            return None

    @staticmethod
    def OntologyTriples(includeBOT: bool = True):
        g = Ontology.OntologyRDFGraph(silent=True)
        if g is None: return []
        result = []
        for s, p, o in g:
            ss = Ontology.QName(str(s), str(s)); pp = Ontology.QName(str(p), str(p))
            if hasattr(o, "datatype") or hasattr(o, "language"):
                try:
                    from rdflib import Literal
                    if isinstance(o, Literal):
                        oo = _RDFLiteral(str(o), str(o.datatype) if o.datatype else None, o.language)
                    else: oo = Ontology.QName(str(o), str(o))
                except Exception: oo = str(o)
            else: oo = Ontology.QName(str(o), str(o))
            result.append((ss, pp, oo))
        return result

    @staticmethod
    def TurtleFromTriples(triples: Iterable[Tuple[Any, Any, Any]], namespacePrefix: str = "inst",
                          instanceNamespace: str = "http://w3id.org/topologicpy/instance#",
                          namespaces: Dict[str, str] = None, includeHeader: bool = True, **kwargs):
        """Serializes triples to Turtle, with or without RDFLib."""
        ns = dict(Ontology.NAMESPACES)
        ns.update(dict(namespaces or {}))
        ns[namespacePrefix] = instanceNamespace
        rd = Ontology._rdflib(silent=True)
        if rd is not None:
            try:
                g = rd.Graph()
                for prefix, uri in ns.items():
                    g.bind(prefix, rd.Namespace(uri), replace=True)
                for subject, predicate, obj in triples or []:
                    ss = Ontology._rdflib_term(subject, "subject", g)
                    pp = Ontology._rdflib_term(predicate, "predicate", g)
                    oo = Ontology._rdflib_term(obj, "object", g)
                    if None not in (ss, pp, oo):
                        g.add((ss, pp, oo))
                data = g.serialize(format="turtle")
                return data.decode("utf-8") if isinstance(data, bytes) else str(data)
            except Exception:
                pass

        def resource_token(value):
            if value is None:
                return None
            text = str(value).strip()
            if text.startswith("<") and text.endswith(">"):
                return text
            if text.startswith(("http://", "https://", "urn:")):
                return "<" + text + ">"
            if text.startswith("_:"):
                return text
            if Ontology.IsQName(text):
                return text
            return "inst:" + Ontology._safe_local_name(text)

        def literal_token(value):
            if isinstance(value, _RDFLiteral):
                lexical = value.lexical
                datatype = value.datatype
                language = value.language
            else:
                # KnowledgeGraph stores literals as legal Turtle/N3 tokens.
                # Preserve such tokens verbatim in the dependency-free writer
                # instead of quoting the token a second time.
                raw = str(value).strip()
                if raw.startswith('"'):
                    return raw
                lexical = str(value)
                datatype = None
                language = None
            escaped = lexical.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            token = '"' + escaped + '"'
            if language:
                token += "@" + str(language)
            elif datatype:
                dt = Ontology.QName(datatype, defaultValue=None)
                token += "^^" + (dt if dt else "<" + str(datatype).strip("<>") + ">")
            return token

        lines = []
        if includeHeader:
            for prefix, uri in ns.items():
                lines.append(f"@prefix {prefix}: <{uri}> .")
            lines.append("")
        for subject, predicate, obj in Ontology._dedupe(triples or []):
            ss = resource_token(subject)
            pp = resource_token(predicate)
            if isinstance(obj, _RDFLiteral):
                oo = literal_token(obj)
            elif Ontology.IsResourceString(obj):
                oo = resource_token(obj)
            else:
                oo = literal_token(obj)
            if None not in (ss, pp, oo):
                lines.append(f"{ss} {pp} {oo} .")
        return "\n".join(lines) + ("\n" if lines else "")

    # ------------------------------------------------------------------
    # Lossless RDF -> TGraph bridge
    # ------------------------------------------------------------------
    @staticmethod
    def _python_value(literal):
        try: return literal.toPython()
        except Exception: return str(literal)

    @staticmethod
    def _properties_for_subject(g, subject, exclude_predicates=None):
        rd = Ontology._rdflib(silent=True); exclude = set(exclude_predicates or [])
        out = []
        for p, o in g.predicate_objects(subject):
            if p in exclude or (rd is not None and p == rd.RDF.type): continue
            out.append({"predicate": str(p), "object": Ontology._encoded_rdf_object(o)})
        return out

    @staticmethod
    def _convenience_dictionary(g, subject, structural_predicates=None):
        rd = Ontology._rdflib(silent=True); d: Dict[str, Any] = {}
        if rd is None: return d
        subject_token = "_:" + str(subject) if isinstance(subject, rd.BNode) else str(subject)
        d[Ontology.RDF_URI_KEY] = subject_token
        d[Ontology.URI_KEY] = subject_token
        types = [str(o) for o in g.objects(subject, rd.RDF.type)]
        if types: d[Ontology.RDF_TYPES_KEY] = types
        top_ns = Ontology.NAMESPACES["top"]
        top_types = [Ontology.QName(t, t) for t in types if t.startswith(top_ns)]
        if top_types: d[Ontology.ONTOLOGY_CLASS_KEY] = top_types[0]
        labels = list(g.objects(subject, rd.RDFS.label))
        if labels: d[Ontology.LABEL_KEY] = str(labels[0])
        excluded = set(structural_predicates or []) | {rd.RDF.type}
        d[Ontology.RDF_PROPERTIES_KEY] = Ontology._properties_for_subject(g, subject, excluded)
        # Convenience values only for canonical top:/dict: datatype properties.
        for p in set(g.predicates(subject, None)):
            if p in excluded or p == rd.RDFS.label: continue
            q = Ontology.QName(str(p), str(p))
            if not (q.startswith("top:") or q.startswith("dict:")): continue
            vals = list(g.objects(subject, p))
            if not vals or any(isinstance(v, (rd.URIRef, rd.BNode)) for v in vals): continue
            key = q.split(":", 1)[1]
            py = [Ontology._python_value(v) for v in vals]
            d[key] = py[0] if len(py) == 1 else py
        return d

    @staticmethod
    def GraphByRDFGraph(rdfGraph, graphSubject: str = None, namespacePrefix: str = "inst",
                        tolerance: float = 0.0001, silent: bool = False):
        rd = Ontology._rdflib(silent=silent)
        if rd is None or rdfGraph is None: return None
        try:
            from topologicpy.TGraph import TGraph
        except Exception:
            if not silent: print("Ontology.GraphByRDFGraph - Error: TGraph is unavailable. Returning None.")
            return None
        TOP = rd.Namespace(Ontology.NAMESPACES["top"])
        def as_uri(value):
            if value is None: return None
            expanded = Ontology.ExpandQName(value, defaultValue=str(value).strip("<>"))
            return rd.URIRef(expanded)
        gs = as_uri(graphSubject) if graphSubject else next(iter(rdfGraph.subjects(TOP.hasNode, None)), None)
        if gs is None:
            for s in rdfGraph.subjects(rd.RDF.type, TOP.Graph): gs = s; break
        if gs is None:
            if not silent: print("Ontology.GraphByRDFGraph - Error: Could not identify a graph subject. Returning None.")
            return None
        structural_graph = {TOP.hasNode, TOP.hasRelationship, TOP.directed, TOP.allowsSelfLoops, TOP.allowsParallelEdges}
        gd = Ontology._convenience_dictionary(rdfGraph, gs, structural_graph)
        def bool_obj(pred, default):
            value = next(iter(rdfGraph.objects(gs, pred)), None)
            return bool(value.toPython()) if value is not None else default
        directed = bool_obj(TOP.directed, False)
        allow_self = bool_obj(TOP.allowsSelfLoops, True)
        allow_parallel = bool_obj(TOP.allowsParallelEdges, False)
        graph = TGraph(directed=directed, allowSelfLoops=allow_self, allowParallelEdges=allow_parallel, dictionary=gd)

        node_subjects = list(dict.fromkeys(rdfGraph.objects(gs, TOP.hasNode)))
        node_index = {}
        node_structural = {TOP.x, TOP.y, TOP.z, TOP.index}
        for node in node_subjects:
            nd = Ontology._convenience_dictionary(rdfGraph, node, node_structural)
            for axis in ("x", "y", "z"):
                pred = TOP[axis]; val = next(iter(rdfGraph.objects(node, pred)), None)
                if val is not None: nd[axis] = Ontology._python_value(val)
            idx = graph.AddVertex(dictionary=nd, tolerance=tolerance, silent=True)
            node_index[node] = idx

        rel_structural = {TOP.startsAt, TOP.endsAt, TOP.hasPredicate, TOP.hasInversePredicate, TOP.directed, TOP.index}
        for rel in rdfGraph.objects(gs, TOP.hasRelationship):
            src = next(iter(rdfGraph.objects(rel, TOP.startsAt)), None)
            dst = next(iter(rdfGraph.objects(rel, TOP.endsAt)), None)
            if src not in node_index or dst not in node_index: continue
            ed = Ontology._convenience_dictionary(rdfGraph, rel, rel_structural)
            pred = next(iter(rdfGraph.objects(rel, TOP.hasPredicate)), None)
            inv = next(iter(rdfGraph.objects(rel, TOP.hasInversePredicate)), None)
            if pred is not None: ed["ontology_predicate"] = Ontology.QName(str(pred), str(pred))
            if inv is not None: ed["inverse_predicate"] = Ontology.QName(str(inv), str(inv))
            edge_dir = next(iter(rdfGraph.objects(rel, TOP.directed)), None)
            edge_dir = bool(edge_dir.toPython()) if edge_dir is not None else directed
            graph.AddEdge(node_index[src], node_index[dst], directed=edge_dir, dictionary=ed, silent=True)
        return graph

    @staticmethod
    def GraphByRDFFile(path: str, format: str = None, graphSubject: str = None,
                       namespacePrefix: str = "inst", tolerance: float = 0.0001, silent: bool = False):
        rd = Ontology._rdflib(silent=silent)
        if rd is None: return None
        g = rd.Graph()
        try:
            g.parse(str(path), format=format)
        except Exception as exc:
            if not silent: print("Ontology.GraphByRDFFile - Error:", exc)
            return None
        return Ontology.GraphByRDFGraph(g, graphSubject=graphSubject, namespacePrefix=namespacePrefix,
                                        tolerance=tolerance, silent=silent)

    @staticmethod
    def GraphByTTL(path: str, **kwargs):
        return Ontology.GraphByRDFFile(path, format="turtle", **kwargs)

    @staticmethod
    def GraphByTTLString(ttlString: str, graphSubject: str = None, namespacePrefix: str = "inst",
                         tolerance: float = 0.0001, silent: bool = False):
        rd = Ontology._rdflib(silent=silent)
        if rd is None: return None
        g = rd.Graph()
        try: g.parse(data=ttlString, format="turtle")
        except Exception as exc:
            if not silent: print("Ontology.GraphByTTLString - Error:", exc)
            return None
        return Ontology.GraphByRDFGraph(g, graphSubject=graphSubject, namespacePrefix=namespacePrefix,
                                        tolerance=tolerance, silent=silent)

    @staticmethod
    def Validate(topology: Any, silent: bool = False, requiredKeys: Iterable[str] = None, **kwargs) -> Dict[str, Any]:
        """Validates ontology annotations against the canonical TTL vocabulary."""
        report = {"valid": True, "ok": True, "errors": [], "warnings": []}
        if topology is None:
            report["valid"] = report["ok"] = False
            report["errors"].append("Input topology is None.")
            return report
        d = Ontology._dictionary(topology)
        required = list(requiredKeys or kwargs.get("required_keys") or [])
        for key in required:
            if key not in d or d.get(key) in (None, ""):
                report["errors"].append("Missing required key: " + str(key))

        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        if cls:
            q = Ontology.QName(cls, defaultValue=cls)
            if isinstance(q, str) and q.startswith("top:") and q not in Ontology._vocabulary()["classes"]:
                report["errors"].append("Unknown top: class: " + q)
        else:
            report["warnings"].append("No ontology_class is assigned.")

        vocab = Ontology._vocabulary()
        known = vocab["object"] | vocab["data"] | vocab["annotation"] | vocab.get("property", set())
        for key in d:
            pred = Ontology._predicate_for_internal_key(str(key))
            if pred and pred.startswith("top:") and pred not in known:
                report["errors"].append("Unknown top: predicate: " + pred)
        report["valid"] = report["ok"] = len(report["errors"]) == 0
        return report

    @staticmethod
    def ValidateGraph(graph: Any, silent: bool = False, **kwargs) -> Dict[str, Any]:
        """Validates a TGraph's ontology annotations and structural endpoints."""
        report = {"valid": True, "ok": True, "errors": [], "warnings": []}
        if graph is None:
            report["errors"].append("Input graph is None.")
            report["valid"] = report["ok"] = False
            return report
        base = Ontology.Validate(graph, silent=True, **kwargs)
        report["errors"].extend(base.get("errors", []))
        report["warnings"].extend(base.get("warnings", []))
        vertices = getattr(graph, "_vertices", None)
        edges = getattr(graph, "_edges", None)
        if not isinstance(vertices, list) or not isinstance(edges, list):
            report["errors"].append("Input object is not a valid TGraph representation.")
        else:
            active = {v.get("index", i) for i, v in enumerate(vertices) if isinstance(v, dict) and v.get("active", True)}
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict) or not edge.get("active", True):
                    continue
                src, dst = edge.get("src"), edge.get("dst")
                if src not in active:
                    report["errors"].append(f"Edge {edge.get('index', i)} has unresolved source endpoint: {src}")
                if dst not in active:
                    report["errors"].append(f"Edge {edge.get('index', i)} has unresolved target endpoint: {dst}")
        report["valid"] = report["ok"] = len(report["errors"]) == 0
        report["errors"] = list(dict.fromkeys(report["errors"]))
        report["warnings"] = list(dict.fromkeys(report["warnings"]))
        return report

    @staticmethod
    def AnnotateSubtopologies(topology: Any, topologyTypes: Iterable[str] = None,
                              inferClass: bool = True, silent: bool = False, **kwargs):
        """Annotates requested subtopologies using canonical TopologicPy classes.

        ``topologyTypes`` may contain names such as ``"Vertex"`` or ``"Face"``.
        For convenience, boolean keyword flags such as ``vertices=True`` and
        ``faces=True`` are also accepted. The input topology is returned.
        """
        if topology is None:
            return None
        try:
            from topologicpy.Topology import Topology
        except Exception:
            if not silent:
                print("Ontology.AnnotateSubtopologies - Error: Topology.py is unavailable. Returning None.")
            return None
        requested = list(topologyTypes or kwargs.pop("types", []) or [])
        aliases = {
            "vertices": "Vertex", "edges": "Edge", "wires": "Wire", "faces": "Face",
            "shells": "Shell", "cells": "Cell", "cellcomplexes": "CellComplex",
            "cell_complexes": "CellComplex", "clusters": "Cluster", "apertures": "Aperture",
        }
        for key, typename in aliases.items():
            if kwargs.get(key) is True:
                requested.append(typename)
        if not requested:
            requested = ["Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster"]
        singular_to_method = {
            "Vertex": "Vertices", "Edge": "Edges", "Wire": "Wires", "Face": "Faces",
            "Shell": "Shells", "Cell": "Cells", "CellComplex": "CellComplexes",
            "Cluster": "Clusters", "Aperture": "Apertures",
        }
        seen = set()
        for typename in requested:
            name = str(typename).strip()
            if not name:
                continue
            name = name[0].upper() + name[1:]
            method_name = singular_to_method.get(name, name + "s")
            extractor = getattr(Topology, method_name, None)
            if not callable(extractor):
                continue
            try:
                subs = extractor(topology) or []
            except Exception:
                continue
            for sub in subs:
                marker = id(sub)
                if marker in seen:
                    continue
                seen.add(marker)
                cls = "top:" + name if inferClass else kwargs.get("ontologyClass")
                if cls and Ontology.CanonicalClass(cls, defaultValue=None):
                    Ontology.Annotate(sub, ontologyClass=cls, category=Ontology.CategoryByClass(cls), silent=True)
        return topology

    @staticmethod
    def _undeclared_top_terms(rdfGraph: Any) -> List[str]:
        """Returns errors for undeclared resources in the ``top:`` namespace."""
        rd = Ontology._rdflib(silent=True)
        if rd is None or rdfGraph is None:
            return []

        vocab = Ontology._vocabulary()

        def expanded(qnames):
            return {
                Ontology.ExpandQName(qname, defaultValue=str(qname))
                for qname in qnames
            }

        classes = expanded(vocab["classes"])
        properties = expanded(
            vocab["object"]
            | vocab["data"]
            | vocab["annotation"]
            | vocab.get("property", set())
        )
        declared = classes | properties
        top_namespace = Ontology.NAMESPACES["top"]
        rdf_type = str(rd.RDF.type)
        has_predicate = top_namespace + "hasPredicate"
        has_inverse_predicate = top_namespace + "hasInversePredicate"
        errors = []

        def is_top_uri(term):
            return isinstance(term, rd.URIRef) and str(term).startswith(top_namespace)

        for subject, predicate, obj in rdfGraph:
            predicate_uri = str(predicate)

            if is_top_uri(predicate) and predicate_uri not in properties:
                errors.append("Unknown top: predicate: " + Ontology.QName(predicate_uri, predicate_uri))

            if predicate_uri == rdf_type and is_top_uri(obj) and str(obj) not in classes:
                errors.append("Unknown top: class: " + Ontology.QName(str(obj), str(obj)))
            elif predicate_uri in {has_predicate, has_inverse_predicate} and is_top_uri(obj) and str(obj) not in properties:
                errors.append("Unknown top: predicate resource: " + Ontology.QName(str(obj), str(obj)))
            elif is_top_uri(obj) and str(obj) not in declared:
                errors.append("Unknown top: resource: " + Ontology.QName(str(obj), str(obj)))

            if is_top_uri(subject) and str(subject) not in declared:
                errors.append("Unknown top: resource: " + Ontology.QName(str(subject), str(subject)))

        return list(dict.fromkeys(errors))

    @staticmethod
    def ValidateTTLString(ttlString: str, silent: bool = False) -> Dict[str, Any]:
        """Validates Turtle syntax and canonical TopologicPy vocabulary use.

        Every URI in the ``top:`` namespace must be declared by the canonical
        ontology as a class or property. Thus ingested Turtle is held to the
        same no-invented-terms rule as exported RDF.
        """
        report = {"available": False, "valid": None, "ok": None, "errors": [], "warnings": []}
        if not isinstance(ttlString, str) or not ttlString.strip():
            report["available"] = True
            report["valid"] = report["ok"] = False
            report["errors"].append("Input Turtle string is empty or invalid.")
            return report
        rd = Ontology._rdflib(silent=True)
        if rd is None:
            report["warnings"].append("RDFLib is unavailable; Turtle syntax was not parsed.")
            return report
        report["available"] = True
        try:
            g = rd.Graph()
            g.parse(data=ttlString, format="turtle")
            report["triple_count"] = len(g)
            report["errors"].extend(Ontology._undeclared_top_terms(g))
            report["valid"] = report["ok"] = len(report["errors"]) == 0
        except Exception as exc:
            report["valid"] = report["ok"] = False
            report["errors"].append(str(exc))
        return report

    @staticmethod
    def ValidateTTL(path: str, silent: bool = False) -> Dict[str, Any]:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            return {"available": True, "valid": False, "ok": False, "errors": [str(exc)], "warnings": []}
        return Ontology.ValidateTTLString(text, silent=silent)

    @staticmethod
    def ValidateRDFGraph(rdfGraph: Any, silent: bool = False) -> Dict[str, Any]:
        report = {"available": Ontology._rdflib(silent=True) is not None, "valid": None, "ok": None, "errors": [], "warnings": []}
        if rdfGraph is None:
            report["valid"] = report["ok"] = False
            report["errors"].append("Input RDF graph is None.")
            return report
        if not report["available"]:
            report["warnings"].append("RDFLib is unavailable; RDF graph validation was not performed.")
            return report
        try:
            list(rdfGraph)
            report["errors"].extend(Ontology._undeclared_top_terms(rdfGraph))
            report["valid"] = report["ok"] = len(report["errors"]) == 0
        except Exception as exc:
            report["valid"] = report["ok"] = False
            report["errors"].append(str(exc))
        return report


# Populate the public vocabulary tables from the canonical TTL.  This is data
# initialization only; no methods are replaced or monkey-patched.
try:
    Ontology._vocabulary()
except Exception:
    pass
