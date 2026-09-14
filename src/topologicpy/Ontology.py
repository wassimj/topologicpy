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
        """Returns the packaged canonical TTL path when available."""
        try:
            p = resources.files("topologicpy").joinpath("ontology/topologicpy.ttl")
            if p.is_file():
                return Path(str(p))
        except Exception:
            pass
        # Development checkout fallback.
        here = Path(__file__).resolve()
        candidates = [
            here.parent / "ontology" / "topologicpy.ttl",
            here.parents[2] / "ontology" / "topologicpy.ttl",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @staticmethod
    def OntologyTTLString(includeBOT: bool = True, silent: bool = False) -> Optional[str]:
        path = Ontology._ontology_resource_path()
        if path is None:
            if not silent:
                print("Ontology.OntologyTTLString - Error: Canonical ontology file was not found. Returning None.")
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            if not silent:
                print("Ontology.OntologyTTLString - Error: Could not read canonical ontology. Returning None.")
                print("Error:", exc)
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
        if Ontology._VOCAB_CACHE is not None:
            return Ontology._VOCAB_CACHE
        rd = Ontology._rdflib(silent=True)
        g = Ontology.OntologyRDFGraph(silent=True)
        result = {"classes": set(), "object": set(), "data": set(), "annotation": set(), "super": {}}
        if rd is not None and g is not None:
            RDF, RDFS, OWL = rd.RDF, rd.RDFS, rd.OWL
            for kind, bucket in ((OWL.Class, "classes"), (OWL.ObjectProperty, "object"),
                                 (OWL.DatatypeProperty, "data"), (OWL.AnnotationProperty, "annotation")):
                for s in g.subjects(RDF.type, kind):
                    q = Ontology.QName(str(s), defaultValue=str(s))
                    result[bucket].add(q)
            for s, o in g.subject_objects(RDFS.subClassOf):
                sq, oq = Ontology.QName(str(s), str(s)), Ontology.QName(str(o), str(o))
                result["super"].setdefault(sq, []).append(oq)
        Ontology._VOCAB_CACHE = result
        Ontology.TOP_SUPERCLASSES = {k: list(v) for k, v in result["super"].items()}
        for cls in result["classes"]:
            Ontology.TOP_SUPERCLASSES.setdefault(cls, [])
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
        q = Ontology.QName(text, defaultValue=text)
        known = Ontology._vocabulary()["object"] | Ontology._vocabulary()["data"] | Ontology._vocabulary()["annotation"]
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
            if not silent: print("Ontology.SetClass - Error: Invalid ontology class. Returning input unchanged.")
            return obj
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
    def NormalizeDictionary(topology: Any, **kwargs):
        # Clean-slate policy intentionally performs no legacy alias conversion.
        d = dict(Ontology._dictionary(topology))
        if Ontology.ONTOLOGY_CLASS_KEY in d:
            canonical = Ontology.CanonicalClass(d[Ontology.ONTOLOGY_CLASS_KEY], defaultValue=None)
            if canonical is None:
                d.pop(Ontology.ONTOLOGY_CLASS_KEY, None)
            else:
                d[Ontology.ONTOLOGY_CLASS_KEY] = canonical
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
        for key in (Ontology.RDF_URI_KEY, Ontology.URI_KEY, Ontology.ONTOLOGY_URI_KEY, "uri", "URI"):
            value = d.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                return value if Ontology.IsResourceString(value) else f"{prefix}:{Ontology._safe_local_name(value)}"
        # Stable identifiers. LABEL IS INTENTIONALLY ABSENT.
        for key in ("uuid", "UUID", "id", "ID", Ontology.IFC_GUID_KEY, "ifc_guid", "ifcGUID", "GlobalId"):
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
                   "ontologyPredicate", "inversePredicate"}:
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
        d = Ontology._dictionary(topology)
        subject = subject or Ontology._identity(topology, prefix=namespacePrefix)
        triples: List[Tuple[Any, Any, Any]] = []

        # Preserve every imported RDF type, then assert canonical TopologicPy type.
        for t in d.get(Ontology.RDF_TYPES_KEY, []) or []:
            triples.append((subject, "rdf:type", Ontology.QName(t, defaultValue=t)))
        cls = Ontology.Class(topology)
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
        triples.append((graph_subject, "top:directed", Ontology._literal(bool(graph._directed))))
        triples.append((graph_subject, "top:allowsSelfLoops", Ontology._literal(bool(graph._allow_self_loops))))
        triples.append((graph_subject, "top:allowsParallelEdges", Ontology._literal(bool(graph._allow_parallel_edges))))

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
                triples.append((subject, "top:directed", Ontology._literal(bool(e.get("directed", graph._directed)))))
                pred = ed.get("ontology_predicate")
                inv = ed.get("inverse_predicate")
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
        if role == "object" and not Ontology.IsResourceString(term):
            return rd.Literal(term)
        text = str(term)
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
        return Ontology.RDFString(topology, format="turtle", **kwargs)

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
        return Ontology.ExportRDF(topology, path, format="turtle", **kwargs)

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
        rd = Ontology._rdflib(silent=True)
        if rd is None: return None
        g = rd.Graph()
        ns = dict(Ontology.NAMESPACES)
        ns.update(dict(namespaces or {}))
        ns[namespacePrefix] = instanceNamespace
        for p, u in ns.items(): g.bind(p, rd.Namespace(u), replace=True)
        for s, p, o in triples or []:
            ss = Ontology._rdflib_term(s, "subject", g); pp = Ontology._rdflib_term(p, "predicate", g); oo = Ontology._rdflib_term(o, "object", g)
            if None not in (ss, pp, oo): g.add((ss, pp, oo))
        data = g.serialize(format="turtle")
        return data.decode("utf-8") if isinstance(data, bytes) else data

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
    def Validate(topology: Any, silent: bool = False, **kwargs) -> Dict[str, Any]:
        report = {"valid": True, "ok": True, "errors": [], "warnings": []}
        d = Ontology._dictionary(topology)
        cls = d.get(Ontology.ONTOLOGY_CLASS_KEY)
        if cls:
            q = Ontology.CanonicalClass(cls, cls)
            if q.startswith("top:") and q not in Ontology._vocabulary()["classes"]:
                report["warnings"].append("Unknown top: class: " + q)
        for key in d:
            pred = Ontology._predicate_for_internal_key(str(key))
            if pred and pred.startswith("top:"):
                known = Ontology._vocabulary()["object"] | Ontology._vocabulary()["data"] | Ontology._vocabulary()["annotation"]
                if pred not in known: report["warnings"].append("Unknown top: predicate: " + pred)
        return report


# Populate the public vocabulary tables from the canonical TTL.  This is data
# initialization only; no methods are replaced or monkey-patched.
try:
    Ontology._vocabulary()
except Exception:
    pass
