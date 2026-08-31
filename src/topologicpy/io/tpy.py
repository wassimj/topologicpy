# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Native TopologicPy ``.tpy`` archive codec.

A TPY archive is a ZIP container with two logically separate layers:

* exact OCCT BREP payloads under ``geometry/``;
* TopologicPy semantics in JSON (topology dictionaries, Content records, Context
  relationships, archive identities, and subtopology locators).

The BREP payload is canonical for geometry. JSON never approximates curves or
surfaces. No triangulation is performed by this codec.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_FORMAT = "TopologicPy Native Archive"
_SCHEMA_VERSION = 2
_GEOMETRY_FIDELITY = "exact_native"
_GEOMETRY_FORMAT = "OCCT-BREP"
_SUBTOPOLOGY_TYPES = (
    "vertex",
    "edge",
    "wire",
    "face",
    "shell",
    "cell",
    "cellcomplex",
    "cluster",
)


class TPYError(RuntimeError):
    """Raised when a TPY archive cannot be serialized losslessly."""


@dataclass
class _LocalRef:
    topology: Any
    owner_id: str
    locator: Dict[str, Any]
    record_id: str


# -----------------------------------------------------------------------------
# JSON-safe semantics
# -----------------------------------------------------------------------------


def _encode_value(value: Any) -> Any:
    """Encode supported dictionary values without silently stringifying them."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"$type": "float", "value": repr(value)}
        return value
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TPYError(
                    "TPY dictionaries require string keys; "
                    f"encountered key {key!r}."
                )
            result[key] = _encode_value(item)
        return result
    raise TPYError(
        "TPY cannot losslessly serialize dictionary value of type "
        f"{type(value).__name__}."
    )


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get("$type")
    if value_type == "tuple":
        return tuple(_decode_value(item) for item in value.get("items", []))
    if value_type == "float":
        text = value.get("value")
        if text == "nan":
            return float("nan")
        if text == "inf":
            return float("inf")
        if text == "-inf":
            return float("-inf")
        return float(text)
    return {key: _decode_value(item) for key, item in value.items()}


def _python_dictionary(topology) -> Dict[str, Any]:
    from topologicpy.Dictionary import Dictionary
    from topologicpy.Topology import Topology

    dictionary = Topology.Dictionary(topology, silent=True)
    if dictionary is None:
        return {}
    if isinstance(dictionary, dict):
        return dict(dictionary)
    try:
        result = Dictionary.PythonDictionary(dictionary)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    try:
        keys = Dictionary.Keys(dictionary) or []
        return {
            key: Dictionary.ValueAtKey(dictionary, key, None)
            for key in keys
        }
    except TypeError:
        try:
            keys = Dictionary.Keys(dictionary) or []
            return {key: Dictionary.ValueAtKey(dictionary, key) for key in keys}
        except Exception:
            return {}
    except Exception:
        return {}


def _semantic_dictionary(value) -> Dict[str, Any]:
    """Return a strict plain-Python dictionary from a semantic object."""
    dictionary = getattr(value, "dictionary", None)
    if dictionary is None:
        return {}
    if not isinstance(dictionary, dict):
        raise TPYError(
            f"Semantic object {type(value).__name__} has a non-dictionary "
            "dictionary payload."
        )
    return dict(dictionary)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def _brep_string(topology) -> str:
    from topologicpy.Core import Core

    text = None
    try:
        text = Core.Topology.BREPString(topology, 0)
    except Exception:
        pass
    if not isinstance(text, str) or not text.strip():
        try:
            text = Core.InstanceCall(topology, "BREPString", 0)
        except Exception:
            text = None
    if not isinstance(text, str) or not text.strip():
        raise TPYError("Could not obtain an exact OCCT BREP payload.")
    return text


def _by_brep_string(text: str):
    from topologicpy.Core import Core

    try:
        return Core.Topology.ByBREPString(text)
    except Exception:
        return None


def _unique(topologies: Iterable[Any]) -> List[Any]:
    from topologicpy.Topology import Topology

    result = []
    for topology in topologies or []:
        if topology is None:
            continue
        duplicate = False
        for existing in result:
            try:
                if Topology.IsSame(topology, existing, silent=True):
                    duplicate = True
                    break
            except Exception:
                if topology is existing:
                    duplicate = True
                    break
        if not duplicate:
            result.append(topology)
    return result


def _type_name(topology) -> Optional[str]:
    from topologicpy.Topology import Topology

    result = Topology.TypeAsString(topology, silent=True)
    return result.lower() if isinstance(result, str) else None


def _local_topologies(owner) -> Dict[str, List[Any]]:
    """Return self plus deterministic type-indexed subtopology lists."""
    from topologicpy.Topology import Topology

    result = {"self": [owner]}
    for topology_type in _SUBTOPOLOGY_TYPES:
        if _type_name(owner) == topology_type:
            result[topology_type] = [owner]
            continue
        try:
            items = Topology.SubTopologies(
                owner,
                subTopologyType=topology_type,
                silent=True,
            )
        except Exception:
            items = None
        result[topology_type] = _unique(items or [])
    return result


def _coords(vertex) -> Optional[List[float]]:
    from topologicpy.Vertex import Vertex

    try:
        result = Vertex.Coordinates(vertex, mantissa=None)
        if isinstance(result, (list, tuple)) and len(result) >= 3:
            return [float(result[0]), float(result[1]), float(result[2])]
    except Exception:
        pass
    return None


def _center(topology) -> Optional[List[float]]:
    from topologicpy.Topology import Topology

    try:
        vertex = Topology.CenterOfMass(topology, silent=True)
    except Exception:
        vertex = None
    return _coords(vertex) if vertex is not None else None


def _measure(topology, topology_type: str) -> Optional[float]:
    try:
        if topology_type == "edge":
            from topologicpy.Edge import Edge
            return float(Edge.Length(topology, mantissa=None, silent=True))
        if topology_type == "wire":
            from topologicpy.Wire import Wire
            return float(Wire.Length(topology, mantissa=None, silent=True))
        if topology_type == "face":
            from topologicpy.Face import Face
            return float(Face.Area(topology, mantissa=None, silent=True))
        if topology_type == "shell":
            from topologicpy.Shell import Shell
            return float(Shell.Area(topology, mantissa=None, silent=True))
        if topology_type == "cell":
            from topologicpy.Cell import Cell
            return float(Cell.Volume(topology, mantissa=None, silent=True))
        if topology_type == "cellcomplex":
            from topologicpy.CellComplex import CellComplex
            return float(CellComplex.Volume(topology, mantissa=None, silent=True))
    except Exception:
        pass
    return None


def _counts(topology) -> Dict[str, int]:
    from topologicpy.Topology import Topology

    result = {}
    for topology_type in ("vertex", "edge", "face", "cell"):
        try:
            items = Topology.SubTopologies(
                topology,
                subTopologyType=topology_type,
                silent=True,
            )
            result[topology_type] = len(_unique(items or []))
        except Exception:
            result[topology_type] = 0
    return result


def _fingerprint(topology) -> Dict[str, Any]:
    topology_type = _type_name(topology)
    brep = _brep_string(topology)
    fingerprint = {
        "type": topology_type,
        "brep_sha256": hashlib.sha256(brep.encode("utf-8")).hexdigest(),
        "counts": _counts(topology),
    }
    if topology_type == "vertex":
        fingerprint["center"] = _coords(topology)
    else:
        fingerprint["center"] = _center(topology)
    measure = _measure(topology, topology_type)
    if measure is not None and math.isfinite(measure):
        fingerprint["measure"] = measure
    return fingerprint


def _near(a, b, rel=1.0e-9, abs_tol=1.0e-9) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)
    except Exception:
        return False


def _fingerprint_matches(saved: Dict[str, Any], topology) -> bool:
    if not saved:
        return True
    current = _fingerprint(topology)
    if saved.get("type") != current.get("type"):
        return False
    # A byte-stable subshape BREP hash is the strongest locator check.
    if saved.get("brep_sha256") == current.get("brep_sha256"):
        return True
    if saved.get("counts") != current.get("counts"):
        return False
    a = saved.get("center")
    b = current.get("center")
    if a is not None or b is not None:
        if not (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b) == 3
            and all(_near(x, y, rel=1.0e-8, abs_tol=1.0e-8) for x, y in zip(a, b))
        ):
            return False
    if "measure" in saved or "measure" in current:
        if not _near(saved.get("measure"), current.get("measure"), rel=1.0e-8, abs_tol=1.0e-9):
            return False
    return True


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------


class _Writer:
    def __init__(self, root):
        from topologicpy.SemanticManager import SemanticManager

        self.root = root
        self.manager = SemanticManager.GetInstance()
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.owner_roots: Dict[str, Any] = {}
        self.owner_locals: Dict[str, Dict[str, List[Any]]] = {}
        self.local_refs: List[_LocalRef] = []
        self.record_by_locator: Dict[Tuple[str, str, int], str] = {}
        self.contents: Dict[str, Dict[str, Any]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.content_ids: Dict[int, str] = {}
        self.context_ids: Dict[int, str] = {}
        self.content_queue: List[Any] = []
        self.queued_contents = set()
        self.seeded_owners = set()
        self.next_object = 1
        self.next_record = 1
        self.next_content = 1
        self.next_context = 1

    def _new_id(self, prefix: str, attr: str) -> str:
        value = getattr(self, attr)
        setattr(self, attr, value + 1)
        return f"{prefix}{value:06d}"

    def _find_owner(self, topology) -> Optional[str]:
        from topologicpy.Topology import Topology

        for owner_id, owner in self.owner_roots.items():
            try:
                if Topology.IsSame(topology, owner, silent=True):
                    return owner_id
            except Exception:
                if topology is owner:
                    return owner_id
        return None

    def _has_semantics(self, topology) -> bool:
        if _python_dictionary(topology):
            return True
        if self.manager.content_for_topology(topology, create=False) is not None:
            return True
        if self.manager.contents_for_host(topology):
            return True
        return False

    def _add_owner(self, topology) -> str:
        existing = self._find_owner(topology)
        if existing is not None:
            return existing

        owner_id = self._new_id("g", "next_object")
        self.owner_roots[owner_id] = topology
        local = _local_topologies(topology)
        self.owner_locals[owner_id] = local

        owner_record = {
            "id": owner_id,
            "type": _type_name(topology),
            "geometry": f"geometry/{owner_id}.brep",
            "root_record": None,
            "subtopologies": {},
        }
        self.objects[owner_id] = owner_record

        root_record = self._make_record(topology, owner_id, {"kind": "self"})
        owner_record["root_record"] = root_record["id"]
        owner_record["records"] = {root_record["id"]: root_record}

        for topology_type in _SUBTOPOLOGY_TYPES:
            owner_record["subtopologies"][topology_type] = []
            for index, item in enumerate(local.get(topology_type, [])):
                try:
                    from topologicpy.Topology import Topology
                    if Topology.IsSame(item, topology, silent=True):
                        continue
                except Exception:
                    if item is topology:
                        continue

                # Persist only subtopologies that carry dictionary/semantic state.
                # Pure geometry remains represented once by the owning BREP.
                if not self._has_semantics(item):
                    continue
                self._ensure_local_record(owner_id, topology_type, index, item)

        return owner_id

    def _ensure_local_record(
        self,
        owner_id: str,
        topology_type: str,
        index: int,
        topology,
    ) -> Dict[str, Any]:
        key = (owner_id, topology_type, int(index))
        existing_id = self.record_by_locator.get(key)
        if existing_id is not None:
            return self.objects[owner_id]["records"][existing_id]
        record = self._make_record(
            topology,
            owner_id,
            {"kind": "subtopology", "type": topology_type, "index": int(index)},
        )
        self.objects[owner_id]["records"][record["id"]] = record
        self.objects[owner_id]["subtopologies"][topology_type].append(record["id"])
        self.record_by_locator[key] = record["id"]
        return record

    def _make_record(self, topology, owner_id: str, locator: Dict[str, Any]) -> Dict[str, Any]:
        record_id = self._new_id("r", "next_record")
        record = {
            "id": record_id,
            "locator": locator,
            "type": _type_name(topology),
            "dictionary": _encode_value(_python_dictionary(topology)),
            "fingerprint": _fingerprint(topology),
        }
        uuid_value = getattr(topology, "_uuid", None)
        if isinstance(uuid_value, str) and uuid_value:
            record["uuid"] = uuid_value
        self.local_refs.append(_LocalRef(topology, owner_id, locator, record_id))
        return record

    def _reference(self, topology) -> Dict[str, Any]:
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            raise TPYError("A semantic reference does not point to a valid topology.")

        for local_ref in self.local_refs:
            try:
                if Topology.IsSame(topology, local_ref.topology, silent=True):
                    return {"object": local_ref.owner_id, "record": local_ref.record_id}
            except Exception:
                if topology is local_ref.topology:
                    return {"object": local_ref.owner_id, "record": local_ref.record_id}

        # Reuse an unrecorded subtopology of an existing BREP whenever possible.
        for owner_id, local in self.owner_locals.items():
            owner = self.owner_roots[owner_id]
            try:
                if Topology.IsSame(topology, owner, silent=True):
                    return {
                        "object": owner_id,
                        "record": self.objects[owner_id]["root_record"],
                    }
            except Exception:
                if topology is owner:
                    return {
                        "object": owner_id,
                        "record": self.objects[owner_id]["root_record"],
                    }

            for topology_type in _SUBTOPOLOGY_TYPES:
                for index, item in enumerate(local.get(topology_type, [])):
                    try:
                        same = Topology.IsSame(topology, item, silent=True)
                    except Exception:
                        same = topology is item
                    if same:
                        record = self._ensure_local_record(
                            owner_id, topology_type, index, item
                        )
                        return {"object": owner_id, "record": record["id"]}

        # Otherwise this topology owns a separate exact BREP payload.
        owner_id = self._add_owner(topology)
        return {
            "object": owner_id,
            "record": self.objects[owner_id]["root_record"],
        }

    def _enqueue_content(self, content):
        if content is None:
            return
        key = id(content)
        if key in self.queued_contents:
            return
        self.queued_contents.add(key)
        self.content_queue.append(content)

    def _seed_owner_contents(self, owner_id: str):
        """Seed semantics from a root/content-owned BREP, but not context-only hosts."""
        if owner_id in self.seeded_owners:
            return
        self.seeded_owners.add(owner_id)

        local = self.owner_locals.get(owner_id) or {}
        topologies = [self.owner_roots[owner_id]]
        for topology_type in _SUBTOPOLOGY_TYPES:
            topologies.extend(local.get(topology_type, []))
        topologies = _unique(topologies)

        for topology in topologies:
            represented = self.manager.content_for_topology(topology, create=False)
            if represented is not None:
                self._enqueue_content(represented)
            for content in self.manager.contents_for_host(topology):
                self._enqueue_content(content)

    def _content_id(self, content) -> str:
        from topologicpy.Aperture import Aperture
        from topologicpy.Content import Content

        key = id(content)
        existing = self.content_ids.get(key)
        if existing is not None:
            return existing
        if not isinstance(content, Content):
            raise TPYError("SemanticManager returned an invalid Content object.")

        content_id = self._new_id("c", "next_content")
        topology_ref = self._reference(content.Topology())
        record = {
            "id": content_id,
            "kind": "Aperture" if isinstance(content, Aperture) else "Content",
            "topology": topology_ref,
            "dictionary": _encode_value(_semantic_dictionary(content)),
        }
        uuid_value = getattr(content, "_uuid", None)
        if isinstance(uuid_value, str) and uuid_value:
            record["uuid"] = uuid_value
        self.contents[content_id] = record
        self.content_ids[key] = content_id

        # Content geometry can itself host nested Content. Follow that semantic
        # branch, but do not transitively import unrelated Content from hosts that
        # are referenced only by this Content's Contexts.
        self._seed_owner_contents(topology_ref["object"])
        return content_id

    def _context_id(self, context, content_id: str) -> str:
        from topologicpy.Context import Context
        from topologicpy.Topology import Topology

        key = id(context)
        existing = self.context_ids.get(key)
        if existing is not None:
            return existing
        if not isinstance(context, Context):
            raise TPYError("SemanticManager returned an invalid Context object.")

        host = context.Host()
        if not Topology.IsInstance(host, "Topology"):
            raise TPYError("A Context does not reference a valid host topology.")

        context_id = self._new_id("x", "next_context")
        record = {
            "id": context_id,
            "content": content_id,
            "host": self._reference(host),
            "parameters": _encode_value(context.Parameters()),
            "dictionary": _encode_value(_semantic_dictionary(context)),
        }
        uuid_value = getattr(context, "_uuid", None)
        if isinstance(uuid_value, str) and uuid_value:
            record["uuid"] = uuid_value
        self.contexts[context_id] = record
        self.context_ids[key] = context_id
        return context_id

    def build(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        root_owner = self._add_owner(self.root)
        self._seed_owner_contents(root_owner)

        index = 0
        while index < len(self.content_queue):
            content = self.content_queue[index]
            index += 1
            content_id = self._content_id(content)
            for context in self.manager.contexts_for_content(content):
                self._context_id(context, content_id)

        geometry_payloads = {
            owner_id: _brep_string(owner)
            for owner_id, owner in self.owner_roots.items()
        }

        semantics = {
            "schema_version": _SCHEMA_VERSION,
            "semantic_model": "Content-Context-v1",
            "root": {
                "object": root_owner,
                "record": self.objects[root_owner]["root_record"],
            },
            "objects": self.objects,
            "contents": self.contents,
            "contexts": self.contexts,
        }
        return semantics, geometry_payloads


# -----------------------------------------------------------------------------
# Reader
# -----------------------------------------------------------------------------


class _Reader:
    def __init__(self, semantics: Dict[str, Any], geometry_payloads: Dict[str, str]):
        self.semantics = semantics
        self.geometry_payloads = geometry_payloads
        self.owners: Dict[str, Any] = {}
        self.records: Dict[str, Any] = {}
        self.contents: Dict[str, Any] = {}
        self.contexts: Dict[str, Any] = {}

    def _candidate_for_record(self, owner, record: Dict[str, Any]):
        locator = record.get("locator") or {}
        if locator.get("kind") == "self":
            return owner
        topology_type = locator.get("type")
        index = locator.get("index")
        local = _local_topologies(owner).get(topology_type, [])
        if isinstance(index, int) and 0 <= index < len(local):
            candidate = local[index]
            if _fingerprint_matches(record.get("fingerprint") or {}, candidate):
                return candidate
        matches = [
            item
            for item in local
            if _fingerprint_matches(record.get("fingerprint") or {}, item)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_ref(self, ref: Dict[str, Any]):
        if not isinstance(ref, dict):
            return None
        return self.records.get(ref.get("record"))

    def load_geometry(self):
        from topologicpy.Topology import Topology

        objects = self.semantics.get("objects") or {}
        for owner_id, object_record in objects.items():
            text = self.geometry_payloads.get(owner_id)
            topology = _by_brep_string(text) if isinstance(text, str) else None
            if not Topology.IsInstance(topology, "Topology"):
                raise TPYError(f"Could not reconstruct BREP object {owner_id}.")
            expected_type = object_record.get("type")
            actual_type = _type_name(topology)
            if expected_type != actual_type:
                raise TPYError(
                    f"BREP object {owner_id} changed type from {expected_type} "
                    f"to {actual_type}."
                )
            self.owners[owner_id] = topology

        for owner_id, object_record in objects.items():
            owner = self.owners[owner_id]
            for record_id, record in (object_record.get("records") or {}).items():
                topology = self._candidate_for_record(owner, record)
                if topology is None:
                    raise TPYError(
                        f"Could not resolve semantic subtopology record {record_id} "
                        f"inside object {owner_id}."
                    )
                dictionary = _decode_value(record.get("dictionary") or {})
                Topology.SetDictionary(topology, dictionary, silent=True)
                uuid_value = record.get("uuid")
                if isinstance(uuid_value, str) and uuid_value:
                    try:
                        topology._uuid = uuid_value
                    except Exception:
                        pass
                self.records[record_id] = topology

    def load_contents(self):
        from topologicpy.Aperture import Aperture
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager
        from topologicpy.Topology import Topology

        manager = SemanticManager.GetInstance()
        for content_id, record in (self.semantics.get("contents") or {}).items():
            topology = self._resolve_ref(record.get("topology"))
            if not Topology.IsInstance(topology, "Topology"):
                raise TPYError(f"Content {content_id} references invalid geometry.")

            kind = record.get("kind")
            dictionary = _decode_value(record.get("dictionary") or {})
            uuid_value = record.get("uuid")
            if kind == "Aperture":
                content = Aperture(topology, dictionary=dictionary, uuid_value=uuid_value)
            elif kind == "Content":
                content = Content(topology, dictionary=dictionary, uuid_value=uuid_value)
            else:
                raise TPYError(f"Content {content_id} has unsupported kind {kind!r}.")

            registered = manager.content_for_topology(
                content,
                aperture=(kind == "Aperture"),
                create=False,
                dictionary=dictionary,
            )
            if registered is None:
                raise TPYError(f"Could not register Content {content_id}.")
            # Preserve the archive semantic identity even when the manager had to
            # normalize/promote the object during registration.
            if isinstance(uuid_value, str) and uuid_value:
                registered._uuid = uuid_value
            self.contents[content_id] = registered

    def load_contexts(self):
        from topologicpy.Aperture import Aperture
        from topologicpy.Context import Context
        from topologicpy.SemanticManager import SemanticManager
        from topologicpy.Topology import Topology

        manager = SemanticManager.GetInstance()
        for context_id, record in (self.semantics.get("contexts") or {}).items():
            content = self.contents.get(record.get("content"))
            host = self._resolve_ref(record.get("host"))
            if content is None:
                raise TPYError(f"Context {context_id} references invalid Content.")
            if not Topology.IsInstance(host, "Topology"):
                raise TPYError(f"Context {context_id} references an invalid host topology.")

            parameters = _decode_value(record.get("parameters"))
            dictionary = _decode_value(record.get("dictionary") or {})
            uuid_value = record.get("uuid")
            context = Context(
                content=content,
                host=host,
                parameters=parameters,
                dictionary=dictionary,
                uuid_value=uuid_value,
            )
            _, registered = manager.register(
                content,
                host,
                aperture=isinstance(content, Aperture),
                parameters=parameters,
                context_dictionary=dictionary,
                context=context,
            )
            if registered is None:
                raise TPYError(f"Could not register Context {context_id}.")
            if isinstance(uuid_value, str) and uuid_value:
                registered._uuid = uuid_value
            self.contexts[context_id] = registered

    def result(self):
        root = self.semantics.get("root") or {}
        topology = self.records.get(root.get("record"))
        if topology is None:
            raise TPYError("TPY root topology could not be resolved.")
        return topology


# -----------------------------------------------------------------------------
# Public codec
# -----------------------------------------------------------------------------


class TPYCodec:
    """Codec for TopologicPy's exact native ``.tpy`` persistence format."""

    extension = ".tpy"
    preserves_brep = True
    preserves_curves = True
    preserves_surfaces = True
    preserves_topologic_metadata = True
    tessellated = False
    geometry_fidelity = _GEOMETRY_FIDELITY

    @staticmethod
    def save(topology, path, overwrite: bool = False, silent: bool = False) -> bool:
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("TPY.Save - Error: The input topology is not valid. Returning False.")
            return False
        try:
            path = os.fspath(path)
        except Exception:
            if not silent:
                print("TPY.Save - Error: The input path is not valid. Returning False.")
            return False
        if not path.lower().endswith(".tpy"):
            path += ".tpy"
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("TPY.Save - Error: The output file already exists. Returning False.")
            return False

        try:
            writer = _Writer(topology)
            semantics, geometry_payloads = writer.build()

            try:
                import topologicpy
                version = getattr(topologicpy, "__version__", None)
            except Exception:
                version = None

            semantics_text = json.dumps(
                semantics,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file_hashes = {
                "semantics.json": hashlib.sha256(
                    semantics_text.encode("utf-8")
                ).hexdigest()
            }
            for owner_id, brep in geometry_payloads.items():
                file_hashes[f"geometry/{owner_id}.brep"] = hashlib.sha256(
                    brep.encode("utf-8")
                ).hexdigest()

            manifest = {
                "format": _FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "semantic_model": semantics["semantic_model"],
                "geometry_kernel": "OCCT",
                "geometry_format": _GEOMETRY_FORMAT,
                "geometry_fidelity": _GEOMETRY_FIDELITY,
                "topologicpy_version": version,
                "root": semantics["root"],
                "object_count": len(geometry_payloads),
                "content_count": len(semantics.get("contents") or {}),
                "context_count": len(semantics.get("contexts") or {}),
                "sha256": file_hashes,
            }

            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                prefix=".topologicpy_",
                suffix=".tpy.tmp",
                dir=directory,
            )
            os.close(fd)
            try:
                with zipfile.ZipFile(
                    temporary_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                ) as archive:
                    archive.writestr(
                        "manifest.json",
                        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
                    )
                    archive.writestr("semantics.json", semantics_text)
                    for owner_id, brep in geometry_payloads.items():
                        archive.writestr(f"geometry/{owner_id}.brep", brep)
                os.replace(temporary_path, path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            return True
        except Exception as error:
            if not silent:
                print(f"TPY.Save - Error: {error}. Returning False.")
            return False

    @staticmethod
    def load(path, silent: bool = False):
        try:
            path = os.fspath(path)
        except Exception:
            if not silent:
                print("TPY.Load - Error: The input path is not valid. Returning None.")
            return None
        if not os.path.exists(path) and not path.lower().endswith(".tpy"):
            alternate = path + ".tpy"
            if os.path.exists(alternate):
                path = alternate
        if not os.path.isfile(path):
            if not silent:
                print("TPY.Load - Error: The input file does not exist. Returning None.")
            return None

        try:
            with zipfile.ZipFile(path, "r") as archive:
                manifest_bytes = archive.read("manifest.json")
                semantics_bytes = archive.read("semantics.json")
                manifest = json.loads(manifest_bytes.decode("utf-8"))
                semantics = json.loads(semantics_bytes.decode("utf-8"))

                if manifest.get("format") != _FORMAT:
                    raise TPYError("The archive is not a TopologicPy native archive.")
                if int(manifest.get("schema_version", -1)) != _SCHEMA_VERSION:
                    raise TPYError(
                        "Unsupported TPY schema version "
                        f"{manifest.get('schema_version')}."
                    )
                if manifest.get("semantic_model") != "Content-Context-v1":
                    raise TPYError("Unsupported TPY semantic model.")
                if semantics.get("semantic_model") != manifest.get("semantic_model"):
                    raise TPYError("TPY semantic-model metadata is inconsistent.")
                if manifest.get("geometry_format") != _GEOMETRY_FORMAT:
                    raise TPYError("Unsupported TPY geometry encoding.")
                if manifest.get("geometry_fidelity") != _GEOMETRY_FIDELITY:
                    raise TPYError("The TPY archive does not declare exact native geometry.")
                if manifest.get("root") != semantics.get("root"):
                    raise TPYError("TPY root metadata is inconsistent.")
                if int(manifest.get("object_count", -1)) != len(
                    semantics.get("objects") or {}
                ):
                    raise TPYError("TPY object-count metadata is inconsistent.")
                if int(manifest.get("content_count", -1)) != len(
                    semantics.get("contents") or {}
                ):
                    raise TPYError("TPY content-count metadata is inconsistent.")
                if int(manifest.get("context_count", -1)) != len(
                    semantics.get("contexts") or {}
                ):
                    raise TPYError("TPY context-count metadata is inconsistent.")

                expected_hashes = manifest.get("sha256") or {}
                actual_semantics_hash = hashlib.sha256(semantics_bytes).hexdigest()
                if expected_hashes.get("semantics.json") != actual_semantics_hash:
                    raise TPYError("TPY semantics checksum verification failed.")

                geometry_payloads = {}
                for owner_id, object_record in (semantics.get("objects") or {}).items():
                    geometry_path = object_record.get("geometry")
                    if not isinstance(geometry_path, str):
                        raise TPYError(f"Object {owner_id} has no geometry payload.")
                    geometry_bytes = archive.read(geometry_path)
                    actual_hash = hashlib.sha256(geometry_bytes).hexdigest()
                    if expected_hashes.get(geometry_path) != actual_hash:
                        raise TPYError(
                            f"TPY geometry checksum verification failed for {owner_id}."
                        )
                    geometry_payloads[owner_id] = geometry_bytes.decode(
                        "utf-8",
                        errors="strict",
                    )

            reader = _Reader(semantics, geometry_payloads)
            reader.load_geometry()
            reader.load_contents()
            reader.load_contexts()
            return reader.result()
        except Exception as error:
            if not silent:
                print(f"TPY.Load - Error: {error}. Returning None.")
            return None


__all__ = ["TPYCodec", "TPYError"]
