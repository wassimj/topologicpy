# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""
Native TopologicPy persistence.

A ``.tpy`` file is a ZIP container containing:

    manifest.json
    geometry/<object-id>.brep
    geometry/<object-id-2>.brep
    ...

The BREP payloads preserve exact backend geometry. The manifest stores
TopologicPy dictionaries, dictionary-bearing subtopologies, Contents,
Apertures, and Context parameters separately from the geometry.

No pickle or executable serialization is used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile


_FORMAT = "topologicpy.tpy"
_VERSION = 2
_MANIFEST_NAME = "manifest.json"

_SUBTOPOLOGY_TYPES = (
    "cellcomplex",
    "cell",
    "shell",
    "face",
    "wire",
    "edge",
    "vertex",
    "cluster",
)


def _sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _encode_value(value):
    return TPYCodec._EncodeValue(value)


def _decode_value(value):
    return TPYCodec._DecodeValue(value, legacy=True)


def _python_dictionary(topology):
    from topologicpy.Dictionary import Dictionary
    from topologicpy.Topology import Topology

    try:
        dictionary = Topology.Dictionary(
            topology,
            silent=True,
        )
    except Exception:
        dictionary = None

    if dictionary is None:
        return {}

    try:
        result = Dictionary.PythonDictionary(
            dictionary,
            silent=True,
        )
    except Exception:
        result = None

    return (
        result
        if isinstance(result, dict)
        else {}
    )


def _apply_dictionary(topology, encoded):
    from topologicpy.Dictionary import Dictionary
    from topologicpy.Topology import Topology

    if topology is None:
        return None

    python_dictionary = _decode_value(
        encoded or {}
    )

    if not isinstance(
        python_dictionary,
        dict,
    ):
        python_dictionary = {}

    try:
        dictionary = Dictionary.ByPythonDictionary(
            python_dictionary,
            silent=True,
        )
    except Exception:
        dictionary = None

    if dictionary is None:
        return topology

    try:
        updated = Topology.SetDictionary(
            topology,
            dictionary,
            silent=True,
        )
        return (
            updated
            if updated is not None
            else topology
        )
    except Exception:
        return topology


def _vertex_coordinates(vertex):
    from topologicpy.Vertex import Vertex

    if vertex is None:
        return None

    try:
        coordinates = Vertex.Coordinates(
            vertex,
            mantissa=12,
        )
    except Exception:
        coordinates = None

    if (
        isinstance(
            coordinates,
            (list, tuple),
        )
        and len(coordinates) >= 3
    ):
        try:
            return [
                float(coordinates[0]),
                float(coordinates[1]),
                float(coordinates[2]),
            ]
        except Exception:
            pass

    return None


def _rounded_point(point, digits=8):
    if (
        not isinstance(
            point,
            (list, tuple),
        )
        or len(point) < 3
    ):
        return None

    try:
        return [
            round(float(point[0]), digits),
            round(float(point[1]), digits),
            round(float(point[2]), digits),
        ]
    except Exception:
        return None


def _safe_measure(topology, type_name):
    try:
        if type_name == "Edge":
            from topologicpy.Edge import Edge
            return Edge.Length(
                topology,
                mantissa=None,
                silent=True,
            )

        if type_name == "Wire":
            from topologicpy.Wire import Wire
            return Wire.Length(
                topology,
                mantissa=None,
                silent=True,
            )

        if type_name == "Face":
            from topologicpy.Face import Face
            return Face.Area(
                topology,
                mantissa=None,
                silent=True,
            )

        if type_name in (
            "Shell",
            "Cluster",
        ):
            from topologicpy.Face import Face
            from topologicpy.Topology import Topology

            faces = Topology.Faces(
                topology,
                silent=True,
            ) or []

            return sum(
                float(
                    Face.Area(
                        face,
                        mantissa=None,
                        silent=True,
                    )
                    or 0.0
                )
                for face in faces
            )

        if type_name == "Cell":
            from topologicpy.Cell import Cell
            return Cell.Volume(
                topology,
                mantissa=None,
                silent=True,
            )

        if type_name == "CellComplex":
            from topologicpy.CellComplex import (
                CellComplex,
            )
            return CellComplex.Volume(
                topology,
                mantissa=None,
                silent=True,
            )

    except Exception:
        pass

    return None


def _signature(topology):
    """
    Produce a backend-neutral geometric signature used as a fallback locator.

    BREP SHA-256 is tried first during reattachment. This signature is only
    needed when two backends serialize the same exact geometry differently.
    """
    from topologicpy.Topology import Topology

    type_name = Topology.TypeAsString(
        topology,
        silent=True,
    )

    signature = {
        "type": type_name,
    }

    try:
        center = Topology.CenterOfMass(
            topology,
            silent=True,
        )
    except Exception:
        center = None

    signature["center"] = _rounded_point(
        _vertex_coordinates(center)
    )

    measure = _safe_measure(
        topology,
        type_name,
    )

    if measure is not None:
        try:
            signature["measure"] = round(
                float(measure),
                8,
            )
        except Exception:
            pass

    try:
        vertices = Topology.Vertices(
            topology,
            silent=True,
        ) or []
    except Exception:
        vertices = []

    points = [
        _rounded_point(
            _vertex_coordinates(vertex)
        )
        for vertex in vertices
    ]

    points = [
        point
        for point in points
        if point is not None
    ]

    if points:
        signature["vertexCount"] = len(
            points
        )

        # Sorted coordinates remove traversal-order dependence.
        signature["vertices"] = sorted(
            points
        )[:128]

    # Add interior curve samples so a curved Edge is not identified only by
    # its topological endpoints.
    if type_name == "Edge":
        try:
            from topologicpy.Edge import Edge

            samples = []

            for u in (
                0.0,
                0.125,
                0.25,
                0.5,
                0.75,
                0.875,
                1.0,
            ):
                vertex = Edge.VertexByParameter(
                    topology,
                    u=u,
                )

                point = _rounded_point(
                    _vertex_coordinates(
                        vertex
                    )
                )

                if point is not None:
                    samples.append(point)

            if samples:
                direct = samples
                reverse = list(
                    reversed(samples)
                )

                signature["edgeSamples"] = min(
                    direct,
                    reverse,
                )

        except Exception:
            pass

    if type_name == "Face":
        try:
            from topologicpy.Face import Face

            samples = []

            for u, v in (
                (0.25, 0.25),
                (0.75, 0.25),
                (0.50, 0.50),
                (0.25, 0.75),
                (0.75, 0.75),
            ):
                vertex = Face.VertexByParameters(
                    topology,
                    u=u,
                    v=v,
                )

                point = _rounded_point(
                    _vertex_coordinates(
                        vertex
                    )
                )

                if point is not None:
                    samples.append(point)

            if samples:
                signature["surfaceSamples"] = (
                    samples
                )

        except Exception:
            pass

    try:
        signature["counts"] = {
            "vertices": len(
                Topology.Vertices(
                    topology,
                    silent=True,
                )
                or []
            ),
            "edges": len(
                Topology.Edges(
                    topology,
                    silent=True,
                )
                or []
            ),
            "faces": len(
                Topology.Faces(
                    topology,
                    silent=True,
                )
                or []
            ),
            "cells": len(
                Topology.Cells(
                    topology,
                    silent=True,
                )
                or []
            ),
        }
    except Exception:
        pass

    packed = json.dumps(
        signature,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    signature["hash"] = hashlib.sha256(
        packed.encode("utf-8")
    ).hexdigest()

    return signature


def _brep_data(topology):
    from topologicpy.Topology import Topology

    try:
        brep = Topology.BREPString(
            topology,
            version=3,
            silent=True,
        )
    except Exception:
        brep = None

    if not isinstance(
        brep,
        str,
    ) or len(brep) < 1:
        return None, None

    return (
        brep,
        _sha256_text(brep),
    )


def _same_topology(a, b):
    from topologicpy.Core import Core
    if a is b:
        return True
    # Callers supply native topologies. Avoid repeating public API type
    # dispatch for every pair, retaining the backend's exact identity test.
    try:
        return bool(Core.Topology.IsSame(a, b))
    except Exception:
        return False


def _subtopologies(topology):
    """
    Return unique subtopologies grouped by type, excluding the topology itself.
    """
    from topologicpy.Topology import Topology

    result = []

    for type_name in _SUBTOPOLOGY_TYPES:
        try:
            items = Topology.SubTopologies(
                topology,
                subTopologyType=type_name,
                silent=True,
            ) or []
        except Exception:
            items = []

        unique = []
        for item in TPYCodec._UniqueNative(items, topology):
            # Native identity excludes the carrier without writing a BREP
            # file for every edge/vertex. Large shading collections otherwise
            # spend most persistence time on dictionary-free constituents.
            if _same_topology(item, topology):
                continue
            unique.append(item)

        for index, item in enumerate(unique):
            result.append(
                (
                    type_name,
                    index,
                    item,
                )
            )

    return result


def _exact_geometry_key(topology):
    # Exact coordinate/connectivity identity, independent of unstable numerical
    # integrations used in centres, areas and volumes. BREP preserves doubles.
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    def point(v): return tuple(float(x).hex() for x in Vertex.Coordinates(v, mantissa=12))
    vertices = sorted(point(v) for v in Topology.Vertices(topology, silent=True) or [])
    edges = sorted(tuple(sorted(point(v) for v in Topology.Vertices(e, silent=True) or [])) for e in Topology.Edges(topology, silent=True) or [])
    return hashlib.sha256(json.dumps([Topology.TypeAsString(topology), vertices, edges], separators=(',', ':')).encode()).hexdigest()


def _make_locator(
    type_name,
    index,
    topology,
    native_path=None,
):
    from topologicpy.Topology import Topology
    _, brep_hash = _brep_data(
        topology
    )

    return {
        "type": type_name,
        "index": int(index),
        "brepSha256": brep_hash,
        "exactGeometry": _exact_geometry_key(topology),
        "signature": {"type": Topology.TypeAsString(topology,silent=True)} if native_path is not None else _signature(topology),
        **({"nativePath":native_path} if native_path is not None else {}),
    }


def _candidate_brep_hash(topology):
    _, value = _brep_data(
        topology
    )
    return value


def _resolve_locator(topology, locator, cache=None):
    from topologicpy.Topology import Topology
    if locator is None:
        return topology
    if not isinstance(locator, dict):
        return None
    def exact_target():
        kind=str(locator.get('type','')).lower()
        exact=locator.get('exactGeometry')
        if kind not in _SUBTOPOLOGY_TYPES or not exact:return None
        lookup=TPYCodec._LocatorIndex(topology,kind,cache)
        matches=TPYCodec._LocatorMatches(lookup,'exact',exact)
        return matches[0] if len(matches)==1 else None
    if 'nativePath' in locator:
        path = locator['nativePath']
        if not isinstance(path, list) or len(path) > 100 or any(type(index) is not int or index < 0 for index in path): return None
        try:
            shape = Topology.OCCTShape(topology, silent=True)
            if cache is not None:cache.setdefault(('native_root',id(topology)),topology)
            prefix=()
            for index in path:
                key=('native_children',id(topology),prefix)
                children=cache.get(key) if cache is not None else None
                if children is None:
                    children=TPYCodec._NativeChildren(shape)
                    if cache is not None:cache[key]=children
                shape=children[index];prefix=prefix+(index,)
            target = Topology.ByOCCTShape(shape, silent=True)
            # The hierarchy path distinguishes independent coincident shapes.
            if target is None or Topology.TypeAsString(target).lower() != str(locator.get('type', '')).lower(): return exact_target()
            # A checksum-verified BREP loaded without sewing preserves this
            # occurrence path exactly, including coincident independent faces.
            if cache is not None and cache.get(('verified_native_hierarchy',id(topology))) is topology:return target
            exact=locator.get('exactGeometry')
            if exact:
                if _exact_geometry_key(target)==exact:return target
                # Writing/reading BREP can cross a decimal-rounding tie in the
                # coordinate key. The path plus identical BREP bytes still
                # establishes exact native identity without a spatial guess.
                checksum=locator.get('brepSha256')
                if checksum and _candidate_brep_hash(target)==checksum:return target
                return exact_target()
            expected = locator.get('signature', {}).get('hash')
            if expected and _signature(target).get('hash') != expected:
                if _candidate_brep_hash(target) != locator.get('brepSha256'): return exact_target()
            return target
        except ImportError:
            # A producer backend may expose OCCT hierarchy paths while the
            # reader provides native TopologicCore only. Restore only an
            # unambiguous exact signature; never guess coincident subshapes.
            kind = str(locator.get("type", "")).lower()
            expected = locator.get("signature", {}).get("hash")
            if kind not in _SUBTOPOLOGY_TYPES or not (expected or locator.get('exactGeometry')):
                return None
            lookup = TPYCodec._LocatorIndex(topology, kind, cache)
            matches = TPYCodec._LocatorMatches(lookup, 'exact' if locator.get('exactGeometry') else 'signature', locator.get('exactGeometry') or expected)
            return matches[0] if len(matches) == 1 else None
        except (AttributeError, IndexError, TypeError, ValueError): return exact_target()
    kind = str(locator.get("type", "")).lower()
    if kind not in _SUBTOPOLOGY_TYPES:
        return None
    lookup = TPYCodec._LocatorIndex(topology, kind, cache)
    candidates = lookup['candidates']
    index = locator.get("index", -1)
    index = index if type(index) is int else -1
    # The stored occurrence wins when it validates, exactly as choose() below.
    # Avoid serializing every candidate just to rediscover a valid occurrence.
    if 0 <= index < len(candidates):
        candidate = candidates[index]
        checksum = locator.get('brepSha256')
        signature = locator.get('signature', {}).get('hash')
        if checksum and TPYCodec._LocatorValue(lookup, 'brep', candidate) == checksum:
            return candidate
        if not checksum and signature and TPYCodec._LocatorValue(lookup, 'signature', candidate) == signature:
            return candidate
    def choose(matches):
        # Equal geometry does not imply equal native identity. Prefer the
        # stored occurrence when several distinct coincident shapes match.
        if 0 <= index < len(candidates) and any(candidates[index] is item for item in matches):
            return candidates[index]
        return matches[0] if len(matches) == 1 else None
    checksum = locator.get("brepSha256")
    if checksum:
        matches = TPYCodec._LocatorMatches(lookup, 'brep', checksum)
        if matches:
            return choose(matches)
    exact = locator.get("exactGeometry")
    if exact:
        matches = TPYCodec._LocatorMatches(lookup, 'exact', exact)
        return matches[0] if len(matches) == 1 else None
    signature = locator.get("signature", {}).get("hash")
    if signature:
        matches = TPYCodec._LocatorMatches(lookup, 'signature', signature)
        if matches:
            return choose(matches)
    return candidates[index] if not checksum and not signature and 0 <= index < len(candidates) else None


def _raw_contexts(subject):
    from topologicpy.Core import Core

    result = []

    try:
        value = Core.InstanceCall(
            subject,
            "Contexts",
            result,
        )

        if isinstance(value, list):
            result.extend(value)

    except Exception:
        try:
            value = Core.InstanceCall(
                subject,
                "Contexts",
            )

            if isinstance(value, list):
                result.extend(value)

        except Exception:
            pass

    unique = []

    for context in result:
        if context is None:
            continue

        if any(
            context is previous
            for previous in unique
        ):
            continue

        unique.append(context)

    return unique


def _context_host(context):
    from topologicpy.Context import Context

    try:
        return Context.Topology(
            context
        )
    except Exception:
        return None


def _context_axis_value(
    context,
    names,
    default,
):
    from topologicpy.Core import Core

    for name in names:
        try:
            value = getattr(
                context,
                name,
            )

            if callable(value):
                value = value()

            value = float(value)

            if math.isfinite(value):
                return value

        except Exception:
            pass

        try:
            value = Core.InstanceCall(
                context,
                name,
            )

            value = float(value)

            if math.isfinite(value):
                return value

        except Exception:
            pass

    return float(default)


def _context_parameters(
    subject,
    host,
    explicit_contexts=None,
):
    contexts = (
        list(explicit_contexts)
        if explicit_contexts
        else _raw_contexts(subject)
    )

    selected = None

    for context in contexts:
        context_host = _context_host(
            context
        )

        if (
            context_host is not None
            and _same_topology(
                context_host,
                host,
            )
        ):
            selected = context
            break

    if (
        selected is None
        and len(contexts) > 0
    ):
        selected = contexts[0]

    if selected is None:
        return [
            0.5,
            0.5,
            0.5,
        ]

    return [
        _context_axis_value(
            selected,
            ("x", "u", "U"),
            0.5,
        ),
        _context_axis_value(
            selected,
            ("y", "v", "V"),
            0.5,
        ),
        _context_axis_value(
            selected,
            ("z", "w", "W"),
            0.5,
        ),
    ]


def _raw_apertures(host):
    """
    Return (aperture-wrapper, aperture-topology, contexts) triples.

    This intentionally uses the backend aperture collection rather than the
    public Topology.Apertures convenience function. Content topologies tagged
    with dictionary type="Aperture" are already preserved as Contents.
    """
    from topologicpy.Aperture import Aperture
    from topologicpy.Core import Core
    from topologicpy.Topology import Topology

    raw = []

    try:
        value = Core.InstanceCall(
            host,
            "Apertures",
            raw,
        )

        if isinstance(value, list):
            raw.extend(value)

    except Exception:
        try:
            value = Core.InstanceCall(
                host,
                "Apertures",
            )

            if isinstance(value, list):
                raw.extend(value)

        except Exception:
            pass

    result = []

    for wrapper in raw:
        aperture_topology = None

        try:
            aperture_topology = (
                Aperture.Topology(
                    wrapper
                )
            )
        except Exception:
            aperture_topology = None

        if not Topology.IsInstance(
            aperture_topology,
            "Topology",
        ):
            continue

        result.append(
            (
                wrapper,
                aperture_topology,
                _raw_contexts(
                    wrapper
                ),
            )
        )

    return result


class _ArchiveBuilder:
    def __init__(
        self,
        include_subtopology_dictionaries=True,
        include_contents=True,
        include_apertures=True,
    ):
        self.include_subtopology_dictionaries = bool(
            include_subtopology_dictionaries
        )
        self.include_contents = bool(
            include_contents
        )
        self.include_apertures = bool(
            include_apertures
        )

        self.records = {}
        self.geometry = {}
        self.relations = []

        self._object_by_python_id = {}
        self._object_references = {}
        self._active_geometry = {}

    def _next_id(self):
        return (
            f"obj_{len(self.records):06d}"
        )

    def register(
        self,
        topology,
        ancestry=None,
    ):
        from topologicpy.Topology import Topology
        from topologicpy.Core import Core

        if not Topology.IsInstance(
            topology,
            "Topology",
        ):
            return None

        python_id = id(topology)

        if python_id in self._object_by_python_id:
            return self._object_by_python_id[
                python_id
            ]

        brep, brep_hash = _brep_data(
            topology
        )

        if brep is None:
            return None

        ancestry = dict(
            ancestry or {}
        )

        # Prevent recursive Content cycles even when the backend returns a new
        # Python wrapper for the same underlying topology.
        if brep_hash in ancestry:
            return ancestry[
                brep_hash
            ]

        object_id = self._next_id()

        self._object_by_python_id[
            python_id
        ] = object_id
        # Keep wrappers alive while their Python ids are used as cache keys.
        # Otherwise temporary aperture wrappers can be collected and their ids
        # reused for an unrelated opening later in the archive.
        self._object_references[python_id] = topology

        ancestry[
            brep_hash
        ] = object_id

        geometry_path = (
            f"geometry/{object_id}.brep"
        )

        self.geometry[
            geometry_path
        ] = brep

        type_name = Topology.TypeAsString(
            topology,
            silent=True,
        )

        record = {
            "id": object_id,
            "type": type_name,
            "signature": {"type":type_name,"nativeBrepSha256":brep_hash} if type_name=='Cluster' and type(Core.Backend()).__module__.startswith('topologicpy.pythonocc_backend') else _signature(topology),
            "geometry": {
                "path": geometry_path,
                "sha256": brep_hash,
                "encoding": "utf-8",
                "format": "BREP",
            },
            "dictionary": _encode_value(
                _python_dictionary(
                    topology
                )
            ),
            "subtopologies": [],
        }

        self.records[
            object_id
        ] = record

        hosts = [
            (
                None,
                topology,
            )
        ]

        native_paths = TPYCodec._NativePaths(topology)
        cached_contents={};cached_apertures={}
        for (
            sub_type,
            index,
            subtopology,
        ) in _subtopologies(topology):
            dictionary=_python_dictionary(subtopology) if self.include_subtopology_dictionaries else {}
            try:contents=Topology.Contents(subtopology,silent=True) or [] if self.include_contents else []
            except Exception:contents=[]
            apertures=_raw_apertures(subtopology) if self.include_apertures else []
            if not dictionary and not contents and not apertures:continue
            cached_contents[id(subtopology)]=contents;cached_apertures[id(subtopology)]=apertures
            native = Topology.OCCTShape(subtopology, silent=True)
            paths = [path for shape, path in native_paths.get(hash(native), []) if shape.IsSame(native)] if native is not None else []
            locator = _make_locator(
                sub_type,
                index,
                subtopology,
                native_path=paths[0] if len(paths)==1 else None,
            )
            hosts.append(
                (
                    locator,
                    subtopology,
                )
            )

            if (
                self.include_subtopology_dictionaries
            ):
                if len(dictionary) > 0:
                    record[
                        "subtopologies"
                    ].append(
                        {
                            "locator": locator,
                            "dictionary": (
                                _encode_value(
                                    dictionary
                                )
                            ),
                        }
                    )

        for host_locator, host in hosts:
            if self.include_contents:
                try:
                    contents = cached_contents[id(host)] if id(host) in cached_contents else Topology.Contents(host,silent=True) or []
                except Exception:
                    contents = []

                relation_seen = set()

                for content in contents:
                    # TopologicCore includes Aperture wrappers in Contents.
                    # Persist their underlying geometry through the aperture
                    # relation below, rather than a non-geometric object type.
                    if Topology.IsInstance(content, "Aperture"):
                        continue
                    if not Topology.IsInstance(
                        content,
                        "Topology",
                    ):
                        continue

                    child_id = self.register(
                        content,
                        ancestry=ancestry,
                    )

                    if child_id is None:
                        continue

                    key = (
                        "content",
                        child_id,
                    )

                    if key in relation_seen:
                        continue

                    relation_seen.add(key)

                    self.relations.append(
                        {
                            "kind": "content",
                            "hostObject": object_id,
                            "hostLocator": host_locator,
                            "childObject": child_id,
                            "context": (
                                _context_parameters(
                                    content,
                                    host,
                                )
                            ),
                        }
                    )

            if self.include_apertures:
                aperture_seen = set()

                for (
                    wrapper,
                    aperture_topology,
                    contexts,
                ) in (cached_apertures[id(host)] if id(host) in cached_apertures else _raw_apertures(host)):
                    child_id = self.register(
                        aperture_topology,
                        ancestry=ancestry,
                    )

                    if child_id is None:
                        continue

                    key = (
                        child_id,
                        _sha256_text(
                            json.dumps(
                                _context_parameters(
                                    wrapper,
                                    host,
                                    explicit_contexts=contexts,
                                ),
                                separators=(",", ":"),
                            )
                        ),
                    )

                    if key in aperture_seen:
                        continue

                    aperture_seen.add(key)

                    self.relations.append(
                        {
                            "kind": "aperture",
                            "hostObject": object_id,
                            "hostLocator": host_locator,
                            "childObject": child_id,
                            "context": (
                                _context_parameters(
                                    wrapper,
                                    host,
                                    explicit_contexts=contexts,
                                )
                            ),
                        }
                    )

        return object_id


class TPYCodec:
    """Versioned native TopologicPy archive codec."""

    @staticmethod
    def _IdentityBucket(topology):
        # OCCT hashes ignore orientation, just like IsSame. A hash is only a
        # bucket: every collision is still resolved by exact native identity.
        from topologicpy.Topology import Topology
        try:
            native = Topology.OCCTShape(topology, silent=True)
            if native is None or type(native).__module__.split('.')[0] not in ('OCC', 'OCP'):
                return None
            return hash(native)
        except Exception:
            return None

    @staticmethod
    def _UniqueNative(items, root):
        from topologicpy.Topology import Topology
        unique, buckets, unkeyed = [], {}, []
        for item in items:
            if not Topology.IsInstance(item, 'Topology') or _same_topology(item, root):
                continue
            key = TPYCodec._IdentityBucket(item)
            previous = unique if key is None else buckets.get(key, []) + unkeyed
            if any(_same_topology(item, old) for old in previous):
                continue
            unique.append(item)
            if key is None:
                unkeyed.append(item)
            else:
                buckets.setdefault(key, []).append(item)
        return unique

    @staticmethod
    def _LocatorIndex(topology, kind, cache=None):
        """Load-local candidates in their original native occurrence order."""
        from topologicpy.Topology import Topology
        key = (id(topology), kind)
        if cache is not None and key in cache:
            return cache[key]
        candidates = TPYCodec._UniqueNative(
            Topology.SubTopologies(topology, subTopologyType=kind, silent=True) or [], topology)
        # Retain the root object, so Python cannot reuse its id during the load.
        result = {'root': topology, 'candidates': candidates}
        if cache is not None:
            cache[key] = result
        return result

    @staticmethod
    def _LocatorValue(lookup, kind, item):
        values = lookup.setdefault(kind + '_values', {})
        key = id(item)
        if key not in values:
            values[key] = _candidate_brep_hash(item) if kind == 'brep' else _exact_geometry_key(item) if kind == 'exact' else _signature(item).get('hash')
        return values[key]

    @staticmethod
    def _LocatorMatches(lookup, kind, value):
        """Index each checksum/signature once; collisions retain all matches."""
        if kind not in lookup:
            values = {}
            for item in lookup['candidates']:
                identifier = TPYCodec._LocatorValue(lookup, kind, item)
                values.setdefault(identifier, []).append(item)
            lookup[kind] = values
        return lookup[kind].get(value, [])

    @staticmethod
    def _CanReadMetadataManifest(manifest):
        if manifest.get('formatVersion') not in (1, 2) or manifest.get('relations'):
            return False
        objects = manifest.get('objects', [])
        if any(record.get('type') in ('TGraph', 'Collection') for record in objects):
            return False
        def references(value):
            if isinstance(value, dict):
                if value.get('__tpy_type__') == 'reference':
                    return True
                return any(references(child) for child in value.values())
            return isinstance(value, (list, tuple)) and any(references(child) for child in value)
        return not any(references(record.get('dictionary', {})) or
                       references(record.get('subtopologies', [])) for record in objects)

    @staticmethod
    def _ReadMetadataManifest(manifest, loaded):
        """Read actual restored v1 metadata without rewriting native BREPs.

        Geometry bytes/checksums have already been validated by load(). This
        path is deliberately limited to native archives without relationships;
        callers retain full archive verification for richer archive variants.
        Every dictionary-bearing constituent is enumerated, including metadata
        absent from the source manifest, so additions/losses are both detected.
        """
        from topologicpy.Topology import Topology
        if not TPYCodec._CanReadMetadataManifest(manifest):
            raise ValueError('This archive requires full relationship verification.')
        objects = manifest.get('objects', [])
        if len(loaded) != len(objects):
            raise ValueError('Not all native objects were restored.')
        records = []
        cache = {}
        for source in objects:
            topology = loaded.get(source.get('id'))
            if topology is None:
                raise ValueError('Native object was not restored.')
            record = dict(source)
            record['type'] = Topology.TypeAsString(topology, silent=True)
            record['dictionary'] = _encode_value(_python_dictionary(topology))
            record['signature'] = {'type':record['type'],'nativeBrepSha256':_candidate_brep_hash(topology)} if 'nativeBrepSha256' in source.get('signature',{}) else _signature(topology)
            records.append(record)
            metadata = []
            for kind in _SUBTOPOLOGY_TYPES:
                lookup = TPYCodec._LocatorIndex(topology, kind, cache)
                for index, shape in enumerate(lookup['candidates']):
                    dictionary = _python_dictionary(shape)
                    if not dictionary:
                        continue
                    metadata.append({'locator': {'type': kind, 'index': index,
                                     'signature': _signature(shape)},
                                     'dictionary': _encode_value(dictionary)})
            record['subtopologies'] = metadata
        result = dict(manifest)
        result['objects'] = records
        return result

    @staticmethod
    def _EncodeValue(value, reference=None, active=None, depth=0):
        """Explicit data encodings only; never textualize unsupported objects."""
        if depth > 200:
            raise ValueError("TPY metadata nesting exceeds 200 levels.")
        if reference is not None:
            result = reference(value)
            if result is not None:
                return result
        active = set() if active is None else active
        encode = lambda item: TPYCodec._EncodeValue(item, reference, active, depth + 1)
        if type(value).__module__.split('.')[0] == "numpy":
            import numpy as np
            if isinstance(value, np.ndarray):
                if value.dtype.fields is not None:
                    raise TypeError("Structured NumPy dtypes need an explicit TPY encoding.")
                return {"__tpy_type__": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape), "items": encode(value.ravel().tolist())}
            if isinstance(value, np.generic):
                return {"__tpy_type__": "numpy_scalar", "dtype": str(value.dtype), "value": encode(value.item())}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"__tpy_type__": "float", "value": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            identity = id(value)
            if identity in active:
                raise ValueError("Cyclic Python containers require an explicit object encoding; TPY graph/topology reference cycles are supported.")
            active.add(identity)
            try:
                if isinstance(value, dict):
                    return {"__tpy_type__": "mapping", "items": [[encode(k), encode(v)] for k, v in value.items()]}
                items = [encode(item) for item in value]
                if isinstance(value, list):
                    return items
                if isinstance(value, (set, frozenset)):
                    items.sort(key=lambda item: json.dumps(item, sort_keys=True))
                return {"__tpy_type__": type(value).__name__, "items": items}
            finally:
                active.remove(identity)
        if isinstance(value, bytes):
            return {"__tpy_type__": "bytes", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, complex):
            return {"__tpy_type__": "complex", "real": encode(value.real), "imag": encode(value.imag)}
        from datetime import date, datetime, time
        from decimal import Decimal
        from fractions import Fraction
        from uuid import UUID
        if isinstance(value, (datetime, date, time)):
            return {"__tpy_type__": type(value).__name__, "value": value.isoformat(), "fold": getattr(value, "fold", 0), "zone": getattr(getattr(value, "tzinfo", None), "key", None)}
        if isinstance(value, (Decimal, UUID)):
            return {"__tpy_type__": type(value).__name__, "value": str(value)}
        if isinstance(value, Fraction):
            return {"__tpy_type__": "Fraction", "numerator": value.numerator, "denominator": value.denominator}
        raise TypeError("Unsupported TPY value type: " + type(value).__module__ + "." + type(value).__name__)

    @staticmethod
    def _DecodeValue(value, reference=None, legacy=False, depth=0):
        if depth > 200:
            raise ValueError("TPY metadata nesting exceeds 200 levels.")
        decode = lambda item: TPYCodec._DecodeValue(item, reference, legacy, depth + 1)
        if isinstance(value, list):
            return [decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        tag = value.get("__tpy_type__")
        if tag == "reference":
            if reference is None:
                raise ValueError("Object reference outside a TPY object archive.")
            return reference(value)
        if tag == "mapping":
            result = {}
            for pair in value.get("items", []):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Invalid TPY mapping entry.")
                key, item = decode(pair[0]), decode(pair[1])
                if key in result:
                    raise ValueError("Duplicate TPY mapping key.")
                result[key] = item
            return result
        if tag == "float":
            if value.get("value") not in ("nan", "+inf", "-inf"):
                raise ValueError("Invalid tagged TPY float.")
            return float(value["value"])
        if tag == "bytes":
            return base64.b64decode(value["data"], validate=True)
        if tag in ("tuple", "set", "frozenset"):
            return {"tuple": tuple, "set": set, "frozenset": frozenset}[tag](decode(value["items"]))
        if tag == "complex":
            return complex(decode(value["real"]), decode(value["imag"]))
        if tag in ("date", "datetime", "time"):
            import datetime
            result = getattr(datetime, tag).fromisoformat(value["value"])
            if tag != "date":
                if value.get("zone") is not None:
                    from zoneinfo import ZoneInfo
                    zone = ZoneInfo(value["zone"])
                    result = result.astimezone(zone) if tag == "datetime" and result.tzinfo is not None else result.replace(tzinfo=zone)
                result = result.replace(fold=value.get("fold", 0))
            return result
        if tag == "Decimal":
            from decimal import Decimal
            return Decimal(value["value"])
        if tag == "UUID":
            from uuid import UUID
            return UUID(value["value"])
        if tag == "Fraction":
            from fractions import Fraction
            return Fraction(value["numerator"], value["denominator"])
        if tag in ("ndarray", "numpy_scalar"):
            import numpy as np
            dtype = np.dtype(value["dtype"])
            if dtype.fields is not None:
                raise ValueError("Unsupported structured TPY dtype.")
            if tag == "numpy_scalar":
                return np.array(decode(value["value"]), dtype=dtype)[()]
            shape = value["shape"]
            items = decode(value["items"])
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape) or math.prod(shape) != len(items):
                raise ValueError("Invalid TPY array shape.")
            result = np.empty(len(items), dtype=dtype)
            result[:] = items
            return result.reshape(shape)
        if tag == "repr" and legacy:
            return value.get("value", "")
        if tag is not None and not legacy:
            raise ValueError("Unsupported TPY value tag: " + str(tag))
        return {key: decode(item) for key, item in value.items()}

    @staticmethod
    def _IsGraphOrCollection(value):
        from topologicpy.TGraph import TGraph
        return isinstance(value, (TGraph, list, tuple, dict))

    @staticmethod
    def _NativeChildren(shape):
        try:
            from OCC.Core.TopoDS import TopoDS_Iterator
        except ImportError:
            from OCP.TopoDS import TopoDS_Iterator
        iterator = TopoDS_Iterator(shape)
        children = []
        while iterator.More():
            children.append(iterator.Value());iterator.Next()
        return children

    @staticmethod
    def _NativePaths(topology):
        from topologicpy.Topology import Topology
        root = Topology.OCCTShape(topology, silent=True)
        result, seen = {}, {}
        def visit(shape, path):
            bucket = seen.setdefault(hash(shape), [])
            if any(shape.IsSame(old) for old in bucket): return
            bucket.append(shape)
            result.setdefault(hash(shape), []).append((shape, path))
            for index, child in enumerate(TPYCodec._NativeChildren(shape)):
                visit(child, path + [index])
        try:
            if root is not None: visit(root, [])
        except (ImportError, TypeError, AttributeError):
            return {}
        return result

    @staticmethod
    def _PortableNativeReferences(natives, builder):
        """Use public topology identity when native OCCT traversal is opaque.

        TopologicCore exposes TopoDS_Shape as an opaque binding. Its public
        topology traversal still preserves native identity and verified BREP
        locators; PythonOCC continues to use exact native hierarchy paths.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster
        roots, included = [], []
        for topology in natives:
            if any(_same_topology(topology, old) for old in included): continue
            roots.append(topology);included.append(topology)
            included.extend(child for kind, ordinal, child in _subtopologies(topology))
        carrier = roots[0] if len(roots) == 1 else Cluster.ByTopologies(roots, silent=True)
        object_id = builder.register(carrier)
        if object_id is None: raise ValueError('Could not serialize the native geometry container.')
        root_ref = {'__tpy_type__': 'reference', 'object': object_id}
        references = [(carrier, root_ref)]
        for kind, ordinal, child in _subtopologies(carrier):
            references.append((child, dict(root_ref, locator=_make_locator(kind, ordinal, child))))
        result = {}
        for topology in natives:
            matches = [ref for old, ref in references if _same_topology(topology, old)]
            if len(matches) != 1: raise ValueError('Native topology identity could not be resolved in its container.')
            result[id(topology)] = TPYCodec._OrientedReference(topology, matches[0])
        return result

    @staticmethod
    def _OrientedReference(topology, reference):
        from topologicpy.Topology import Topology
        result = dict(reference)
        try:
            shape = Topology.OCCTShape(topology, silent=True)
            if shape is not None: result['orientation'] = int(shape.Orientation())
        except (AttributeError, TypeError):
            pass
        return result

    @staticmethod
    def _SaveGraphArchive(value, path, overwrite=False, silent=False, **options):
        """Build a mixed object archive, with native subshape references."""
        from topologicpy.Topology import Topology
        from topologicpy.TGraph import TGraph
        try:
            output = Path(os.fspath(path))
            if output.suffix.lower() != '.tpy': output = Path(str(output) + '.tpy')
            if output.exists() and not overwrite: raise ValueError("Output exists and overwrite is False.")
            builder = _ArchiveBuilder(options.get('includeSubtopologyDictionaries', True), options.get('includeContents', True), options.get('includeApertures', True))
            natives, seen, states = [], set(), {}
            def collect(item):
                if id(item) in seen: return
                seen.add(id(item))
                if isinstance(item, TGraph):
                    states[id(item)] = TGraph._TPYState(item)
                    collect(states[id(item)])
                elif Topology.IsInstance(item, 'Topology'): natives.append(item)
                elif isinstance(item, dict):
                    for key, child in item.items(): collect(key);collect(child)
                elif isinstance(item, (list, tuple, set, frozenset)):
                    for child in item: collect(child)
                elif type(item).__module__.split('.')[0] == 'numpy' and getattr(item, 'dtype', None) is not None and item.dtype.kind == 'O':
                    for child in item.flat: collect(child)
            collect(value)
            # Register containing shapes first so references to their faces,
            # edges or vertices resolve to subshapes of that same loaded object.
            natives.sort(key=lambda item: Topology.Type(item), reverse=True)
            native_refs = {}
            opaque_shapes = any(not hasattr(Topology.OCCTShape(item, silent=True), 'IsSame') for item in natives)
            if opaque_shapes:
                native_refs = TPYCodec._PortableNativeReferences(natives, builder)
            else:
                roots, included = [], {}
                for item in natives:
                    shape = Topology.OCCTShape(item, silent=True)
                    if shape is None: raise ValueError('Native geometry has no OCCT shape.')
                    if any(shape.IsSame(old) for old in included.get(hash(shape), [])): continue
                    roots.append(item)
                    for key, entries in TPYCodec._NativePaths(item).items():
                        included.setdefault(key, []).extend(native for native, path in entries)
                if roots:
                    from topologicpy.Cluster import Cluster
                    carrier = roots[0] if len(roots) == 1 else Cluster.ByTopologies(roots, silent=True)
                    native_paths = TPYCodec._NativePaths(carrier)
                    object_id = builder.register(carrier)
                    if object_id is None: raise ValueError('Could not serialize the native geometry container.')
                    root_shape = Topology.OCCTShape(carrier, silent=True)
                    for item in natives:
                        native = Topology.OCCTShape(item, silent=True)
                        paths = [path for shape, path in native_paths.get(hash(native), []) if shape.IsSame(native)]
                        if len(paths) != 1: raise ValueError('Could not identify an exact native subshape path.')
                        ref = {'__tpy_type__': 'reference', 'object': object_id}
                        if not native.IsSame(root_shape):
                            ref['locator'] = dict(_make_locator(Topology.TypeAsString(item).lower(), 0, item), nativePath=paths[0])
                        native_refs[id(item)] = TPYCodec._OrientedReference(item, ref)
            graph_ids = {}
            def reference(item):
                if isinstance(item, TGraph):
                    if id(item) not in graph_ids:
                        object_id = builder._next_id();graph_ids[id(item)] = object_id
                        record = {'id': object_id, 'type': 'TGraph'};builder.records[object_id] = record
                        record['state'] = TPYCodec._EncodeValue(states[id(item)], reference)
                    return {'__tpy_type__': 'reference', 'object': graph_ids[id(item)]}
                if Topology.IsInstance(item, 'Topology'):
                    if id(item) not in native_refs: raise ValueError("Unregistered native geometry reference.")
                    return native_refs[id(item)]
                return None
            if isinstance(value, TGraph):
                root = reference(value)['object']
            else:
                root = builder._next_id();record = {'id': root, 'type': 'Collection'};builder.records[root] = record
                record['value'] = TPYCodec._EncodeValue(value, reference)
            manifest = {'format': _FORMAT, 'formatVersion': 2, 'rootObject': root,
                        'options': {'includeSubtopologyDictionaries': builder.include_subtopology_dictionaries,
                                    'includeContents': builder.include_contents, 'includeApertures': builder.include_apertures,
                                    'tolerance': float(options.get('tolerance', .0001))},
                        'objects': list(builder.records.values()), 'relations': builder.relations}
            return TPYCodec._WriteGraphArchive(output, manifest, builder.geometry)
        except Exception as exc:
            if not silent: print('TPYCodec.save - Error: ' + str(exc) + '. Returning False.')
            return False

    @staticmethod
    def _WriteGraphArchive(output, manifest, geometry):
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = None
        try:
            text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name+'.', suffix='.tmp', delete=False) as handle: pending = Path(handle.name)
            with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                archive.writestr(_MANIFEST_NAME, text)
                for name, brep in geometry.items(): archive.writestr(name, brep)
            os.replace(pending, output)
            return True
        finally:
            if pending is not None: pending.unlink(missing_ok=True)

    @staticmethod
    def _ValidateGraphManifest(manifest, names):
        """Validate object IDs and references before constructing native objects."""
        records = manifest.get('objects')
        if not isinstance(records, list) or not 1 <= len(records) <= 100000: raise ValueError('Invalid TPY object table.')
        by_id = {}
        native_types = {'Vertex','Edge','Wire','Face','Shell','Cell','CellComplex','Cluster'}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get('id'), str) or record['id'] in by_id: raise ValueError('Invalid or duplicate TPY object ID.')
            if record.get('type') not in native_types | {'TGraph','Collection'}: raise ValueError('Unsupported TPY object type.')
            by_id[record['id']] = record
            if record['type'] in ('TGraph', 'Collection') and 'geometry' in record:
                raise ValueError('Graph/collection records cannot declare BREP payloads.')
        if manifest.get('rootObject') not in by_id: raise ValueError('Missing TPY root object.')
        used = {_MANIFEST_NAME}
        for record in records:
            if record['type'] in native_types:
                geometry = record.get('geometry', {})
                name = geometry.get('path')
                if not isinstance(name, str) or name not in names or not name.startswith('geometry/') or not name.endswith('.brep') or '\\' in name or '..' in name.split('/'):
                    raise ValueError('Invalid native geometry member.')
                if not isinstance(geometry.get('sha256'), str) or len(geometry['sha256']) != 64: raise ValueError('Missing native geometry checksum.')
                used.add(name)
            def visit(item, depth=0):
                if depth > 200: raise ValueError('TPY reference nesting exceeds 200 levels.')
                if isinstance(item, dict):
                    if item.get('__tpy_type__') == 'reference':
                        if set(item)-{'__tpy_type__','object','locator','orientation'} or item.get('object') not in by_id: raise ValueError('Invalid TPY object reference.')
                        if 'orientation' in item and (type(item['orientation']) is not int or item['orientation'] not in (0,1,2,3)): raise ValueError('Invalid native reference orientation.')
                        if ('orientation' in item or item.get('locator') is not None) and by_id[item['object']]['type'] not in native_types: raise ValueError('Subshape/orientation reference requires a native topology.')
                        if item.get('locator') is not None:
                            loc = item['locator']
                            if not isinstance(loc, dict) or loc.get('type') not in _SUBTOPOLOGY_TYPES or type(loc.get('index')) is not int or loc['index'] < 0: raise ValueError('Invalid native subshape locator.')
                            if 'nativePath' in loc and (not isinstance(loc['nativePath'], list) or len(loc['nativePath']) > 100 or any(type(index) is not int or index < 0 for index in loc['nativePath'])): raise ValueError('Invalid native subshape path.')
                    if item.get('__tpy_type__') == 'repr': raise ValueError('Text-only object encoding is not supported by TPY v2.')
                    for child in item.values(): visit(child, depth+1)
                elif isinstance(item, list):
                    for child in item: visit(child, depth+1)
            visit(record)
        if used != set(names): raise ValueError('Unreferenced or unexpected TPY archive members.')
        for relation in manifest.get('relations', []):
            if not isinstance(relation, dict) or relation.get('kind') not in ('content','aperture') or relation.get('hostObject') not in by_id or relation.get('childObject') not in by_id:
                raise ValueError('Invalid native TPY relationship.')
            if any(by_id[relation[key]]['type'] not in native_types for key in ('hostObject','childObject')): raise ValueError('Native relationships require native topology objects.')
        return by_id

    @staticmethod
    def _LoadGraphArchive(archive, manifest, tolerance, silent, _loaded=None):
        from topologicpy.TGraph import TGraph
        by_id = TPYCodec._ValidateGraphManifest(manifest, archive.namelist())
        loaded = {}
        natives = [record for record in by_id.values() if 'geometry' in record]
        if natives:
            native_manifest = dict(manifest, formatVersion=1, objects=natives, rootObject=natives[0]['id'])
            with tempfile.TemporaryDirectory(prefix='topologicpy-tpy-native-') as directory:
                path = Path(directory)/'native.tpy'
                geometry = {record['geometry']['path']: archive.read(record['geometry']['path']) for record in natives}
                TPYCodec._WriteGraphArchive(path, native_manifest, geometry)
                if TPYCodec.load(path, tolerance=tolerance, silent=silent, _loaded=loaded) is None: raise ValueError('Native objects could not be restored.')
        for identifier, record in by_id.items():
            if record['type'] == 'TGraph': loaded[identifier] = TGraph()
        resolved_refs = {}
        locator_cache = {}
        def reference(item):
            identifier = item['object']
            if identifier not in loaded: raise ValueError('Unresolved TPY object reference.')
            locator = item.get('locator')
            key = (identifier, json.dumps(locator, sort_keys=True), item.get('orientation'))
            if key not in resolved_refs:
                target = loaded[identifier] if locator is None else _resolve_locator(loaded[identifier], locator, cache=locator_cache)
                if target is None: raise ValueError('Native subshape reference could not be restored.')
                if item.get('orientation') is not None:
                    from topologicpy.Topology import Topology
                    native = Topology.OCCTShape(target, silent=True)
                    if native is None: raise ValueError('This backend cannot restore native orientation references.')
                    if int(native.Orientation()) != item['orientation']:
                        target = Topology.ByOCCTShape(native.Oriented(type(native.Orientation())(item['orientation'])), silent=True)
                        if target is None: raise ValueError('Could not restore native reference orientation.')
                resolved_refs[key] = target
            return resolved_refs[key]
        for identifier, record in by_id.items():
            if record['type'] == 'TGraph':
                state = TPYCodec._DecodeValue(record.get('state'), reference)
                TGraph._RestoreTPYState(loaded[identifier], state)
        for identifier, record in by_id.items():
            if record['type'] == 'Collection': loaded[identifier] = TPYCodec._DecodeValue(record['value'], reference)
        if _loaded is not None:
            _loaded.update(loaded)
        return loaded[manifest['rootObject']]

    @staticmethod
    def _MetadataFingerprint(manifest, include_geometry_signatures=True):
        """Canonical object/reference semantics, independent of archive IDs.

        BREP checksums protect exact payload bytes separately. Geometric
        signatures are used here because BREP writers may reorder shape tables.
        """
        by_id = {record['id']: record for record in manifest['objects']}
        assigned, records, subshapes = {}, [], {}
        dumps = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        def geometric_signature(value):
            # Backend enumeration may repeat the same native vertex before a
            # BREP round-trip. Compare geometry independently of wrapper counts.
            if not isinstance(value, dict): return None
            result = {key: item for key, item in value.items() if key not in ('hash', 'counts', 'vertexCount')}
            if 'vertices' in result:
                result['vertices'] = sorted({tuple(point) for point in result['vertices']})
            def canonical(item):
                if isinstance(item, float) and item == 0: return 0.0
                if isinstance(item, (list, tuple)): return [canonical(child) for child in item]
                if isinstance(item, dict): return {key: canonical(child) for key, child in item.items()}
                return item
            return canonical(result)
        def locator(value):
            if value is None: return None
            return {'type': value.get('type'), 'signature': value.get('signature', {}).get('hash')}
        def metadata(value):
            return normalize(TPYCodec._EncodeValue(TPYCodec._DecodeValue(value, legacy=True)))
        def shallow(identifier):
            record = by_id[identifier]
            return dumps({'type': record['type'], 'dictionary': record.get('dictionary'),
                          'signature': geometric_signature(record.get('signature')) if include_geometry_signatures else None})
        def normalize(value):
            if isinstance(value, list): return [normalize(item) for item in value]
            if not isinstance(value, dict): return value
            if value.get('__tpy_type__') == 'reference':
                result = {'object': resolve(value['object'])}
                if value.get('locator') is not None:
                    result['locator'] = locator(value['locator'])
                    identity = (value['object'], value['locator'].get('type'), tuple(value['locator']['nativePath']) if 'nativePath' in value['locator'] else value['locator'].get('index'))
                    if identity not in subshapes: subshapes[identity] = len(subshapes)
                    result['subshape'] = subshapes[identity]
                if 'orientation' in value: result['orientation'] = value['orientation']
                return result
            if value.get('__tpy_type__') == 'mapping':
                return {'__tpy_type__': 'mapping', 'items': [[normalize(k), normalize(v)] for k, v in sorted(value['items'], key=lambda pair: dumps(pair[0]))]}
            return {key: normalize(item) for key, item in value.items()}
        def resolve(identifier):
            if identifier in assigned: return assigned[identifier]
            number = len(records);assigned[identifier] = number;records.append(None)
            record = by_id[identifier];kind = record['type']
            result = {'type': kind}
            if kind == 'TGraph': result['state'] = normalize(record['state'])
            elif kind == 'Collection': result['value'] = normalize(record['value'])
            else:
                result['dictionary'] = metadata(record.get('dictionary', {}))
                result['subtopologies'] = sorted([{'locator': locator(item['locator']), 'dictionary': metadata(item.get('dictionary', {}))} for item in record.get('subtopologies', [])], key=dumps)
                if include_geometry_signatures: result['signature'] = geometric_signature(record.get('signature'))
                relations = [edge for edge in manifest.get('relations', []) if edge['hostObject'] == identifier]
                relations.sort(key=lambda edge: (edge['kind'], dumps(locator(edge.get('hostLocator'))), shallow(edge['childObject']), dumps(edge.get('context'))))
                result['relations'] = [{'kind': edge['kind'], 'locator': locator(edge.get('hostLocator')), 'child': resolve(edge['childObject']), 'context': edge.get('context')} for edge in relations]
            records[number] = result
            return number
        root = resolve(manifest['rootObject'])
        for identifier in sorted(by_id, key=shallow): resolve(identifier)
        return dumps({'root': root, 'objects': records})

    @staticmethod
    def save(
        topology,
        path,
        overwrite=False,
        includeSubtopologyDictionaries=True,
        includeContents=True,
        includeApertures=True,
        tolerance=0.0001,
        silent=False,
    ):
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology") and TPYCodec._IsGraphOrCollection(topology):
            return TPYCodec._SaveGraphArchive(topology, path, overwrite=overwrite, silent=silent,
                includeSubtopologyDictionaries=includeSubtopologyDictionaries,
                includeContents=includeContents, includeApertures=includeApertures, tolerance=tolerance)

        if not Topology.IsInstance(
            topology,
            "Topology",
        ):
            if not silent:
                print(
                    "TPYCodec.save - Error: The input topology is invalid. "
                    "Returning False."
                )
            return False

        try:
            tolerance = abs(
                float(tolerance)
            )
        except Exception:
            tolerance = 0.0001

        if tolerance <= 0.0:
            tolerance = 0.0001

        try:
            output_path = Path(
                os.fspath(path)
            )
        except Exception:
            if not silent:
                print(
                    "TPYCodec.save - Error: The output path is invalid. "
                    "Returning False."
                )
            return False

        if output_path.suffix.lower() != ".tpy":
            output_path = Path(
                str(output_path) + ".tpy"
            )

        if (
            output_path.exists()
            and not bool(overwrite)
        ):
            if not silent:
                print(
                    "TPYCodec.save - Error: The output file already exists "
                    "and overwrite is False. Returning False."
                )
            return False

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        except Exception:
            return False

        builder = _ArchiveBuilder(
            include_subtopology_dictionaries=(
                includeSubtopologyDictionaries
            ),
            include_contents=includeContents,
            include_apertures=includeApertures,
        )

        try:
            root_id = builder.register(topology)
        except (TypeError, ValueError) as exc:
            if not silent: print("TPYCodec.save - Error: " + str(exc) + ". Returning False.")
            return False

        if root_id is None:
            if not silent:
                print(
                    "TPYCodec.save - Error: Could not serialize the topology "
                    "geometry. Returning False."
                )
            return False

        try:
            import topologicpy

            package_version = getattr(
                topologicpy,
                "__version__",
                None,
            )
        except Exception:
            package_version = None

        try:
            backend_name = (
                Core.Backend().__class__.__name__
            )
        except Exception:
            backend_name = None

        manifest = {
            "format": _FORMAT,
            "formatVersion": _VERSION,
            "producer": {
                "package": "topologicpy",
                "version": package_version,
                "backend": backend_name,
            },
            "rootObject": root_id,
            "options": {
                "includeSubtopologyDictionaries": bool(
                    includeSubtopologyDictionaries
                ),
                "includeContents": bool(
                    includeContents
                ),
                "includeApertures": bool(
                    includeApertures
                ),
                "tolerance": tolerance,
            },
            "objects": [
                builder.records[key]
                for key in sorted(
                    builder.records.keys()
                )
            ],
            "relations": builder.relations,
        }

        try:
            manifest_text = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        except Exception as exc:
            if not silent:
                print(
                    "TPYCodec.save - Error: Could not encode the TPY manifest: "
                    f"{exc}. Returning False."
                )
            return False

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=str(
                    output_path.parent
                ),
                prefix=(
                    output_path.name
                    + "."
                ),
                suffix=".tmp",
            ) as temp_file:
                temp_path = Path(
                    temp_file.name
                )

            with zipfile.ZipFile(
                temp_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                archive.writestr(
                    _MANIFEST_NAME,
                    manifest_text,
                )

                for geometry_path, brep in (
                    builder.geometry.items()
                ):
                    archive.writestr(
                        geometry_path,
                        brep,
                    )

            os.replace(
                temp_path,
                output_path,
            )

            return True

        except Exception as exc:
            try:
                if (
                    temp_path is not None
                    and temp_path.exists()
                ):
                    temp_path.unlink()
            except Exception:
                pass

            if not silent:
                print(
                    "TPYCodec.save - Error: Could not write the TPY archive: "
                    f"{exc}. Returning False."
                )

            return False

    @staticmethod
    def load(
        path,
        tolerance=0.0001,
        silent=False,
        _loaded=None,
    ):
        from topologicpy.Aperture import Aperture
        from topologicpy.Context import Context
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology

        locator_cache = {}

        try:
            tolerance = abs(
                float(tolerance)
            )
        except Exception:
            tolerance = 0.0001

        if tolerance <= 0.0:
            tolerance = 0.0001

        try:
            input_path = Path(
                os.fspath(path)
            )
        except Exception:
            return None

        if not input_path.is_file():
            if not silent:
                print(
                    "TPYCodec.load - Error: The input path does not exist. "
                    "Returning None."
                )
            return None

        try:
            with zipfile.ZipFile(
                input_path,
                "r",
            ) as archive:
                names = set(
                    archive.namelist()
                )

                if _MANIFEST_NAME not in names:
                    if not silent:
                        print(
                            "TPYCodec.load - Error: The archive does not "
                            "contain manifest.json. Returning None."
                        )
                    return None

                # Prevent path traversal in future extraction-based versions.
                for name in names:
                    normalized = Path(
                        name
                    )

                    if (
                        normalized.is_absolute()
                        or ".." in normalized.parts
                    ):
                        if not silent:
                            print(
                                "TPYCodec.load - Error: Unsafe archive member "
                                "path. Returning None."
                            )
                        return None

                manifest_bytes = archive.read(
                    _MANIFEST_NAME
                )

                if len(manifest_bytes) > (
                    64 * 1024 * 1024
                ):
                    if not silent:
                        print(
                            "TPYCodec.load - Error: Manifest is unreasonably "
                            "large. Returning None."
                        )
                    return None

                manifest = json.loads(
                    manifest_bytes.decode(
                        "utf-8"
                    )
                )

                if (
                    not isinstance(
                        manifest,
                        dict,
                    )
                    or manifest.get("format")
                    != _FORMAT
                    or manifest.get(
                        "formatVersion"
                    )
                    not in (1, 2)
                ):
                    if not silent:
                        print(
                            "TPYCodec.load - Error: Unsupported TPY format or "
                            "version. Returning None."
                        )
                    return None

                if manifest['formatVersion'] == 2:
                    if len(archive.namelist()) != len(names):
                        raise ValueError("Duplicate TPY archive members.")
                    return TPYCodec._LoadGraphArchive(archive, manifest, tolerance, silent, _loaded=_loaded)

                object_records = manifest.get(
                    "objects",
                    []
                )

                if not isinstance(
                    object_records,
                    list,
                ):
                    return None

                if len(object_records) > 100000:
                    if not silent:
                        print(
                            "TPYCodec.load - Error: Archive contains too many "
                            "objects. Returning None."
                        )
                    return None

                loaded = {}
                record_by_id = {}

                # ----------------------------------------------------------
                # 1. Reconstruct all exact BREP objects first.
                # ----------------------------------------------------------

                for record in object_records:
                    if not isinstance(
                        record,
                        dict,
                    ):
                        return None

                    object_id = record.get(
                        "id"
                    )

                    geometry = record.get(
                        "geometry",
                        {},
                    )

                    if (
                        not isinstance(
                            object_id,
                            str,
                        )
                        or not isinstance(
                            geometry,
                            dict,
                        )
                    ):
                        return None

                    geometry_path = geometry.get(
                        "path"
                    )

                    if (
                        not isinstance(
                            geometry_path,
                            str,
                        )
                        or geometry_path
                        not in names
                        or not geometry_path.startswith(
                            "geometry/"
                        )
                    ):
                        return None

                    brep_bytes = archive.read(
                        geometry_path
                    )

                    brep_text = brep_bytes.decode(
                        "utf-8"
                    )

                    expected_hash = geometry.get(
                        "sha256"
                    )

                    if (
                        isinstance(
                            expected_hash,
                            str,
                        )
                        and _sha256_text(
                            brep_text
                        )
                        != expected_hash
                    ):
                        if not silent:
                            print(
                                "TPYCodec.load - Error: Geometry checksum "
                                "validation failed. Returning None."
                            )
                        return None

                    # Compound-to-Shell promotion in the general importer can
                    # sew detached shading faces and replace native identities.
                    # A TPY archive already records the exact hierarchy: retain
                    # its compound containers and child faces without sewing.
                    if record.get('type')=='Cluster' and type(Core.Backend()).__module__.startswith('topologicpy.pythonocc_backend'):
                        from OCC.Core.TopAbs import TopAbs_COMPOUND
                        def compound_wrapper(shape):
                            if shape.ShapeType()!=TopAbs_COMPOUND:
                                return Topology.ByOCCTShape(shape,silent=True)
                            children=[compound_wrapper(child) for child in TPYCodec._NativeChildren(shape)]
                            wrapper=Core.Cluster.ByTopologies(children,False)
                            wrapper.shape=shape
                            return wrapper
                        from topologicpy.pythonocc_backend.topology import _shape_from_brep_text
                        topology=compound_wrapper(_shape_from_brep_text(brep_text))
                        if isinstance(expected_hash,str):locator_cache[('verified_native_hierarchy',id(topology))]=topology
                    else:
                        topology=Topology.ByBREPString(brep_text,silent=True)


                    if not Topology.IsInstance(
                        topology,
                        "Topology",
                    ):
                        if not silent:
                            print(
                                "TPYCodec.load - Error: Could not reconstruct "
                                f"object '{object_id}'. Returning None."
                            )
                        return None

                    topology = _apply_dictionary(
                        topology,
                        record.get(
                            "dictionary",
                            {},
                        ),
                    )

                    loaded[
                        object_id
                    ] = topology

                    record_by_id[
                        object_id
                    ] = record

                # ----------------------------------------------------------
                # 2. Restore dictionaries on constituent subtopologies.
                # ----------------------------------------------------------

                for object_id, topology in (
                    loaded.items()
                ):
                    record = record_by_id[
                        object_id
                    ]

                    for item in record.get(
                        "subtopologies",
                        [],
                    ):
                        if not isinstance(
                            item,
                            dict,
                        ):
                            continue

                        target = _resolve_locator(topology, item.get('locator'), cache=locator_cache)

                        if target is None:
                            continue

                        _apply_dictionary(
                            target,
                            item.get(
                                "dictionary",
                                {},
                            ),
                        )

                # ----------------------------------------------------------
                # 3. Restore Content and Aperture relationships.
                # ----------------------------------------------------------

                relations = manifest.get(
                    "relations",
                    [],
                )

                if not isinstance(
                    relations,
                    list,
                ):
                    relations = []

                for relation in relations:
                    if not isinstance(
                        relation,
                        dict,
                    ):
                        continue

                    host_object = loaded.get(
                        relation.get(
                            "hostObject"
                        )
                    )

                    child = loaded.get(
                        relation.get(
                            "childObject"
                        )
                    )

                    if (
                        host_object is None
                        or child is None
                    ):
                        continue

                    host = _resolve_locator(host_object, relation.get('hostLocator'), cache=locator_cache)

                    if host is None:
                        continue

                    parameters = relation.get(
                        "context",
                        [
                            0.5,
                            0.5,
                            0.5,
                        ],
                    )

                    try:
                        u = float(parameters[0])
                        v = float(parameters[1])
                        w = float(parameters[2])
                    except Exception:
                        u = v = w = 0.5

                    context = (
                        Context.ByTopologyParameters(
                            host,
                            u=u,
                            v=v,
                            w=w,
                        )
                    )

                    kind = str(
                        relation.get(
                            "kind",
                            "",
                        )
                    ).lower()

                    if kind == "content":
                        try:
                            if context is not None:
                                Core.InstanceCall(
                                    child,
                                    "AddContext",
                                    context,
                                )

                            Core.InstanceCall(
                                host,
                                "AddContent",
                                child,
                            )

                        except Exception:
                            # Public fallback. It may copy the content, but
                            # keeps compatibility with unusual backend builds.
                            try:
                                Topology.AddContent(
                                    host,
                                    [child],
                                    subTopologyType="self",
                                    tolerance=tolerance,
                                    silent=True,
                                )
                            except Exception:
                                pass

                    elif kind == "aperture":
                        try:
                            if context is not None:
                                Aperture.ByTopologyContext(
                                    child,
                                    context,
                                )
                        except Exception:
                            pass

                root_id = manifest.get(
                    "rootObject"
                )

                root = loaded.get(
                    root_id
                )

                if not Topology.IsInstance(
                    root,
                    "Topology",
                ):
                    return None

                if _loaded is not None:
                    _loaded.update(loaded)
                return root

        except Exception as exc:
            if not silent:
                print(
                    "TPYCodec.load - Error: Could not read the TPY archive: "
                    f"{exc}. Returning None."
                )
            return None
