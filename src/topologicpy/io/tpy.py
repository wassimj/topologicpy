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
_VERSION = 1
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
    """Encode common Python values to JSON-safe tagged data."""
    if value is None or isinstance(
        value,
        (bool, int, str),
    ):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return {
                "__tpy_type__": "float",
                "value": "nan",
            }
        if math.isinf(value):
            return {
                "__tpy_type__": "float",
                "value": (
                    "+inf"
                    if value > 0
                    else "-inf"
                ),
            }
        return value

    if isinstance(value, bytes):
        return {
            "__tpy_type__": "bytes",
            "data": base64.b64encode(
                value
            ).decode("ascii"),
        }

    if isinstance(value, tuple):
        return {
            "__tpy_type__": "tuple",
            "items": [
                _encode_value(item)
                for item in value
            ],
        }

    if isinstance(value, list):
        return [
            _encode_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _encode_value(item)
            for key, item in value.items()
        }

    # Topologic dictionaries normally contain primitive/list values.
    # Preserve an unsupported value textually rather than making the entire
    # archive fail.
    return {
        "__tpy_type__": "repr",
        "class": (
            value.__class__.__name__
            if value is not None
            else "NoneType"
        ),
        "value": repr(value),
    }


def _decode_value(value):
    if isinstance(value, list):
        return [
            _decode_value(item)
            for item in value
        ]

    if not isinstance(value, dict):
        return value

    tag = value.get(
        "__tpy_type__",
        None,
    )

    if tag == "float":
        raw = value.get("value")
        if raw == "nan":
            return float("nan")
        if raw == "+inf":
            return float("inf")
        if raw == "-inf":
            return float("-inf")
        return None

    if tag == "bytes":
        try:
            return base64.b64decode(
                value.get(
                    "data",
                    "",
                )
            )
        except Exception:
            return b""

    if tag == "tuple":
        return tuple(
            _decode_value(item)
            for item in value.get(
                "items",
                [],
            )
        )

    if tag == "repr":
        # Do not eval archived text.
        return value.get(
            "value",
            "",
        )

    return {
        str(key): _decode_value(item)
        for key, item in value.items()
    }


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
    from topologicpy.Topology import Topology

    if a is b:
        return True

    try:
        return bool(
            Topology.IsSame(
                a,
                b,
                silent=True,
            )
        )
    except Exception:
        return False


def _subtopologies(topology):
    """
    Return unique subtopologies grouped by type, excluding the topology itself.
    """
    from topologicpy.Topology import Topology

    root_brep, root_hash = _brep_data(
        topology
    )

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

        for item in items:
            if not Topology.IsInstance(
                item,
                "Topology",
            ):
                continue

            if _same_topology(
                item,
                topology,
            ):
                continue

            _, item_hash = _brep_data(
                item
            )

            if (
                root_hash is not None
                and item_hash == root_hash
            ):
                continue

            is_duplicate = False

            for previous in unique:
                if _same_topology(
                    item,
                    previous,
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
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


def _make_locator(
    type_name,
    index,
    topology,
):
    _, brep_hash = _brep_data(
        topology
    )

    return {
        "type": type_name,
        "index": int(index),
        "brepSha256": brep_hash,
        "signature": _signature(
            topology
        ),
    }


def _candidate_brep_hash(topology):
    _, value = _brep_data(
        topology
    )
    return value


def _resolve_locator(
    topology,
    locator,
):
    from topologicpy.Topology import Topology

    if locator is None:
        return topology

    if not isinstance(
        locator,
        dict,
    ):
        return None

    type_name = str(
        locator.get(
            "type",
            "",
        )
    ).lower()

    if type_name not in _SUBTOPOLOGY_TYPES:
        return None

    try:
        candidates = Topology.SubTopologies(
            topology,
            subTopologyType=type_name,
            silent=True,
        ) or []
    except Exception:
        candidates = []

    cleaned = []

    for candidate in candidates:
        if not Topology.IsInstance(
            candidate,
            "Topology",
        ):
            continue

        if _same_topology(
            candidate,
            topology,
        ):
            continue

        cleaned.append(candidate)

    candidates = cleaned

    if len(candidates) < 1:
        return None

    expected_brep_hash = locator.get(
        "brepSha256"
    )

    if expected_brep_hash:
        for candidate in candidates:
            if (
                _candidate_brep_hash(
                    candidate
                )
                == expected_brep_hash
            ):
                return candidate

    expected_signature = locator.get(
        "signature",
        {}
    )

    expected_signature_hash = (
        expected_signature.get(
            "hash"
        )
        if isinstance(
            expected_signature,
            dict,
        )
        else None
    )

    if expected_signature_hash:
        for candidate in candidates:
            try:
                if (
                    _signature(
                        candidate
                    ).get("hash")
                    == expected_signature_hash
                ):
                    return candidate
            except Exception:
                pass

    try:
        index = int(
            locator.get(
                "index",
                -1,
            )
        )
    except Exception:
        index = -1

    if (
        index >= 0
        and index < len(candidates)
    ):
        return candidates[index]

    return None


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

        for (
            sub_type,
            index,
            subtopology,
        ) in _subtopologies(topology):
            locator = _make_locator(
                sub_type,
                index,
                subtopology,
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
                dictionary = (
                    _python_dictionary(
                        subtopology
                    )
                )

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
                    contents = Topology.Contents(
                        host,
                        silent=True,
                    ) or []
                except Exception:
                    contents = []

                relation_seen = set()

                for content in contents:
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
                ) in _raw_apertures(host):
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

        root_id = builder.register(
            topology
        )

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
    ):
        from topologicpy.Aperture import Aperture
        from topologicpy.Context import Context
        from topologicpy.Core import Core
        from topologicpy.Topology import Topology

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
                    != _VERSION
                ):
                    if not silent:
                        print(
                            "TPYCodec.load - Error: Unsupported TPY format or "
                            "version. Returning None."
                        )
                    return None

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

                    topology = Topology.ByBREPString(
                        brep_text,
                        silent=True,
                    )

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

                        target = _resolve_locator(
                            topology,
                            item.get(
                                "locator"
                            ),
                        )

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

                    host = _resolve_locator(
                        host_object,
                        relation.get(
                            "hostLocator"
                        ),
                    )

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

                return root

        except Exception as exc:
            if not silent:
                print(
                    "TPYCodec.load - Error: Could not read the TPY archive: "
                    f"{exc}. Returning None."
                )
            return None
