# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Private compatibility helper for :meth:`Topology.Tessellate`."""

from __future__ import annotations

import math


def _mesh_dict(
    vertices,
    faces,
    metadata=None,
    face_sources=None,
):
    vertices = [
        list(vertex)
        for vertex in (vertices or [])
    ]

    faces = [
        list(face)
        for face in (faces or [])
    ]

    result = {
        "schema": "topologicpy.mesh/1",
        "vertices": vertices,
        "faces": faces,
        "cells": [],
        "metadata": dict(metadata or {}),
        "verts": vertices,
        "tris": [
            list(face)
            for face in faces
            if len(face) == 3
        ],
        "quads": [],
        "tets": [],
    }

    if face_sources is not None:
        result["faceSources"] = list(
            face_sources
        )

    return result


def tessellate_topologic_core(
    topology,
    quality="medium",
    linearDeflection=None,
    angularDeflection=None,
    relative=True,
    parallel=True,
    weld=True,
    weldTolerance=0.0001,
    remesh=True,
    mantissa=6,
    silent=False,
):
    """
    Compatibility tessellation for the legacy TopologicCore backend.

    TopologicCore does not expose OCCT's complete tessellation controls through
    its Python API. This path therefore calls the kernel's Face triangulation
    primitive directly and returns the same indexed mesh-data schema as the
    PythonOCC path.

    This helper intentionally does *not* call ``Face.Triangulate`` or
    ``Topology.Triangulate``. Keeping tessellation at the kernel/helper level
    prevents recursion now that ``Face.Triangulate`` is a thin wrapper around
    ``Topology.Triangulate``.
    """
    from topologicpy.Core import Core
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    if not Topology.IsInstance(
        topology,
        "Topology",
    ):
        return None

    if (
        not isinstance(quality, str)
        or quality.strip().lower()
        not in ("coarse", "medium", "fine")
    ):
        return None

    if linearDeflection is not None:
        try:
            value = abs(
                float(linearDeflection)
            )
        except Exception:
            value = 0.0

        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            return None

    if angularDeflection is not None:
        try:
            value = abs(
                float(angularDeflection)
            )
        except Exception:
            value = 0.0

        if (
            not math.isfinite(value)
            or value <= 0.0
            or value >= 180.0
        ):
            return None

    try:
        precision = max(
            0,
            int(mantissa),
        )

        weld_tolerance = max(
            abs(float(weldTolerance)),
            1.0e-12,
        )

    except Exception:
        return None

    source_faces = (
        Topology.Faces(
            topology,
            silent=True,
        )
        or []
    )

    vertices = []
    faces = []
    face_sources = []

    buckets = {}
    inv_tolerance = (
        1.0 / weld_tolerance
    )

    def add_point(point):
        if (
            not isinstance(
                point,
                (list, tuple),
            )
            or len(point) < 3
        ):
            return None

        coords = [
            round(
                float(point[0]),
                precision,
            ),
            round(
                float(point[1]),
                precision,
            ),
            round(
                float(point[2]),
                precision,
            ),
        ]

        if not bool(weld):
            vertices.append(coords)
            return len(vertices) - 1

        key = tuple(
            int(
                math.floor(
                    value * inv_tolerance
                )
            )
            for value in coords
        )

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (
                        key[0] + dx,
                        key[1] + dy,
                        key[2] + dz,
                    )

                    for index in buckets.get(
                        neighbor,
                        [],
                    ):
                        existing = vertices[index]

                        if all(
                            abs(
                                existing[i]
                                - coords[i]
                            )
                            <= weld_tolerance
                            for i in range(3)
                        ):
                            return index

        index = len(vertices)

        vertices.append(coords)

        buckets.setdefault(
            key,
            [],
        ).append(index)

        return index

    def triangulate_face(source_face):
        """Return TopologicCore triangular Faces without public-API recursion."""
        source_vertices = (
            Topology.Vertices(
                source_face,
                silent=True,
            )
            or []
        )

        if len(source_vertices) == 3:
            return [source_face]

        # Match the historical TopologicCore retry policy without flattening,
        # unflattening, or entering Face/Topology public triangulation methods.
        # The second argument is the kernel deflection used by the legacy API.
        for deflection in (0.0, 0.1, 0.2, 0.3, 0.4):
            triangles = []
            try:
                Core.FaceUtility.Triangulate(
                    source_face,
                    float(deflection),
                    triangles,
                )
            except Exception:
                triangles = []

            triangles = [
                triangle
                for triangle in triangles
                if Topology.IsInstance(triangle, "Face")
                and len(Topology.Vertices(triangle, silent=True) or []) == 3
            ]

            if triangles:
                return triangles

        return []

    for source_index, source_face in enumerate(
        source_faces
    ):
        triangles = triangulate_face(
            source_face
        )

        for triangle in triangles:
            triangle_vertices = (
                Topology.Vertices(
                    triangle,
                    silent=True,
                )
                or []
            )

            if len(triangle_vertices) != 3:
                continue

            indices = []

            for vertex in triangle_vertices:
                coordinates = (
                    Vertex.Coordinates(
                        vertex,
                        mantissa=precision,
                    )
                )

                index = add_point(
                    coordinates
                )

                if index is None:
                    indices = []
                    break

                indices.append(index)

            if (
                len(indices) == 3
                and len(set(indices)) == 3
            ):
                faces.append(indices)

                face_sources.append(
                    source_index
                )

    if len(faces) == 0:
        for vertex in (
            Topology.Vertices(
                topology,
                silent=True,
            )
            or []
        ):
            try:
                add_point(
                    Vertex.Coordinates(
                        vertex,
                        mantissa=precision,
                    )
                )
            except Exception:
                pass

    metadata = {
        "source": "topologic_core",
        "quality": quality.strip().lower(),
        "qualityControl": "limited",
        "relative": bool(relative),
        "parallel": bool(parallel),
        "weld": bool(weld),
        "weldTolerance": float(
            weld_tolerance
        ),
        "remesh": bool(remesh),
        "vertexCount": len(vertices),
        "faceCount": len(faces),
        "triangleCount": len(faces),
        "quadCount": 0,
        "cellCount": 0,
    }

    return _mesh_dict(
        vertices,
        faces,
        metadata=metadata,
        face_sources=face_sources,
    )
