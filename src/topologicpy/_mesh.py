# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Private numerical-meshing helpers for :meth:`Topology.Mesh`.

This module intentionally contains only gmsh-based numerical meshing.

* ``Topology.Triangulate`` is a backend-native topological conversion.
* ``Topology.Tessellate`` is a backend-native BRep -> triangular surface mesh.
* ``Topology.Mesh`` is a gmsh-based numerical 2D/3D mesher.

No package is installed at runtime. If gmsh is unavailable, ``Topology.Mesh``
returns None with an explanatory message unless ``silent=True``.
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid


def _mesh_dict(vertices, faces, cells=None, metadata=None, face_sources=None):
    """Returns canonical mesh data plus compatibility aliases."""
    vertices = [list(vertex) for vertex in (vertices or [])]
    faces = [list(face) for face in (faces or [])]
    cells = [list(cell) for cell in (cells or [])]
    triangles = [list(face) for face in faces if len(face) == 3]
    quads = [list(face) for face in faces if len(face) == 4]
    tets = [list(cell) for cell in cells if len(cell) == 4]

    result = {
        "schema": "topologicpy.mesh/1",
        "vertices": vertices,
        "faces": faces,
        "cells": cells,
        "metadata": dict(metadata or {}),
        "verts": vertices,
        "tris": triangles,
        "quads": quads,
        "tets": tets,
    }
    if face_sources is not None:
        result["faceSources"] = list(face_sources)
    return result


def _gmsh_element_arity(gmsh, element_type):
    """Returns the number of nodes for a supported first-order gmsh element."""
    # First-order gmsh element types:
    # 1 line2, 2 tri3, 3 quad4, 4 tet4, 5 hex8, 6 prism6, 7 pyramid5.
    fallback = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5}
    if element_type in fallback:
        return fallback[element_type]
    try:
        props = gmsh.model.mesh.getElementProperties(int(element_type))
        if len(props) >= 4:
            return int(props[3])
    except Exception:
        pass
    return None


def gmsh_mesh(
    topology,
    minSize: float = 0.1,
    maxSize: float = 1.0,
    algorithm2D: int = 1,
    algorithm3D: int = 1,
    refineEdges: bool = True,
    refineFaces: bool = True,
    optimize: bool = True,
    meshDim: int = None,
    mantissa: int = 6,
    silent: bool = False,
    recombine: bool = None,
):
    """Creates 2D/3D gmsh mesh data using the canonical TopologicPy schema."""
    from topologicpy.Topology import Topology

    try:
        import gmsh
    except Exception:
        if not silent:
            print(
                "Topology.Mesh - Error: The optional gmsh package is not installed. "
                "Install gmsh and try again. Returning None."
            )
        return None

    if not Topology.IsInstance(topology, "Topology"):
        if not silent:
            print("Topology.Mesh - Error: The input topology parameter is not a valid topology. Returning None.")
        return None

    try:
        min_size = float(minSize)
        max_size = float(maxSize)
    except Exception:
        if not silent:
            print("Topology.Mesh - Error: minSize and maxSize must be valid numbers. Returning None.")
        return None

    if not math.isfinite(min_size) or not math.isfinite(max_size) or min_size <= 0.0 or max_size <= 0.0 or min_size > max_size:
        if not silent:
            print("Topology.Mesh - Error: minSize and maxSize must be positive and minSize must not exceed maxSize. Returning None.")
        return None

    if algorithm2D not in range(1, 10):
        if not silent:
            print("Topology.Mesh - Error: Bad algorithm2D number. Returning None.")
        return None
    if algorithm3D not in range(1, 7):
        if not silent:
            print("Topology.Mesh - Error: Bad algorithm3D number. Returning None.")
        return None

    algorithm_mapping_2d = {
        1: 1,   # MeshAdapt
        2: 2,   # Automatic
        3: 3,   # Initial mesh only
        4: 5,   # Delaunay
        5: 6,   # Frontal-Delaunay
        6: 7,   # BAMG
        7: 8,   # Frontal-Delaunay for Quads
        8: 9,   # Packing of Parallelograms
        9: 11,  # Quasi-structured Quad
    }
    algorithm_mapping_3d = {
        1: 1,   # Delaunay
        2: 3,   # Initial mesh only
        3: 4,   # Frontal
        4: 7,   # MMG3D
        5: 9,   # R-tree
        6: 10,  # HXT
    }

    if meshDim is None:
        try:
            meshDim = 3 if bool(Topology.Cells(topology, silent=True)) else 2
        except Exception:
            meshDim = 2
    if meshDim not in (2, 3):
        if not silent:
            print("Topology.Mesh - Error: meshDim must be 2 or 3. Returning None.")
        return None

    try:
        brep_string = Topology.BREPString(topology)
    except Exception as exc:
        if not silent:
            print(f"Topology.Mesh - Error: Could not obtain BREP data: {exc}. Returning None.")
        return None
    if not isinstance(brep_string, str) or not brep_string:
        if not silent:
            print("Topology.Mesh - Error: Could not obtain BREP data. Returning None.")
        return None

    temp_path = None
    was_initialized = False
    model_added = False

    try:
        with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as handle:
            handle.write(brep_string.encode("utf-8"))
            temp_path = handle.name

        try:
            was_initialized = bool(gmsh.isInitialized())
        except Exception:
            was_initialized = False
        if not was_initialized:
            gmsh.initialize()

        model_name = f"topologic_mesh_{uuid.uuid4().hex}"
        gmsh.model.add(model_name)
        model_added = True

        gmsh.option.setNumber("Mesh.Algorithm", algorithm_mapping_2d[algorithm2D])
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_mapping_3d[algorithm3D])
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        if recombine is None:
            recombine = algorithm2D in (7, 8, 9)
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if bool(recombine) else 0)

        occ = gmsh.model.occ
        occ.importShapes(temp_path)
        occ.synchronize()

        if min_size < max_size and (refineEdges or refineFaces):
            field = gmsh.model.mesh.field
            distance_id = 1
            field.add("Distance", distance_id)
            has_sources = False

            if refineEdges:
                edges = gmsh.model.getEntities(1)
                if edges:
                    field.setNumbers(distance_id, "EdgesList", [entity[1] for entity in edges])
                    has_sources = True

            if refineFaces:
                faces = gmsh.model.getEntities(2)
                if faces:
                    field.setNumbers(distance_id, "FacesList", [entity[1] for entity in faces])
                    has_sources = True

            if has_sources:
                threshold_id = 2
                field.add("Threshold", threshold_id)
                field.setNumber(threshold_id, "InField", distance_id)
                field.setNumber(threshold_id, "SizeMin", min_size)
                field.setNumber(threshold_id, "SizeMax", max_size)
                field.setNumber(threshold_id, "DistMin", min_size * 2.0)
                field.setNumber(threshold_id, "DistMax", max_size * 2.0)
                field.setAsBackgroundMesh(threshold_id)

        gmsh.model.mesh.generate(meshDim)
        if optimize:
            try:
                gmsh.model.mesh.optimize()
            except Exception:
                pass

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_coords) == 0:
            if not silent:
                print("Topology.Mesh - Error: gmsh returned no mesh nodes. Returning None.")
            return None

        precision = max(0, int(mantissa))
        tag_to_index = {}
        vertices = []
        for index, tag in enumerate(node_tags):
            coords = [
                round(float(node_coords[3 * index]), precision),
                round(float(node_coords[3 * index + 1]), precision),
                round(float(node_coords[3 * index + 2]), precision),
            ]
            tag_to_index[int(tag)] = len(vertices)
            vertices.append(coords)

        def extract_elements(dim, allowed_types):
            result = []
            element_types, _, element_nodes = gmsh.model.mesh.getElements(dim)
            for element_type, connectivity in zip(element_types, element_nodes):
                element_type = int(element_type)
                if element_type not in allowed_types:
                    continue
                arity = _gmsh_element_arity(gmsh, element_type)
                if not arity:
                    continue
                for offset in range(0, len(connectivity), arity):
                    nodes = connectivity[offset: offset + arity]
                    if len(nodes) != arity:
                        continue
                    try:
                        element = [tag_to_index[int(tag)] for tag in nodes]
                    except KeyError:
                        continue
                    if len(set(element)) == len(element):
                        result.append(element)
            return result

        faces = extract_elements(2, {2, 3})
        if meshDim == 2 and not faces:
            if not silent:
                print("Topology.Mesh - Error: gmsh returned no triangle or quad surface elements. Returning None.")
            return None

        cells = []
        if meshDim == 3:
            cells = extract_elements(3, {4, 5, 6, 7})
            if not cells:
                if not silent:
                    print("Topology.Mesh - Error: gmsh returned no supported first-order volume elements. Returning None.")
                return None

        metadata = {
            "source": "gmsh",
            "meshDimension": int(meshDim),
            "algorithm2D": int(algorithm2D),
            "algorithm3D": int(algorithm3D),
            "minSize": min_size,
            "maxSize": max_size,
            "recombine": bool(recombine),
            "vertexCount": len(vertices),
            "faceCount": len(faces),
            "triangleCount": sum(1 for face in faces if len(face) == 3),
            "quadCount": sum(1 for face in faces if len(face) == 4),
            "cellCount": len(cells),
        }
        return _mesh_dict(vertices, faces, cells=cells, metadata=metadata)

    except Exception as exc:
        if not silent:
            print(f"Topology.Mesh - Error: gmsh meshing failed: {exc}. Returning None.")
        return None
    finally:
        if model_added:
            try:
                gmsh.model.remove()
            except Exception:
                pass
        if not was_initialized:
            try:
                if gmsh.isInitialized():
                    gmsh.finalize()
            except Exception:
                try:
                    gmsh.finalize()
                except Exception:
                    pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
