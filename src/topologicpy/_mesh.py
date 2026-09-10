# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Shared mesh helpers for TopologicPy.

This private module deliberately separates two different operations:

* ``Topology.Tessellate`` -- CAD BRep -> triangular surface mesh.
* ``Topology.Mesh``       -- gmsh-based numerical mesh, which may contain
  triangles, quads and volume elements.

The canonical mesh-data schema is::

    {
        "vertices": [[x, y, z], ...],
        "faces":    [[i, j, k], [i, j, k, l], ...],
        "cells":    [[...], ...],
        "metadata": {...},
    }

Transitional aliases ``verts``, ``tris``, ``quads`` and ``tets`` are also
returned so existing callers of ``Topology.Mesh`` continue to work while the
new schema is adopted.
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
        # Compatibility aliases. These can be deprecated later.
        "verts": vertices,
        "tris": triangles,
        "quads": quads,
        "tets": tets,
    }
    if face_sources is not None:
        result["faceSources"] = list(face_sources)
    return result


def _gmsh_element_arity(gmsh, element_type):
    """Returns the number of nodes for a first-order gmsh element type."""
    # First-order types documented by gmsh:
    # 1 line2, 2 tri3, 3 quad4, 4 tet4, 5 hex8, 6 prism6, 7 pyramid5.
    fallback = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5}
    if element_type in fallback:
        return fallback[element_type]
    try:
        props = gmsh.model.mesh.getElementProperties(int(element_type))
        # name, dim, order, numNodes, localNodeCoord, numPrimaryNodes
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
    """Creates gmsh mesh data using the canonical TopologicPy mesh schema."""
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

        # Quad-oriented algorithms should actually be allowed to recombine the
        # surface unless the caller explicitly disables it.
        if recombine is None:
            recombine = algorithm2D in (7, 8, 9)
        gmsh.option.setNumber("Mesh.RecombineAll", 1 if bool(recombine) else 0)

        occ = gmsh.model.occ
        occ.importShapes(temp_path)
        occ.synchronize()

        # Variable-size field. A uniform mesh is represented by minSize == maxSize.
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

        # Surface mesh data always represents the boundary that exporters and
        # Face.ByMesh need. gmsh returns these elements for both 2D and 3D meshes.
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


def tessellate_topologic_core(
    topology,
    quality="medium",
    linearDeflection=None,
    angularDeflection=None,
    relative=True,
    interiorLinearDeflection=None,
    interiorAngularDeflection=None,
    minSize=None,
    algorithm="default",
    internalVertices=True,
    parallel=True,
    weld=True,
    weldTolerance=0.0001,
    remesh=True,
    mantissa=6,
    silent=False,
):
    """Compatibility tessellation path for the legacy TopologicCore backend.

    TopologicCore does not expose OCCT's IMeshTools_Parameters through the
    TopologicPy facade, so this path intentionally offers only compatibility,
    not the full quality-control guarantees of the PythonOCC implementation.
    """
    from topologicpy.Face import Face
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex

    if not Topology.IsInstance(topology, "Topology"):
        if not silent:
            print("Topology.Tessellate - Error: The input topology parameter is not a valid topology. Returning None.")
        return None

    if not isinstance(quality, str) or quality.strip().lower() not in ("coarse", "medium", "fine"):
        if not silent:
            print("Topology.Tessellate - Error: quality must be 'coarse', 'medium', or 'fine'. Returning None.")
        return None
    if not isinstance(algorithm, str) or algorithm.strip().lower() not in ("default", "delabella"):
        if not silent:
            print("Topology.Tessellate - Error: algorithm must be 'default' or 'delabella'. Returning None.")
        return None

    # Validate the shared public controls even though TopologicCore cannot apply
    # all of them with OCCT-level fidelity. This keeps backend behavior coherent.
    for name, value in (
        ("linearDeflection", linearDeflection),
        ("interiorLinearDeflection", interiorLinearDeflection),
        ("minSize", minSize),
    ):
        if value is not None:
            try:
                numeric = abs(float(value))
            except Exception:
                numeric = 0.0
            if not math.isfinite(numeric) or numeric <= 0.0:
                if not silent:
                    print(f"Topology.Tessellate - Error: {name} must be greater than zero. Returning None.")
                return None
    for name, value in (
        ("angularDeflection", angularDeflection),
        ("interiorAngularDeflection", interiorAngularDeflection),
    ):
        if value is not None:
            try:
                numeric = abs(float(value))
            except Exception:
                numeric = 0.0
            if not math.isfinite(numeric) or numeric <= 0.0 or numeric >= 180.0:
                if not silent:
                    print(f"Topology.Tessellate - Error: {name} must be between 0 and 180 degrees. Returning None.")
                return None

    try:
        precision = max(0, int(mantissa))
        weld_tolerance = max(abs(float(weldTolerance)), 1.0e-12)
    except Exception:
        return None

    source_faces = Topology.Faces(topology, silent=True)
    if source_faces is None:
        return None

    vertices = []
    faces = []
    face_sources = []
    buckets = {}
    inv = 1.0 / weld_tolerance

    def add_point(point):
        coords = [round(float(v), precision) for v in point]
        if not weld:
            vertices.append(coords)
            return len(vertices) - 1
        key = tuple(int(math.floor(v * inv)) for v in coords)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for index in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        existing = vertices[index]
                        if all(abs(existing[i] - coords[i]) <= weld_tolerance for i in range(3)):
                            return index
        index = len(vertices)
        vertices.append(coords)
        buckets.setdefault(key, []).append(index)
        return index

    for source_index, source_face in enumerate(source_faces):
        source_vertices = Topology.Vertices(source_face, silent=True) or []
        if len(source_vertices) == 3:
            triangles = [source_face]
        else:
            triangles = Face.Triangulate(source_face, mode=0, tolerance=weld_tolerance, silent=True) or []
        for triangle in triangles:
            triangle_vertices = Topology.Vertices(triangle, silent=True) or []
            if len(triangle_vertices) != 3:
                continue
            indices = []
            for vertex in triangle_vertices:
                coords = Vertex.Coordinates(vertex, mantissa=precision)
                indices.append(add_point(coords))
            if len(set(indices)) == 3:
                faces.append(indices)
                face_sources.append(source_index)

    if not faces:
        try:
            for vertex in Topology.Vertices(topology, silent=True) or []:
                coords = Vertex.Coordinates(vertex, mantissa=precision)
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    add_point(coords[:3])
        except Exception:
            pass

    metadata = {
        "source": "topologic_core",
        "quality": quality.strip().lower(),
        "qualityControl": "limited",
        "vertexCount": len(vertices),
        "faceCount": len(faces),
        "triangleCount": len(faces),
        "quadCount": 0,
        "cellCount": 0,
    }
    return _mesh_dict(vertices, faces, cells=[], metadata=metadata, face_sources=face_sources)


def triangulate_topology(
    topology,
    transferDictionaries=False,
    mode=0,
    meshSize=None,
    tolerance=0.0001,
    silent=False,
):
    """Compatibility implementation of ``Topology.Triangulate``.

    Mode 0 delegates to ``Topology.Tessellate``. All other historical modes
    delegate to ``Topology.Mesh`` and then convert triangle/quad mesh elements
    to triangles before typed topology reconstruction.
    """
    from topologicpy.Cell import Cell
    from topologicpy.CellComplex import CellComplex
    from topologicpy.Cluster import Cluster
    from topologicpy.Face import Face
    from topologicpy.Shell import Shell
    from topologicpy.Topology import Topology

    if not Topology.IsInstance(topology, "Topology"):
        if not silent:
            print("Topology.Triangulate - Error: The input topology parameter is not a valid topology. Returning None.")
        return None

    topology_type = Topology.Type(topology)
    if topology_type in (Topology.TypeID("Vertex"), Topology.TypeID("Edge"), Topology.TypeID("Wire")):
        if not silent:
            print("Topology.Triangulate - Warning: The input topology contains no faces. Returning the original topology.")
        return topology

    if topology_type == Topology.TypeID("Cluster"):
        try:
            constituents = Cluster.Topologies(topology, tolerance=tolerance, silent=True)
        except Exception:
            constituents = None
        if not isinstance(constituents, list):
            try:
                constituents = []
                result = topology.Topologies()
                if isinstance(result, list):
                    constituents = result
            except Exception:
                constituents = []
        constituents = [item for item in constituents if Topology.IsInstance(item, "Topology")]
        if not constituents:
            if not silent:
                print("Topology.Triangulate - Error: Could not retrieve constituent topologies from the input Cluster. Returning None.")
            return None
        faceted = []
        for constituent in constituents:
            item = Topology.Triangulate(
                constituent,
                transferDictionaries=transferDictionaries,
                mode=mode,
                meshSize=meshSize,
                tolerance=tolerance,
                silent=True,
            )
            if not Topology.IsInstance(item, "Topology"):
                return None
            faceted.append(item)
        try:
            return Cluster.ByTopologies(faceted, silent=True)
        except TypeError:
            return Cluster.ByTopologies(faceted)

    expected_cell_count = None
    if topology_type == Topology.TypeID("CellComplex"):
        expected_cell_count = len(Topology.Cells(topology, silent=True) or [])

    source_faces = Topology.Faces(topology, silent=True)
    if not isinstance(source_faces, list) or not source_faces:
        if not silent:
            print("Topology.Triangulate - Error: Could not retrieve any Faces from the input topology. Returning None.")
        return None

    face_triangles = []
    selectors = []

    legacy_mode_to_mesh_algorithm = {
        1: 1,  # MeshAdapt
        3: 3,  # Initial mesh only
        5: 4,  # Delaunay
        6: 5,  # Frontal-Delaunay
        7: 6,  # BAMG
        8: 7,  # Frontal-Delaunay for Quads
        9: 8,  # Packing of Parallelograms
    }

    for source_face in source_faces:
        source_vertices = Topology.Vertices(source_face, silent=True) or []
        if len(source_vertices) == 3:
            triangles = [source_face]
        else:
            if mode == 0:
                mesh = Topology.Tessellate(source_face, quality="fine", weld=True, weldTolerance=tolerance, silent=True)
            else:
                # Face.Triangulate historically treated any undocumented non-zero
                # mode as gmsh mode 6 (Frontal-Delaunay). Preserve that behavior.
                algorithm2d = legacy_mode_to_mesh_algorithm.get(mode, 5)
                if meshSize is None:
                    min_size = 0.1
                    max_size = 1.0
                else:
                    try:
                        size = abs(float(meshSize))
                    except Exception:
                        size = 0.0
                    if size <= 0.0:
                        if not silent:
                            print("Topology.Triangulate - Error: meshSize must be greater than zero. Returning None.")
                        return None
                    min_size = size
                    max_size = size
                mesh = Topology.Mesh(
                    source_face,
                    minSize=min_size,
                    maxSize=max_size,
                    algorithm2D=algorithm2d,
                    meshDim=2,
                    silent=True,
                    recombine=(mode in (8, 9)),
                )
            if not isinstance(mesh, dict):
                if not silent:
                    print("Topology.Triangulate - Error: Could not mesh one of the input Faces. Returning None.")
                return None
            triangles = Face.ByMesh(mesh, triangulateQuads=True, quadSplit="shortest", tolerance=tolerance, silent=True)
            if not isinstance(triangles, list) or not triangles:
                if not silent:
                    print("Topology.Triangulate - Error: Mesh conversion returned no triangular Faces. Returning None.")
                return None

        for triangle in triangles:
            if transferDictionaries:
                selector = Topology.Centroid(triangle, silent=True)
                if Topology.IsInstance(selector, "Vertex"):
                    selector = Topology.SetDictionary(selector, Topology.Dictionary(source_face), silent=True)
                    if Topology.IsInstance(selector, "Vertex"):
                        selectors.append(selector)
            face_triangles.append(triangle)

    if not face_triangles:
        return None

    result = None
    if topology_type in (Topology.TypeID("Face"), Topology.TypeID("Shell")):
        try:
            result = Shell.ByFaces(face_triangles, tolerance=tolerance, silent=True)
        except TypeError:
            result = Shell.ByFaces(face_triangles, tolerance=tolerance)
    elif topology_type == Topology.TypeID("Cell"):
        try:
            result = Cell.ByFaces(face_triangles, tolerance=tolerance, silent=True)
        except TypeError:
            result = Cell.ByFaces(face_triangles, tolerance=tolerance)
    elif topology_type == Topology.TypeID("CellComplex"):
        if not Topology._IsTopologicCoreBackend():
            from topologicpy.Core import Core
            attempts = (
                lambda: Core.CellComplex.ByFaces(face_triangles, tolerance, False),
                lambda: Core.CellComplex.ByFaces(face_triangles, tolerance),
                lambda: Core.CellComplex.ByFaces(face_triangles),
            )
            for attempt in attempts:
                try:
                    candidate = attempt()
                except Exception:
                    continue
                if Topology.IsInstance(candidate, "CellComplex"):
                    result = candidate
                    break
            if Topology.IsInstance(result, "CellComplex") and expected_cell_count is not None:
                resulting_count = len(Topology.Cells(result, silent=True) or [])
                if resulting_count != expected_cell_count:
                    if not silent:
                        print(
                            "Topology.Triangulate - Error: The active backend changed the CellComplex cell count "
                            f"from {expected_cell_count} to {resulting_count}. Returning None."
                        )
                    return None
        else:
            try:
                result = CellComplex.ByFaces(face_triangles, tolerance=tolerance, silent=True)
            except TypeError:
                result = CellComplex.ByFaces(face_triangles, tolerance=tolerance)

    if not Topology.IsInstance(result, "Topology") and Topology._IsTopologicCoreBackend():
        try:
            result = Cluster.ByTopologies(face_triangles, silent=True)
        except TypeError:
            result = Cluster.ByTopologies(face_triangles)
        if Topology.IsInstance(result, "Topology"):
            result = Topology.SelfMerge(result, tolerance=tolerance, silent=silent)

    if not Topology.IsInstance(result, "Topology"):
        if not silent:
            print("Topology.Triangulate - Error: Could not reconstruct the triangulated topology. Returning None.")
        return None

    if transferDictionaries and selectors:
        result = Topology.TransferDictionariesBySelectors(
            result,
            selectors,
            tranFaces=True,
            tolerance=tolerance,
        )
    return result
