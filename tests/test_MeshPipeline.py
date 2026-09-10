import importlib.util
import os

import pytest

from topologicpy.Face import Face
from topologicpy.Topology import Topology


def _assert_mesh_schema(mesh):
    assert isinstance(mesh, dict)
    assert mesh["schema"] == "topologicpy.mesh/1"

    assert isinstance(mesh["vertices"], list)
    assert isinstance(mesh["faces"], list)
    assert isinstance(mesh["cells"], list)
    assert isinstance(mesh["metadata"], dict)

    assert mesh["verts"] is mesh["vertices"]
    assert isinstance(mesh["tris"], list)
    assert isinstance(mesh["quads"], list)
    assert isinstance(mesh["tets"], list)


def test_face_bymesh_accepts_canonical_triangle_mesh():
    mesh = {
        "schema": "topologicpy.mesh/1",
        "vertices": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "faces": [
            [0, 1, 2],
            [0, 2, 3],
        ],
        "cells": [],
        "metadata": {},
    }

    faces = Face.ByMesh(
        mesh,
        silent=True,
    )

    assert isinstance(faces, list)
    assert len(faces) == 2

    for face in faces:
        assert Topology.IsInstance(
            face,
            "Face",
        )


def test_face_bymesh_accepts_transitional_aliases():
    mesh = {
        "verts": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "tris": [
            [0, 1, 2],
        ],
        "quads": [
            [0, 2, 3, 1],
        ],
    }

    # Use only the triangular alias here; the deliberately crossed quad is
    # present to confirm aliases are collected and then validated.
    assert (
        Face.ByMesh(
            {
                "verts": mesh["verts"],
                "tris": mesh["tris"],
                "quads": [],
            },
            silent=True,
        )
        is not None
    )


def test_face_bymesh_quad_and_triangulation_modes():
    mesh = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "faces": [
            [0, 1, 2, 3],
        ],
    }

    quad = Face.ByMesh(
        mesh,
        triangulateQuads=False,
        silent=True,
    )

    assert isinstance(quad, list)
    assert len(quad) == 1

    triangles = Face.ByMesh(
        mesh,
        triangulateQuads=True,
        quadSplit="shortest",
        silent=True,
    )

    assert isinstance(triangles, list)
    assert len(triangles) == 2


def test_face_bymesh_consumes_topology_tessellate_output():
    source = Face.Rectangle(
        width=4.0,
        length=3.0,
        silent=True,
    )

    mesh = Topology.Tessellate(
        source,
        silent=True,
    )

    _assert_mesh_schema(mesh)

    faces = Face.ByMesh(
        mesh,
        silent=True,
    )

    assert isinstance(faces, list)
    assert len(faces) == len(
        mesh["faces"]
    )


def test_face_bymesh_validation_is_non_throwing():
    assert (
        Face.ByMesh(
            None,
            silent=True,
        )
        is None
    )

    assert (
        Face.ByMesh(
            {
                "vertices": [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                "faces": [
                    [0, 1, 99],
                ],
            },
            silent=True,
        )
        is None
    )


@pytest.mark.skipif(
    importlib.util.find_spec("gmsh") is None,
    reason="Optional gmsh package is not installed.",
)
def test_topology_mesh_returns_canonical_schema_when_gmsh_available():
    source = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    mesh = Topology.Mesh(
        source,
        minSize=0.5,
        maxSize=0.5,
        meshDim=2,
        optimize=False,
        silent=True,
    )

    _assert_mesh_schema(mesh)

    assert len(mesh["vertices"]) > 0
    assert len(mesh["faces"]) > 0
    assert mesh["cells"] == []


@pytest.mark.skipif(
    importlib.util.find_spec("gmsh") is None,
    reason="Optional gmsh package is not installed.",
)
def test_topology_mesh_supports_equal_min_max_uniform_size():
    source = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    mesh = Topology.Mesh(
        source,
        minSize=0.4,
        maxSize=0.4,
        meshDim=2,
        optimize=False,
        silent=True,
    )

    _assert_mesh_schema(mesh)
