import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Edge import Edge

BACKEND = os.environ.get(
    "TOPOLOGICPY_CORE_BACKEND",
    "",
).lower()

IS_PYTHONOCC = (
    "pythonocc" in BACKEND
)


def _validate_mesh(mesh):
    assert isinstance(mesh, dict)
    assert mesh["schema"] == "topologicpy.mesh/1"

    vertices = mesh["vertices"]
    faces = mesh["faces"]
    cells = mesh["cells"]

    assert isinstance(vertices, list)
    assert isinstance(faces, list)
    assert cells == []

    assert mesh["verts"] is vertices
    assert mesh["tris"] == faces
    assert mesh["quads"] == []
    assert mesh["tets"] == []

    for face in faces:
        assert len(face) == 3
        assert len(set(face)) == 3

        for index in face:
            assert 0 <= index < len(vertices)

    metadata = mesh["metadata"]

    assert metadata["vertexCount"] == len(
        vertices
    )
    assert metadata["faceCount"] == len(
        faces
    )
    assert metadata["triangleCount"] == len(
        faces
    )


def test_tessellate_planar_face_returns_canonical_triangle_mesh():
    face = Face.Rectangle(
        width=4.0,
        length=3.0,
        silent=True,
    )

    assert Topology.IsInstance(
        face,
        "Face",
    )

    mesh = Topology.Tessellate(
        face,
        silent=True,
    )

    _validate_mesh(mesh)

    assert len(mesh["faces"]) >= 2

    assert mesh["metadata"]["source"] in (
        "occt",
        "topologic_core",
    )


def test_tessellate_validation_is_non_throwing():
    assert (
        Topology.Tessellate(
            None,
            silent=True,
        )
        is None
    )

    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    assert (
        Topology.Tessellate(
            face,
            quality="bad",
            silent=True,
        )
        is None
    )

    assert (
        Topology.Tessellate(
            face,
            linearDeflection=0,
            silent=True,
        )
        is None
    )

    assert (
        Topology.Tessellate(
            face,
            angularDeflection=180,
            silent=True,
        )
        is None
    )


def test_tessellate_box_produces_surface_triangles_only():
    cell = Cell.Box(
        width=2.0,
        length=3.0,
        height=4.0,
        silent=True,
    )

    mesh = Topology.Tessellate(
        cell,
        silent=True,
    )

    _validate_mesh(mesh)

    assert len(mesh["faces"]) >= 12
    assert mesh["cells"] == []


@pytest.mark.pythonocc_only
def test_tessellate_exact_cylindrical_shell_preserves_curved_surface():
    c0 = Edge.Circle(
        radius=1.0,
        silent=True,
    )

    c1 = Topology.Translate(
        Edge.Circle(
            radius=1.0,
            silent=True,
        ),
        z=2.0,
        silent=True,
    )
    w0 = Wire.ByEdges(
        [c0],
        silent=True,
    )

    w1 = Wire.ByEdges(
        [c1],
        silent=True,
    )

    shell = Shell.ByWires(
        [w0, w1],
        polyhedron=False,
        silent=True,
    )

    assert Topology.IsInstance(
        shell,
        "Shell",
    )

    coarse = Topology.Tessellate(
        shell,
        quality="coarse",
        remesh=True,
        silent=True,
    )

    fine = Topology.Tessellate(
        shell,
        quality="fine",
        remesh=True,
        silent=True,
    )

    _validate_mesh(coarse)
    _validate_mesh(fine)

    assert coarse["metadata"]["source"] == "occt"
    assert fine["metadata"]["source"] == "occt"

    assert len(coarse["faces"]) > 0
    assert len(fine["faces"]) >= len(
        coarse["faces"]
    )

    for x, y, z in fine["vertices"]:
        radius = math.sqrt(
            x * x
            + y * y
        )

        assert math.isclose(
            radius,
            1.0,
            rel_tol=1.0e-4,
            abs_tol=1.0e-4,
        )

        assert (
            -1.0e-6
            <= z
            <= 2.0 + 1.0e-6
        )


@pytest.mark.pythonocc_only
def test_tessellate_welding_reduces_or_preserves_vertex_count():
    cell = Cell.Cylinder(
        radius=1.0,
        height=2.0,
        uSides=24,
        vSides=1,
        polyhedron=False,
        silent=True,
    )

    welded = Topology.Tessellate(
        cell,
        quality="medium",
        weld=True,
        remesh=True,
        silent=True,
    )

    unwelded = Topology.Tessellate(
        cell,
        quality="medium",
        weld=False,
        remesh=True,
        silent=True,
    )

    _validate_mesh(welded)
    _validate_mesh(unwelded)

    assert len(
        welded["vertices"]
    ) <= len(
        unwelded["vertices"]
    )
