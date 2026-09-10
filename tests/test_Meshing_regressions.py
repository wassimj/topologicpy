"""Additional meshing regressions not duplicated by test_Topology_Tessellate or test_MeshPipeline."""

import inspect

import pytest

from topologicpy.Cell import Cell
from topologicpy.Face import Face
from topologicpy.Topology import Topology


def _rectangle():
    face = Face.Rectangle(width=2, length=1, silent=True)
    assert Topology.IsInstance(face, "Face")
    return face


def _assert_mesh_schema(mesh):
    assert isinstance(mesh, dict)
    assert mesh.get("schema") == "topologicpy.mesh/1"
    assert isinstance(mesh.get("vertices"), list)
    assert isinstance(mesh.get("faces"), list)
    assert isinstance(mesh.get("cells"), list)
    assert isinstance(mesh.get("metadata"), dict)

@pytest.mark.pythonocc_only
def test_pythonocc_quality_presets_control_density_and_are_repeatable():

    sphere = Cell.Sphere(radius=1, silent=True)
    assert Topology.IsInstance(sphere, "Cell")

    coarse = Topology.Tessellate(sphere, quality="coarse", silent=True)
    fine = Topology.Tessellate(sphere, quality="fine", silent=True)
    coarse_again = Topology.Tessellate(sphere, quality="coarse", silent=True)

    _assert_mesh_schema(coarse)
    _assert_mesh_schema(fine)
    _assert_mesh_schema(coarse_again)

    assert fine["metadata"]["linearDeflection"] < coarse["metadata"]["linearDeflection"]
    assert fine["metadata"]["angularDeflection"] < coarse["metadata"]["angularDeflection"]
    assert len(fine["faces"]) >= len(coarse["faces"])
    assert len(coarse_again["faces"]) == len(coarse["faces"])

def test_triangulate_mode_zero_is_legacy_compatible_shell_facade():
    face = _rectangle()
    result = Topology.Triangulate(face, mode=0, silent=True)
    assert Topology.IsInstance(result, "Shell")
    result_faces = Topology.Faces(result, silent=True) or []
    assert len(result_faces) >= 2
    assert all(len(Topology.Vertices(item, silent=True) or []) == 3 for item in result_faces)

@pytest.mark.pythonocc_only
def test_pythonocc_tessellate_exposes_only_supported_controls():

    parameters = list(
        inspect.signature(
            Topology.Tessellate
        ).parameters
    )

    assert parameters == [
        "topology",
        "quality",
        "linearDeflection",
        "angularDeflection",
        "relative",
        "parallel",
        "weld",
        "weldTolerance",
        "remesh",
        "mantissa",
        "silent",
    ]
