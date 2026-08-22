import pytest

from topologicpy.Core import Core
from topologicpy.Cell import Cell
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def _require_pythonocc():
    backend_name = Core.Backend().__class__.__name__.lower()
    if "pythonocc" not in backend_name:
        pytest.skip("PythonOCC backend regression test")


def test_content_survives_fresh_wrapper_for_same_occ_shape():
    _require_pythonocc()

    cell = Cell.Prism()
    face_a = Topology.Faces(cell, silent=True)[0]

    content = Vertex.ByCoordinates(10, 20, 30)
    dictionary = Dictionary.ByKeysValues(
        ["relationship_test"],
        ["persistent_content"]
    )
    content = Topology.SetDictionary(content, dictionary)

    face_a = Topology.AddContent(
        face_a,
        [content],
        subTopologyType="self",
        silent=False
    )

    shape = Core.InstanceCall(face_a, "GetOcctShape")
    face_b = Core.Topology.ByOcctShape(shape)

    assert face_b is not face_a
    assert Topology.IsSame(face_a, face_b, silent=True)

    contents = Topology.Contents(face_b, silent=False)
    assert len(contents) == 1
    assert Dictionary.ValueAtKey(
        Topology.Dictionary(contents[0]),
        "relationship_test"
    ) == "persistent_content"


def test_context_survives_fresh_wrapper_for_same_occ_shape():
    _require_pythonocc()

    cell = Cell.Prism()
    face = Topology.Faces(cell, silent=True)[0]

    content = Vertex.ByCoordinates(1, 2, 3)
    face = Topology.AddContent(
        face,
        [content],
        subTopologyType="self",
        silent=False
    )

    stored_content = Topology.Contents(face, silent=False)[0]
    shape = Core.InstanceCall(stored_content, "GetOcctShape")
    rebuilt_content = Core.Topology.ByOcctShape(shape)

    contexts = Topology.Contexts(rebuilt_content, silent=False)
    assert len(contexts) == 1

    context_topology = Core.InstanceCall(contexts[0], "Topology")
    assert Topology.IsSame(context_topology, face, silent=True)


def test_add_apertures_to_face_subtopology_survives_parent_retraversal():
    _require_pythonocc()

    cell = Cell.Prism()
    target_face = Topology.Faces(cell, silent=True)[0]
    centroid = Topology.Centroid(target_face)

    aperture = Topology.Scale(
        target_face,
        origin=centroid,
        x=0.5,
        y=0.5,
        z=0.5,
        transferDictionaries=False,
        silent=False
    )

    assert Topology.IsInstance(aperture, "Face")

    result = Topology.AddApertures(
        cell,
        [aperture],
        exclusive=False,
        subTopologyType="face",
        tolerance=0.001,
        silent=False
    )

    assert Topology.IsInstance(result, "Cell")

    apertures = Topology.Apertures(
        result,
        subTopologyType="face",
        silent=False
    )

    assert len(apertures) == 1
    aperture_type = Dictionary.ValueAtKey(
        Topology.Dictionary(apertures[0]),
        "type",
        ""
    )
    assert str(aperture_type).lower() == "aperture"


def test_exclusive_uses_topological_identity_not_python_wrapper_identity():
    _require_pythonocc()

    cell = Cell.Prism()
    target_face = Topology.Faces(cell, silent=True)[0]
    centroid = Topology.Centroid(target_face)

    aperture_a = Topology.Scale(
        target_face,
        origin=centroid,
        x=0.5,
        y=0.5,
        z=0.5,
        transferDictionaries=False,
        silent=False
    )

    aperture_b = Topology.Scale(
        target_face,
        origin=centroid,
        x=0.25,
        y=0.25,
        z=0.25,
        transferDictionaries=False,
        silent=False
    )

    result = Topology.AddApertures(
        cell,
        [aperture_a, aperture_b],
        exclusive=True,
        subTopologyType="face",
        tolerance=0.001,
        silent=False
    )

    apertures = Topology.Apertures(
        result,
        subTopologyType="face",
        silent=False
    )

    assert len(apertures) == 1
