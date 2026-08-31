import pytest

from topologicpy.Aperture import Aperture
from topologicpy.Cell import Cell
from topologicpy.Content import Content
from topologicpy.Context import Context
from topologicpy.SemanticManager import SemanticManager
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


@pytest.fixture(autouse=True)
def _fresh_semantic_manager():
    SemanticManager.Reset()
    yield
    SemanticManager.Reset()


def _same_in(items, target):
    return any(Topology.IsSame(item, target, silent=True) for item in items)


def _top_face(cell):
    faces = Topology.Faces(cell, silent=True)
    assert faces
    return max(faces, key=lambda f: Vertex.Z(Topology.Centroid(f, silent=True)))


def test_aperture_is_content_with_multiple_contexts():
    room = Cell.Prism(width=4, length=4, height=3, silent=True)
    host_face = _top_face(room)
    centre = Topology.Centroid(host_face, silent=True)
    window = Topology.Scale(
        host_face,
        origin=centre,
        x=0.25,
        y=0.25,
        z=1.0,
        transferDictionaries=False,
        silent=True,
    )
    chair = Vertex.ByCoordinates(0.0, 0.0, 0.5)

    assert Topology.IsInstance(room, "Cell")
    assert Topology.IsInstance(window, "Face")
    assert Topology.IsInstance(chair, "Vertex")

    # Furniture is ordinary Content of the room.
    assert Topology.AddContent(room, chair, silent=True) is room

    # The *same* window Aperture has two Contexts: its host Face and its room.
    assert Topology.AddApertures(host_face, [window], silent=True) is host_face
    assert Topology.AddApertures(room, [window], silent=True) is room

    room_contents = Topology.Contents(room, silent=True)
    room_apertures = Topology.Apertures(room, silent=True)
    face_apertures = Topology.Apertures(host_face, silent=True)

    assert _same_in(room_contents, chair)
    assert _same_in(room_contents, window)
    assert _same_in(room_apertures, window)
    assert _same_in(face_apertures, window)

    # Apertures(host) is a true subset of Contents(host).
    assert all(_same_in(room_contents, aperture) for aperture in room_apertures)

    semantic_window = SemanticManager.GetInstance().content_for_topology(
        window, create=False
    )
    assert isinstance(semantic_window, Aperture)
    assert isinstance(semantic_window, Content)

    contexts = Topology.Contexts(window, silent=True)
    assert len(contexts) == 2
    assert all(isinstance(context, Context) for context in contexts)
    assert all(context.Content() is semantic_window for context in contexts)
    assert all(context.Parameters() is None for context in contexts)

    hosts = [context.Host() for context in contexts]
    assert _same_in(hosts, host_face)
    assert _same_in(hosts, room)

    # A fresh wrapper around the same OCCT Face must resolve the same relation.
    host_face_fresh = next(
        face
        for face in Topology.Faces(room, silent=True)
        if Topology.IsSame(face, host_face, silent=True)
    )
    assert _same_in(Topology.Apertures(host_face_fresh, silent=True), window)


def test_legacy_context_factory_can_bind_an_aperture_with_parameters():
    host = Cell.Prism(width=2, length=2, height=2, silent=True)
    face = _top_face(host)
    window = Topology.Scale(
        face,
        origin=Topology.Centroid(face, silent=True),
        x=0.5,
        y=0.5,
        z=1.0,
        transferDictionaries=False,
        silent=True,
    )

    context = Context.ByTopologyParameters(face, u=0.2, v=0.3, w=0.0)
    assert Topology.IsInstance(context, "Context")

    aperture = Aperture.ByTopologyContext(window, context)
    assert Topology.IsInstance(aperture, "Aperture")
    assert Topology.IsInstance(aperture, "Content")
    assert Topology.IsSame(Aperture.Topology(aperture), window, silent=True)
    assert Topology.IsSame(Context.Topology(context), face, silent=True)
    assert context.Content() is aperture
    assert context.Parameters() == {"u": 0.2, "v": 0.3, "w": 0.0}


def test_removing_one_context_does_not_destroy_other_contexts():
    room = Cell.Prism(width=4, length=4, height=3, silent=True)
    face = _top_face(room)
    window = Topology.Scale(
        face,
        origin=Topology.Centroid(face, silent=True),
        x=0.25,
        y=0.25,
        z=1.0,
        transferDictionaries=False,
        silent=True,
    )

    Topology.AddApertures(face, [window], silent=True)
    Topology.AddApertures(room, [window], silent=True)
    assert len(Topology.Contexts(window, silent=True)) == 2

    Topology.RemoveContent(room, [window], silent=True)

    assert not _same_in(Topology.Contents(room, silent=True), window)
    assert not _same_in(Topology.Apertures(room, silent=True), window)
    assert _same_in(Topology.Apertures(face, silent=True), window)
    assert len(Topology.Contexts(window, silent=True)) == 1
