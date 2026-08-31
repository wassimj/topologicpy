import pytest

from topologicpy.Aperture import Aperture
from topologicpy.Cell import Cell
from topologicpy.Content import Content
from topologicpy.Context import Context
from topologicpy.SemanticManager import SemanticManager
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


@pytest.fixture(autouse=True)
def _reset_semantics():
    SemanticManager.Reset()
    yield
    SemanticManager.Reset()


def _same_in(items, target):
    return any(Topology.IsSame(item, target, silent=True) for item in (items or []))


def _top_face(cell):
    faces = Topology.Faces(cell, silent=True)
    return max(faces, key=lambda face: Vertex.Z(Topology.Centroid(face, silent=True)))


def test_invalid_aperture_topology_is_safe():
    assert Aperture.Topology(None) is None
    assert Aperture.Topology("not an aperture") is None


def test_semantics_are_backend_independent():
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
    chair = Vertex.ByCoordinates(0, 0, 0.5)

    Topology.AddContent(room, chair, silent=True)
    Topology.AddApertures(face, [window], silent=True)
    Topology.AddApertures(room, [window], silent=True)

    assert _same_in(Topology.Contents(room, silent=True), chair)
    assert _same_in(Topology.Contents(room, silent=True), window)
    assert _same_in(Topology.Apertures(face, silent=True), window)
    assert _same_in(Topology.Apertures(room, silent=True), window)

    semantic_window = SemanticManager.GetInstance().content_for_topology(window, create=False)
    assert isinstance(semantic_window, Aperture)
    assert isinstance(semantic_window, Content)
    assert Topology.IsInstance(semantic_window, "Aperture")
    assert Topology.IsInstance(semantic_window, "Content")

    contexts = Topology.Contexts(window, silent=True)
    assert len(contexts) == 2
    assert all(isinstance(context, Context) for context in contexts)

    Topology.RemoveContent(room, [window], silent=True)
    assert len(Topology.Contexts(window, silent=True)) == 1
    assert not _same_in(Topology.Apertures(room, silent=True), window)
    assert _same_in(Topology.Apertures(face, silent=True), window)


def test_legacy_context_factory_is_semantic_on_both_backends():
    room = Cell.Prism(width=2, length=2, height=2, silent=True)
    face = _top_face(room)
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
    aperture = Aperture.ByTopologyContext(window, context)

    assert Topology.IsInstance(context, "Context")
    assert Topology.IsInstance(aperture, "Aperture")
    assert Topology.IsInstance(aperture, "Content")
    assert context.Content() is aperture
    assert context.Parameters() == {"u": 0.2, "v": 0.3, "w": 0.0}
    assert Topology.IsSame(Context.Topology(context), face, silent=True)
    assert Topology.IsSame(Aperture.Topology(aperture), window, silent=True)
