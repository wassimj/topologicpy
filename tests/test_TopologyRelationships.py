"""Regression tests for TopologicPy topology relationship persistence.

These tests specifically guard against relationships being stored only on a
transient Python wrapper. A subtopology is obtained from a parent, content or
apertures are attached, and then the parent is traversed again to obtain a
fresh wrapper for the same underlying OCCT subshape.

The relationship must still be present after that re-traversal.
"""

import pytest

Cell = pytest.importorskip("topologicpy.Cell").Cell
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex


TOLERANCE = 0.001


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _box():
    """Return a simple 2 x 2 x 2 cell centred at the origin."""
    origin = Vertex.ByCoordinates(0, 0, 0)
    cell = Cell.Box(
        origin=origin,
        width=2,
        length=2,
        height=2,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _top_face(cell):
    """Return the horizontal face with the greatest centroid Z coordinate."""
    faces = Topology.Faces(cell, silent=True)
    assert isinstance(faces, list)
    assert len(faces) == 6

    return max(
        faces,
        key=lambda face: Vertex.Z(Topology.Centroid(face)),
    )


def _same_face_from_parent(cell, reference_face):
    """Re-traverse the parent and recover the same underlying face."""
    faces = Topology.Faces(cell, silent=True)
    matches = [
        face
        for face in faces
        if Topology.IsSame(face, reference_face, silent=True)
    ]

    assert len(matches) == 1
    return matches[0]


def _inset_face(face, scale=0.25, x=0.0, y=0.0):
    """Create a smaller coplanar face inside the input horizontal face."""
    centroid = Topology.Centroid(face)

    inset = Topology.Scale(
        face,
        origin=centroid,
        x=scale,
        y=scale,
        z=1.0,
        transferDictionaries=False,
        silent=True,
    )
    assert Topology.IsInstance(inset, "Face")

    if x != 0.0 or y != 0.0:
        inset = Topology.Translate(
            inset,
            x=x,
            y=y,
            z=0.0,
            transferDictionaries=False,
            silent=True,
        )
        assert Topology.IsInstance(inset, "Face")

    return inset


def _aperture_type(topology):
    """Return the value of the TopologicPy aperture marker, if present."""
    dictionary = Topology.Dictionary(topology)
    return Dictionary.ValueAtKey(dictionary, "type", None)


def test_add_content_to_face_survives_parent_retraversal():
    """Core regression: content attached to a face must survive a fresh wrapper."""
    cell = _box()
    face_before = _top_face(cell)
    content = _inset_face(face_before)

    returned_cell = Topology.AddContent(
        cell,
        content,
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsSame(returned_cell, cell, silent=True)

    face_after = _same_face_from_parent(cell, face_before)

    # This is the invariant that the old PythonOCC wrapper-local implementation
    # violated.
    contents = Topology.Contents(face_after, silent=True)

    assert isinstance(contents, list)
    assert len(contents) == 1
    assert Topology.IsInstance(contents[0], "Face")
    assert Face.Area(contents[0]) == pytest.approx(
        Face.Area(content),
        abs=1e-6,
    )


def test_add_aperture_to_self_is_retrievable():
    """An aperture added directly to a topology must be returned by Apertures."""
    host = Face.Rectangle(width=2, length=2, silent=True)
    aperture = Face.Rectangle(width=0.5, length=0.5, silent=True)

    host = Topology.AddApertures(
        host,
        [aperture],
        tolerance=TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(host, silent=True)

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_aperture_type(apertures[0])).lower() == "aperture"


def test_add_aperture_to_face_survives_parent_retraversal():
    """Main regression: an aperture on a cell face must survive re-traversal."""
    cell = _box()
    face_before = _top_face(cell)
    aperture = _inset_face(face_before)

    returned_cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    assert Topology.IsSame(returned_cell, cell, silent=True)

    # Force a fresh traversal of the cell faces rather than keeping the wrapper
    # that AddApertures originally modified.
    face_after = _same_face_from_parent(cell, face_before)

    apertures = Topology.Apertures(face_after, silent=True)

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_aperture_type(apertures[0])).lower() == "aperture"


def test_parent_can_collect_apertures_from_face_subtopologies():
    """The parent-level convenience query must find face-hosted apertures."""
    cell = _box()
    target_face = _top_face(cell)
    aperture = _inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 1
    assert Topology.IsInstance(apertures[0], "Face")
    assert str(_aperture_type(apertures[0])).lower() == "aperture"


def test_two_apertures_on_same_face_when_not_exclusive():
    """exclusive=False must allow multiple apertures on one subtopology."""
    cell = _box()
    target_face = _top_face(cell)

    aperture_a = _inset_face(target_face, scale=0.20, x=-0.45)
    aperture_b = _inset_face(target_face, scale=0.20, x=0.45)

    cell = Topology.AddApertures(
        cell,
        [aperture_a, aperture_b],
        exclusive=False,
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 2
    assert all(
        str(_aperture_type(aperture)).lower() == "aperture"
        for aperture in apertures
    )


def test_exclusive_allows_only_one_aperture_per_face():
    """exclusive=True must identify a face by topology identity, not wrapper id."""
    cell = _box()
    target_face = _top_face(cell)

    aperture_a = _inset_face(target_face, scale=0.20, x=-0.45)
    aperture_b = _inset_face(target_face, scale=0.20, x=0.45)

    cell = Topology.AddApertures(
        cell,
        [aperture_a, aperture_b],
        exclusive=True,
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    apertures = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert isinstance(apertures, list)
    assert len(apertures) == 1


def test_aperture_is_not_attached_to_unrelated_faces():
    """Only the geometrically matching face should receive the aperture."""
    cell = _box()
    target_face = _top_face(cell)
    aperture = _inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    faces = Topology.Faces(cell, silent=True)
    counts = [
        len(Topology.Apertures(face, silent=True))
        for face in faces
    ]

    assert sorted(counts) == [0, 0, 0, 0, 0, 1]


def test_repeated_aperture_queries_do_not_duplicate_relationships():
    """Hydrating a fresh wrapper repeatedly must not create duplicate links."""
    cell = _box()
    target_face = _top_face(cell)
    aperture = _inset_face(target_face)

    cell = Topology.AddApertures(
        cell,
        [aperture],
        subTopologyType="face",
        tolerance=TOLERANCE,
        silent=True,
    )

    first = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )
    second = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )
    third = Topology.Apertures(
        cell,
        subTopologyType="face",
        silent=True,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert len(third) == 1
