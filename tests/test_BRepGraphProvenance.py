import os

import pytest

pytest.importorskip("OCC.Core.BRepGraph")

if os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower() != "pythonocc":
    pytest.skip("BRepGraph Tranche 3 tests require the pythonocc backend", allow_module_level=True)

from topologicpy.Cell import Cell
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

from topologicpy.pythonocc_backend.attribute_manager import AttributeManager
from topologicpy.pythonocc_backend._provenance import merge_dictionaries, transfer_by_modifier
from topologicpy.pythonocc_backend.topology import Topology as BackendTopology


@pytest.fixture(autouse=True)
def _clear_attribute_manager_between_tests():
    """Keep native-shape metadata from one provenance test out of the next."""
    manager = AttributeManager.GetInstance()
    manager.ClearAll()
    yield
    manager.ClearAll()


def _value(topology, key, default=None):
    dictionary = Topology.Dictionary(topology, silent=True)
    try:
        return Dictionary.ValueAtKey(dictionary, key, defaultValue=default, silent=True)
    except TypeError:
        try:
            return Dictionary.ValueAtKey(dictionary, key, default)
        except Exception:
            pass
    except Exception:
        pass
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default


def _set(topology, **values):
    dictionary = Dictionary.ByPythonDictionary(values)
    return Topology.SetDictionary(topology, dictionary, silent=True)


def _faces(topology):
    return Topology.Faces(topology, silent=True) or []


def _edges(topology):
    return Topology.Edges(topology, silent=True) or []


def _tag_faces(topology, prefix):
    faces = _faces(topology)
    for index, face in enumerate(faces):
        _set(face, **{f"{prefix}_face": index})
    return faces


def _count_with_key(topologies, key):
    count = 0
    for topology in topologies or []:
        if _value(topology, key, None) is not None:
            count += 1
    return count


def _two_overlapping_cells():
    a = Cell.Prism(
        origin=Vertex.ByCoordinates(0, 0, 0),
        width=2.0,
        length=2.0,
        height=2.0,
        placement="center",
    )
    b = Cell.Prism(
        origin=Vertex.ByCoordinates(0.75, 0, 0),
        width=2.0,
        length=2.0,
        height=2.0,
        placement="center",
    )
    assert Topology.IsInstance(a, "Cell")
    assert Topology.IsInstance(b, "Cell")
    return a, b


def test_provenance_normalizes_pythonocc_attribute_wrappers_to_python_values():
    from topologicpy.pythonocc_backend.attributes import (
        DoubleAttribute,
        IntAttribute,
        ListAttribute,
        StringAttribute,
    )

    merged, conflicts = merge_dictionaries(
        [
            {
                "i": IntAttribute(7),
                "d": DoubleAttribute(2.5),
                "s": StringAttribute("alpha"),
                "items": ListAttribute([IntAttribute(1), StringAttribute("x")]),
            }
        ]
    )

    assert conflicts == 0
    assert merged == {"i": 7, "d": 2.5, "s": "alpha", "items": [1, "x"]}
    assert all(
        not value.__class__.__name__.endswith("Attribute")
        for value in [merged["i"], merged["d"], merged["s"]]
    )


def test_merge_policy_is_explicit_and_first_source_wins_conflicts():
    merged, conflicts = merge_dictionaries(
        [
            {"a": 1, "shared": "A"},
            {"b": 2, "shared": "B"},
            {"c": 3, "shared": "C"},
        ]
    )
    assert merged == {"a": 1, "shared": "A", "b": 2, "c": 3}
    assert conflicts == 2


def test_provenance_result_membership_really_uses_brepgraph():
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec

    cell = Cell.Cube(size=2)
    _set(cell, root="indexed")
    _set(_faces(cell)[0], face_token="indexed-face")

    source_shape = getattr(cell, "shape", None)
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(4.0, 1.0, -2.0))
    maker = BRepBuilderAPI_Transform(source_shape, trsf, True)
    assert maker.IsDone()

    report = transfer_by_modifier(
        source_shape,
        maker.Shape(),
        maker,
        root_dictionary=Topology.Dictionary(cell, silent=True),
        operation="test-index",
    )
    assert report.used_brepgraph_index is True
    assert report.mapped_entities >= 2


def test_brepgraph_membership_resolution_does_not_enumerate_result_subshapes(monkeypatch):
    """OCCT 8 membership must use BRepGraph FindNode, not a linear result scan."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec
    import topologicpy.pythonocc_backend._provenance as provenance

    cell = Cell.Cube(size=2)
    source_face = _faces(cell)[0]
    source_shape = getattr(cell, "shape", None)

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(2.0, 3.0, 4.0))
    maker = BRepBuilderAPI_Transform(source_shape, trsf, True)
    assert maker.IsDone()

    target_face = maker.ModifiedShape(getattr(source_face, "shape", None))
    assert target_face is not None and not target_face.IsNull()

    index = provenance._ResultShapeIndex(maker.Shape())
    assert index.used_brepgraph is True

    def _forbid_enumeration(*args, **kwargs):
        raise AssertionError("BRepGraph membership fell back to result enumeration")

    monkeypatch.setattr(provenance, "_iter_unique_subshapes", _forbid_enumeration)
    assert index.resolve(target_face) is not None


def test_attribute_manager_survives_rewrap_of_same_native_face():
    cell = Cell.Cube(size=2)
    face = _faces(cell)[0]
    _set(face, semantic="north")

    native = getattr(face, "shape", None)
    assert native is not None
    rewrapped = BackendTopology.ByOcctShape(native)

    assert rewrapped is not None
    assert _value(rewrapped, "semantic") == "north"


def test_copy_remains_shallow_but_root_dictionary_is_persistent():
    cell = Cell.Cube(size=2)
    _set(cell, root="copy-root")
    source_face = _faces(cell)[0]
    _set(source_face, face_token="do-not-deep-copy")

    copied = Topology.Copy(cell)
    assert Topology.IsInstance(copied, "Cell")
    assert _value(copied, "root") == "copy-root"

    # Copy's documented contract remains shallow for subtopology metadata.
    assert _count_with_key(_faces(copied), "face_token") == 0

    # Rewrapping the copied root must not lose the root dictionary.
    rewrapped = BackendTopology.ByOcctShape(getattr(copied, "shape", None))
    assert _value(rewrapped, "root") == "copy-root"


def test_deepcopy_preserves_subtopology_dictionary_lineage():
    cell = Cell.Cube(size=2)
    _set(cell, root="deep")
    faces = _faces(cell)
    edges = _edges(cell)
    _set(faces[0], face_token="F0")
    _set(edges[0], edge_token="E0")

    copied = Topology.DeepCopy(cell)
    assert Topology.IsInstance(copied, "Cell")
    assert _value(copied, "root") == "deep"
    assert _count_with_key(_faces(copied), "face_token") == 1
    assert _count_with_key(_edges(copied), "edge_token") == 1


def test_translate_preserves_root_face_and_edge_dictionaries_exactly():
    cell = Cell.Cube(size=2)
    _set(cell, root="translated")
    faces = _faces(cell)
    edges = _edges(cell)
    _set(faces[0], face_token="F0")
    _set(faces[1], face_token="F1")
    _set(edges[0], edge_token="E0")

    moved = Topology.Translate(cell, 7.5, -2.0, 3.25)
    assert Topology.IsInstance(moved, "Cell")
    assert _value(moved, "root") == "translated"
    assert _count_with_key(_faces(moved), "face_token") == 2
    assert _count_with_key(_edges(moved), "edge_token") == 1

    # The propagated dictionaries are registered against native target shapes,
    # not merely mirrored on transient wrappers.
    rewrapped = BackendTopology.ByOcctShape(getattr(moved, "shape", None))
    assert _value(rewrapped, "root") == "translated"
    assert _count_with_key(rewrapped.Faces(), "face_token") == 2
    assert _count_with_key(rewrapped.Edges(), "edge_token") == 1


def test_rotate_scale_and_transform_preserve_subtopology_dictionaries():
    operations = [
        lambda topology: Topology.Rotate(
            topology,
            origin=Vertex.Origin(),
            axis=[0, 0, 1],
            angle=31.0,
            transferDictionaries=True,
            silent=True,
        ),
        lambda topology: Topology.Scale(
            topology,
            origin=Vertex.Origin(),
            x=1.25,
            y=0.8,
            z=1.4,
            transferDictionaries=True,
            silent=True,
        ),
        lambda topology: Topology.Transform(
            topology,
            matrix=[
                [1.1, 0.2, 0.0, 2.5],
                [0.0, 0.9, 0.1, -1.0],
                [0.0, 0.0, 1.2, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            transferDictionaries=True,
            silent=True,
        ),
    ]

    for index, operation in enumerate(operations):
        cell = Cell.Cube(size=2)
        _set(cell, root=f"operation-{index}")
        faces = _faces(cell)
        edges = _edges(cell)
        _set(faces[0], face_token=f"F{index}")
        _set(edges[0], edge_token=f"E{index}")

        result = operation(cell)
        assert Topology.IsInstance(result, "Cell")
        assert _value(result, "root") == f"operation-{index}"
        assert _count_with_key(_faces(result), "face_token") == 1
        assert _count_with_key(_edges(result), "edge_token") == 1

        rewrapped = BackendTopology.ByOcctShape(getattr(result, "shape", None))
        assert _value(rewrapped, "root") == f"operation-{index}"
        assert _count_with_key(rewrapped.Faces(), "face_token") == 1
        assert _count_with_key(rewrapped.Edges(), "edge_token") == 1


def test_transform_transfer_false_does_not_propagate_new_shape_metadata():
    operations = [
        lambda topology: Topology.Translate(
            topology, 3.0, -1.0, 2.0, transferDictionaries=False, silent=True
        ),
        lambda topology: Topology.Rotate(
            topology,
            origin=Vertex.Origin(),
            axis=[0, 0, 1],
            angle=17.0,
            transferDictionaries=False,
            silent=True,
        ),
        lambda topology: Topology.Scale(
            topology,
            origin=Vertex.Origin(),
            x=1.15,
            y=0.85,
            z=1.25,
            transferDictionaries=False,
            silent=True,
        ),
        lambda topology: Topology.Transform(
            topology,
            matrix=[
                [1.0, 0.15, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            transferDictionaries=False,
            silent=True,
        ),
    ]

    for operation in operations:
        cell = Cell.Cube(size=2)
        _set(cell, root="must-not-propagate")
        _set(_faces(cell)[0], face_token="must-not-propagate")
        _set(_edges(cell)[0], edge_token="must-not-propagate")

        result = operation(cell)
        assert Topology.IsInstance(result, "Cell")
        assert _value(result, "root", None) is None
        assert _count_with_key(_faces(result), "face_token") == 0
        assert _count_with_key(_edges(result), "edge_token") == 0


def test_difference_root_follows_self_while_cut_interface_can_follow_tool():
    a, b = _two_overlapping_cells()
    _set(a, a_root=True, shared="A")
    _set(b, b_root=True, shared="B")
    _tag_faces(a, "a")
    _tag_faces(b, "b")

    result = Topology.Difference(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")

    # Difference is asymmetric: the result root continues self, not the tool.
    assert bool(_value(result, "a_root")) is True
    assert _value(result, "b_root", None) is None
    assert _value(result, "shared") == "A"

    result_faces = _faces(result)
    assert _count_with_key(result_faces, "a_face") >= 1
    # At least one newly-created cut/interface face should carry exact lineage
    # from the cutting tool's source Face.
    assert _count_with_key(result_faces, "b_face") >= 1


def test_difference_without_transfer_does_not_fabricate_tool_metadata():
    a, b = _two_overlapping_cells()
    tool_faces = _tag_faces(b, "b")

    result = Topology.Difference(a, b, tranDict=False, silent=True)
    assert Topology.IsInstance(result, "Topology")

    # transfer=False does not erase metadata already attached to an unchanged
    # OCCT face reused by the result.  It must, however, never copy the tool's
    # dictionary onto a newly-created result face.
    tagged = [face for face in _faces(result) if _value(face, "b_face", None) is not None]
    for face in tagged:
        assert any(Topology.IsSame(face, source_face) for source_face in tool_faces)


def test_union_root_merge_is_first_source_wins_and_faces_keep_lineage():
    a, b = _two_overlapping_cells()
    _set(a, a_root=1, shared="A")
    _set(b, b_root=2, shared="B")
    _tag_faces(a, "a")
    _tag_faces(b, "b")

    result = Topology.Union(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")

    assert _value(result, "a_root") == 1
    assert _value(result, "b_root") == 2
    assert _value(result, "shared") == "A"

    result_faces = _faces(result)
    assert _count_with_key(result_faces, "a_face") >= 1
    assert _count_with_key(result_faces, "b_face") >= 1


def test_slice_root_follows_self_and_transfers_exact_face_lineage():
    a, b = _two_overlapping_cells()
    _set(a, a_root="slice-source", shared="A")
    _set(b, b_root="slice-tool", shared="B")
    _tag_faces(a, "a")
    _tag_faces(b, "b")

    result = Topology.Slice(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")
    assert _value(result, "a_root") == "slice-source"
    assert _value(result, "b_root", None) is None
    assert _value(result, "shared") == "A"
    assert _count_with_key(_faces(result), "a_face") >= 1


def test_native_result_dictionaries_are_registered_in_attribute_manager():
    a, b = _two_overlapping_cells()
    _set(a, provenance="A")
    result = Topology.Difference(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")

    shape = getattr(result, "shape", None)
    manager = AttributeManager.GetInstance()
    assert shape is not None
    assert manager.HasDictionary(shape)
    stored = manager.GetDictionary(shape)
    assert Dictionary.ValueAtKey(stored, "provenance") == "A"



def test_pythonocc_public_boolean_skips_legacy_geometric_dictionary_transfer(monkeypatch):
    a, b = _two_overlapping_cells()
    _set(a, owner="A")
    _set(_faces(a)[0], face_token="A-face")

    def _legacy_transfer_must_not_run(*args, **kwargs):
        raise AssertionError("legacy Topology.TransferDictionaries was called")

    monkeypatch.setattr(
        Topology,
        "TransferDictionaries",
        staticmethod(_legacy_transfer_must_not_run),
    )

    result = Topology.Difference(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")
    assert _value(result, "owner") == "A"


def test_merge_preserves_each_source_cell_dictionary_on_derived_cells():
    # Two cells sharing one face: Merge should create a CellComplex whose
    # constituent Cells retain their own semantic dictionaries.
    a = Cell.Prism(
        origin=Vertex.ByCoordinates(-0.5, 0, 0),
        width=1.0,
        length=2.0,
        height=2.0,
        placement="center",
    )
    b = Cell.Prism(
        origin=Vertex.ByCoordinates(0.5, 0, 0),
        width=1.0,
        length=2.0,
        height=2.0,
        placement="center",
    )
    _set(a, cell_id="A", shared="A")
    _set(b, cell_id="B", shared="B")

    result = Topology.Merge(a, b, tranDict=True, silent=True)
    assert Topology.IsInstance(result, "Topology")

    cells = Topology.Cells(result, silent=True) or []
    ids = sorted(
        value
        for value in (_value(cell, "cell_id", None) for cell in cells)
        if value is not None
    )
    assert ids == ["A", "B"]

    # The aggregate/root policy is separately deterministic and first-source wins.
    assert _value(result, "shared") == "A"
