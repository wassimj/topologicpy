import pytest

from topologicpy.Core import Core
from topologicpy.Dictionary import Dictionary
from topologicpy.Face import Face
from topologicpy.Topology import Topology


def _require_pythonocc():
    backend_name = Core.Backend().__class__.__name__.lower()
    if "pythonocc" not in backend_name:
        pytest.skip("PythonOCC dictionary-fidelity regression test")


def _value_at(topology, key):
    dictionary = Topology.Dictionary(topology, silent=True)
    try:
        return Dictionary.ValueAtKey(dictionary, key, None, silent=True)
    except TypeError:
        try:
            return Dictionary.ValueAtKey(dictionary, key, None)
        except TypeError:
            return Dictionary.ValueAtKey(dictionary, key)


def test_pythonocc_topology_dictionary_preserves_nested_python_values():
    _require_pythonocc()

    face = Face.Rectangle(width=2.0, length=1.0, silent=True)
    payload = {
        "nested": {
            "exact": True,
            "revision": 2,
            "tuple_value": (1, "a", False),
            "list_value": [1, 2.5, None, {"deep": True}],
        },
        "flag": False,
    }

    result = Topology.SetDictionary(face, payload, silent=True)
    assert Topology.IsSame(result, face, silent=True)

    stored = Topology.Dictionary(face, silent=True)
    assert isinstance(stored, dict)
    assert stored == payload
    assert _value_at(face, "nested") == payload["nested"]
    assert _value_at(face, "flag") is False

    # Fresh Python wrapper around the same OCCT shape must resolve the same
    # shape-keyed Python dictionary from the backend AttributeManager.
    shape = Topology.OCCTShape(face, silent=True)
    rebuilt = Topology.ByOCCTShape(shape, silent=True)
    rebuilt_dictionary = Topology.Dictionary(rebuilt, silent=True)

    assert isinstance(rebuilt_dictionary, dict)
    assert rebuilt_dictionary == payload
    assert _value_at(rebuilt, "nested") == payload["nested"]
    assert _value_at(rebuilt, "flag") is False
