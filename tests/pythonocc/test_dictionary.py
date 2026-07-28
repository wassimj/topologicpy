# Copyright (C) 2026
# PythonOCC backend Dictionary parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_dictionary.py -v

import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary
IntAttribute = pytest.importorskip("topologicpy.IntAttribute").IntAttribute
DoubleAttribute = pytest.importorskip("topologicpy.DoubleAttribute").DoubleAttribute
StringAttribute = pytest.importorskip("topologicpy.StringAttribute").StringAttribute
ListAttribute = pytest.importorskip("topologicpy.ListAttribute").ListAttribute

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


# ===========================================================================
# Constructors
# ===========================================================================

class TestDictionaryConstructors:
    def test_by_keys_values(self):
        d = Dictionary.ByKeysValues(["a", "b", "c"], [1, 2, 3])
        assert d is not None

    def test_by_keys_values_strings(self):
        d = Dictionary.ByKeysValues(["name", "type"], ["test", "example"])
        assert d is not None

    def test_empty_dictionary(self):
        d = Dictionary.ByKeysValues([], [])
        assert d is not None


# ===========================================================================
# Accessors
# ===========================================================================

class TestDictionaryAccessors:
    def test_keys(self):
        d = Dictionary.ByKeysValues(["x", "y", "z"], [1, 2, 3])
        keys = Dictionary.Keys(d)
        assert set(keys) == {"x", "y", "z"}

    def test_values(self):
        d = Dictionary.ByKeysValues(["x", "y"], [10, 20])
        vals = Dictionary.Values(d)
        assert 10 in vals
        assert 20 in vals

    def test_value_at_key(self):
        d = Dictionary.ByKeysValues(["name"], ["hello"])
        val = Dictionary.ValueAtKey(d, "name")
        assert val == "hello"

    def test_value_at_key_int(self):
        d = Dictionary.ByKeysValues(["count"], [42])
        val = Dictionary.ValueAtKey(d, "count")
        assert val == 42

    def test_value_at_key_float(self):
        d = Dictionary.ByKeysValues(["pi"], [3.14159])
        val = Dictionary.ValueAtKey(d, "pi")
        assert val == pytest.approx(3.14159, abs=TOLERANCE)

    def test_nonexistent_key(self):
        d = Dictionary.ByKeysValues(["a"], [1])
        val = Dictionary.ValueAtKey(d, "nonexistent")
        assert val is None

    def test_contains_key(self):
        d = Dictionary.ByKeysValues(["a", "b"], [1, 2])
        assert Dictionary.ContainsKey(d, "a") is True
        assert Dictionary.ContainsKey(d, "c") is False


# ===========================================================================
# Attribute types
# ===========================================================================

class TestDictionaryAttributes:
    def test_int_attribute(self):
        attr = IntAttribute(42)
        assert attr is not None
        d = Dictionary.ByKeysValues(["n"], [attr])
        val = Dictionary.ValueAtKey(d, "n")
        assert val == 42

    def test_double_attribute(self):
        attr = DoubleAttribute(3.14)
        assert attr is not None
        d = Dictionary.ByKeysValues(["pi"], [attr])
        val = Dictionary.ValueAtKey(d, "pi")
        assert val == pytest.approx(3.14, abs=TOLERANCE)

    def test_string_attribute(self):
        attr = StringAttribute("hello")
        assert attr is not None
        d = Dictionary.ByKeysValues(["greeting"], [attr])
        val = Dictionary.ValueAtKey(d, "greeting")
        assert val == "hello"

    def test_list_attribute(self):
        attr = ListAttribute([1, 2, 3])
        assert attr is not None
        d = Dictionary.ByKeysValues(["nums"], [attr])
        val = Dictionary.ValueAtKey(d, "nums")
        assert val == [1, 2, 3]


# ===========================================================================
# Topology integration
# ===========================================================================

class TestDictionaryTopology:
    def test_set_on_vertex(self):
        v = Vertex.ByCoordinates(1, 2, 3)
        d = Dictionary.ByKeysValues(["label"], ["point"])
        v2 = Topology.SetDictionary(v, d)
        d2 = Topology.Dictionary(v2)
        assert Dictionary.ValueAtKey(d2, "label") == "point"

    def test_preserves_geometry(self):
        v = Vertex.ByCoordinates(1, 2, 3)
        d = Dictionary.ByKeysValues(["key"], ["value"])
        v2 = Topology.SetDictionary(v, d)
        assert Vertex.X(v2) == pytest.approx(1, abs=TOLERANCE)
        assert Vertex.Y(v2) == pytest.approx(2, abs=TOLERANCE)
        assert Vertex.Z(v2) == pytest.approx(3, abs=TOLERANCE)


# ===========================================================================
# Serialization
# ===========================================================================

class TestDictionarySerialization:
    def test_brep_roundtrip(self):
        d = Dictionary.ByKeysValues(["a", "b"], [1, "hello"])
        brep = Topology.BREPString(d)
        assert brep is not None
        d2 = Topology.ByBREPString(brep)
        assert Dictionary.ValueAtKey(d2, "a") == 1
        assert Dictionary.ValueAtKey(d2, "b") == "hello"
