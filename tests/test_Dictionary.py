"""Unit tests for topologicpy.Dictionary.

These tests are intentionally deterministic and quiet under pytest. They avoid self-skipping constructs so the CI result is either pass or fail from this file.
"""

import pytest

from topologicpy.Dictionary import Dictionary


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _py(dictionary):
    """Return a Python-dictionary view of any supported Dictionary result."""
    return Dictionary.PythonDictionary(dictionary, silent=True)



def test_plain_python_dictionary_accessors():
    d = {"a": 1, "b": "two", "none": None}

    assert Dictionary.ValueAtKey(d, "a", silent=True) == 1
    assert Dictionary.ValueAtKey(d, "missing", defaultValue="fallback", silent=True) == "fallback"
    assert Dictionary.ValueAtKey(d, "none", defaultValue="fallback", silent=True) is None
    assert Dictionary.Keys(d, silent=True) == ["a", "b", "none"]
    assert Dictionary.Values(d, silent=True) == [1, "two", None]
    assert Dictionary.ValuesAtKeys(d, ["a", "missing"], defaultValue=99, silent=True) == [1, 99]
    assert Dictionary.KeysAtValue(
        {"x": "alpha", "y": "beta", "z": "alpha"},
        "alpha",
        silent=True,
    ) == ["x", "z"]


def test_invalid_dictionary_accessors_return_none_or_defaults():
    assert Dictionary.Keys(None, silent=True) is None
    assert Dictionary.Values(None, silent=True) is None
    assert Dictionary.ValuesAtKeys(None, ["a"], silent=True) is None
    assert Dictionary.ValuesAtKeys({"a": 1}, "a", silent=True) is None
    assert Dictionary.ValuesAtKeys({"a": 1}, ["a", 2], silent=True) is None
    assert Dictionary.ValueAtKey({"a": 1}, 3, defaultValue="default", silent=True) == "default"
    assert Dictionary.ValueAtKey(None, "a", defaultValue="default", silent=True) == "default"
    assert Dictionary.PythonDictionary(None, silent=True) is None










def test_by_key_value_and_by_keys_values_round_trip():
    d1 = Dictionary.ByKeyValue("height", 3.5, silent=True)
    assert _py(d1) == {"height": pytest.approx(3.5)}

    d2 = Dictionary.ByKeysValues(
        ["name", "count", "ratio", "items", "none"],
        ["Room", 4, 0.25, ["a", 2, None], None],
        silent=True,
    )
    py = _py(d2)
    assert py["name"] == "Room"
    assert py["count"] == 4
    assert py["ratio"] == pytest.approx(0.25)
    assert py["items"] == ["a", 2, None]
    assert py["none"] is None


def test_by_key_value_and_by_keys_values_invalid_inputs():
    assert Dictionary.ByKeyValue(1, "value", silent=True) is None
    assert Dictionary.ByKeysValues("a", [1], silent=True) is None
    assert Dictionary.ByKeysValues(["a"], "value", silent=True) is None
    assert Dictionary.ByKeysValues(["a"], [1, 2], silent=True) is None


def test_by_python_dictionary_copy_and_round_trip():
    original = {"a": 1, "b": "two"}
    d = Dictionary.ByPythonDictionary(original, silent=True)
    assert _py(d) == original

    copied = Dictionary.Copy(original, silent=True)
    original["a"] = 99
    assert _py(copied) == {"a": 1, "b": "two"}
    assert Dictionary.ByPythonDictionary(None, silent=True) is None
    assert Dictionary.Copy(None, silent=True) is None


def test_set_value_set_values_and_remove_key_on_python_dictionary():
    d = {"Name": "Old", "keep": True}

    assert Dictionary.SetValueAtKey(d, "Name", "New", silent=True) is d
    assert d["Name"] == "New"

    assert Dictionary.SetValuesAtKeys(d, ["a", 5], [1, 2], silent=True) is d
    assert d["a"] == 1
    assert d["5"] == 2

    assert Dictionary.RemoveKey(d, "name", caseSensitive=False, silent=True) is d
    assert "Name" not in d
    assert Dictionary.RemoveKey(d, "missing", silent=True) is d


def test_set_and_remove_invalid_inputs():
    assert Dictionary.SetValueAtKey(None, "a", 1, silent=True) is None
    assert Dictionary.SetValueAtKey({}, 5, 1, silent=True) is None
    assert Dictionary.SetValuesAtKeys(None, ["a"], [1], silent=True) is None
    assert Dictionary.SetValuesAtKeys({}, "a", [1], silent=True) is None
    assert Dictionary.SetValuesAtKeys({}, ["a"], "1", silent=True) is None
    assert Dictionary.SetValuesAtKeys({}, ["a"], [1, 2], silent=True) is None
    assert Dictionary.RemoveKey(None, "a", silent=True) is None
    assert Dictionary.RemoveKey({}, 5, silent=True) is None


def test_by_merged_dictionaries_preserves_none_and_merges_conflicts():
    d = Dictionary.ByMergedDictionaries(
        {"a": None, "b": "", "c": 1, "d": 2},
        {"a": 10, "b": "value", "c": 1, "d": 3, "e": 4},
        silent=True,
    )
    py = _py(d)
    assert py["a"] == 10
    assert py["b"] == "value"
    assert py["c"] == 1
    assert py["d"] == [2, 3]
    assert py["e"] == 4
    assert Dictionary.ByMergedDictionaries(None, silent=True) is None


def test_union_and_merge_behaviour():
    out = _py(Dictionary.Union({"a": 1, "b": 2}, {"b": 3, "c": 4}, silent=True))
    assert out == {"a": 1, "b": [2, 3], "c": 4}

    merged = _py(Dictionary.Merge({"a": 1}, {"b": 2}, silent=True))
    assert merged == {"a": 1, "b": 2}

    assert _py(Dictionary.Union(None, {"b": 2}, silent=True)) == {"b": 2}
    assert _py(Dictionary.Union({"a": 1}, None, silent=True)) == {"a": 1}
    assert Dictionary.Union(None, None, silent=True) is None


def test_difference_intersection_and_symmetric_difference_behaviour():
    assert _py(Dictionary.Difference({"a": 1, "b": 2}, {"b": 9}, silent=True)) == {"a": 1}
    assert _py(Dictionary.Difference({"a": 1}, None, silent=True)) == {"a": 1}
    assert Dictionary.Difference(None, {"a": 1}, silent=True) is None

    same = _py(Dictionary.Intersection({"a": 1, "b": 2}, {"b": 2, "c": 3}, silent=True))
    different = _py(Dictionary.Intersection({"b": 2}, {"b": 5}, silent=True))
    empty = _py(Dictionary.Intersection(None, {"a": 1}, silent=True))
    assert same == {"b": 2}
    assert different == {"b": [2, 5]}
    assert empty == {}

    xor = _py(Dictionary.SymmetricDifference({"a": 1, "b": 2}, {"b": 9, "c": 3}, silent=True))
    assert xor == {"a": 1, "c": 3}
    assert _py(Dictionary.SymDif({"a": 1}, {"b": 2}, silent=True)) == {"a": 1, "b": 2}
    assert _py(Dictionary.XOR({"a": 1}, {"b": 2}, silent=True)) == {"a": 1, "b": 2}


def test_impose_and_imprint_behaviour():
    imposed = _py(Dictionary.Impose({"a": 1, "b": 2}, {"b": 9, "c": 3}, silent=True))
    imprinted = _py(Dictionary.Imprint({"a": 1, "b": 2}, {"b": 9, "c": 3}, silent=True))

    assert imposed == {"a": 1, "b": 9, "c": 3}
    assert imprinted == {"a": 1, "b": 9}
    assert _py(Dictionary.Impose({"a": 1}, None, silent=True)) == {"a": 1}
    assert _py(Dictionary.Imprint({"a": 1}, None, silent=True)) == {"a": 1}
    assert Dictionary.Impose(None, {"a": 1}, silent=True) is None
    assert Dictionary.Imprint(None, {"a": 1}, silent=True) is None


def test_boolean_dictionaries_by_key_exclusive_union_and_unmatched():
    dictionaries_a = [{"id": "x", "a": 1}, {"id": "y", "a": 2}]
    dictionaries_b = [{"id": "x", "b": 10}, {"id": "z", "b": 30}]

    result = Dictionary.BooleanDictionariesByKey(
        dictionaries_a,
        dictionaries_b,
        "id",
        operation="union",
        exclusive=True,
        silent=True,
    )

    assert len(result) == 2
    assert _py(result[0]) == {"id": "x", "a": 1, "b": 10}
    assert _py(result[1]) == {}


def test_boolean_dictionaries_by_key_nonexclusive_intersect_alias():
    result = Dictionary.BooleanDictionariesByKey(
        {"id": "x", "a": 1},
        [{"id": "x", "b": 2}, {"id": "x", "c": 3}],
        "id",
        operation="intersect",
        exclusive=False,
        silent=True,
    )

    assert len(result) == 1
    assert _py(result[0]) == {"id": "x"}


def test_boolean_dictionaries_by_key_operation_aliases_and_invalid_inputs():
    xor = Dictionary.BooleanDictionariesByKey(
        [{"id": "x", "a": 1}],
        [{"id": "x", "b": 2}],
        "id",
        operation="symdif",
        exclusive=True,
        silent=True,
    )
    assert _py(xor[0]) == {"a": 1, "b": 2}

    assert Dictionary.BooleanDictionariesByKey({"id": "x"}, {"id": "x"}, 5, silent=True) is None
    assert Dictionary.BooleanDictionariesByKey({"id": "x"}, {"id": "x"}, "id", operation=None, silent=True) is None
    assert Dictionary.BooleanDictionariesByKey([], {"id": "x"}, "id", silent=True) is None
    assert Dictionary.BooleanDictionariesByKey({"id": "x"}, [], "id", silent=True) is None
    assert Dictionary.BooleanDictionariesByKey({"id": "x"}, {"id": "x"}, "id", operation="bad", silent=True) is None


def test_filter_by_key_and_any_value_searches():
    dictionaries = [
        {"name": "Office", "zone": "Work"},
        {"name": "Lobby", "zone": "Public"},
        {"name": "Plant", "zone": "Service"},
    ]
    elements = ["e0", "e1", "e2"]

    by_name = Dictionary.Filter(elements, dictionaries, searchType="contains", key="name", value="off")
    assert by_name["filteredDictionaries"] == [dictionaries[0]]
    assert by_name["otherDictionaries"] == dictionaries[1:]
    assert by_name["filteredElements"] == ["e0"]
    assert by_name["filteredIndices"] == [0]

    any_match = Dictionary.Filter(elements, dictionaries, searchType="any", value="public")
    assert any_match["filteredDictionaries"] == [dictionaries[1]]
    assert any_match["filteredElements"] == ["e1"]


def test_filter_empty_or_none_search_value_matches_all_valid_dictionaries():
    dictionaries = [{"a": 1}, {"b": 2}, None]
    result = Dictionary.Filter(["a", "b", "c"], dictionaries, value=None)

    assert result["filteredDictionaries"] == dictionaries[:2]
    assert result["filteredElements"] == ["a", "b"]
    assert result["filteredIndices"] == [0, 1]


def test_filter_search_type_variants():
    dictionaries = [{"name": "North Office"}, {"name": "South Lobby"}]

    assert Dictionary.Filter([], dictionaries, "starts with", "name", "north")["filteredIndices"] == [0]
    assert Dictionary.Filter([], dictionaries, "ends with", "name", "lobby")["filteredIndices"] == [1]
    assert Dictionary.Filter([], dictionaries, "does not contain", "name", "office")["filteredIndices"] == [1]
    assert Dictionary.Filter([], dictionaries, "matches", "name", r"north\s+office")["filteredIndices"] == [0]
    assert Dictionary.Filter([], dictionaries, "equal to", "name", "south lobby")["filteredIndices"] == [1]
    assert Dictionary.Filter([], dictionaries, "not equal to", "name", "south lobby")["filteredIndices"] == [0]


def test_one_hot_encode_string_key_does_not_split_into_characters():
    d = {"zone": "A", "keep": 42}

    result = Dictionary.OneHotEncode(d, "zone", {"zone": ["A", "B"]}, silent=True)

    assert result is d
    assert d == {"keep": 42, "zone_0": 1, "zone_1": 0}
    assert "z_0" not in d
    assert "o_0" not in d
    assert "n_0" not in d
    assert "e_0" not in d


def test_one_hot_encode_multiple_keys_and_invalid_inputs():
    d = {"zone": "A", "type": "room"}

    result = Dictionary.OneHotEncode(
        d,
        ["zone", "type"],
        {"zone": ["A", "B"], "type": ["corridor", "room"]},
        silent=True,
    )

    assert result is d
    assert d == {"zone_0": 1, "zone_1": 0, "type_0": 0, "type_1": 1}
    assert Dictionary.OneHotEncode(None, "zone", {"zone": ["A"]}, silent=True) is None
    assert Dictionary.OneHotEncode({}, None, {}, silent=True) is None
    assert Dictionary.OneHotEncode({}, 5, {}, silent=True) is None
    assert Dictionary.OneHotEncode({}, "zone", [], silent=True) is None
    assert Dictionary.OneHotEncode({}, [], {}, silent=True) == {}


def test_list_attribute_values_accepts_python_lists_and_tuples():
    assert Dictionary.ListAttributeValues([1, "x", None, ("a", 2)]) == [1, "x", None, ["a", 2]]


class _FakeBlenderObject:
    name = "Cube"
    color = (0.1, 0.2, 0.3, 1.0)
    location = (1, 2, 3)
    scale = (1, 1, 1)
    rotation_euler = (0, 0, 0)
    dimensions = (4, 5, 6)

    def __init__(self):
        self._items = {
            "zero": 0,
            "false_value": False,
            "empty_string": "",
            "text": "hello",
            "ignored_list": [1, 2, 3],
        }

    def __getitem__(self, key):
        return self._items[key]

    def items(self):
        return self._items.items()


def test_by_object_properties_preserves_selected_falsy_scalar_values():
    obj = _FakeBlenderObject()
    d = Dictionary.ByObjectProperties(obj, ["zero", "false_value", "empty_string", "Name"], False)
    py = _py(d)

    assert py["zero"] == 0
    assert py["false_value"] in [False, 0]
    assert py["empty_string"] == ""
    assert py["Name"] == "Cube"


def test_by_object_properties_import_all_includes_standard_and_scalar_custom_properties():
    obj = _FakeBlenderObject()
    d = Dictionary.ByObjectProperties(obj, None, True)
    py = _py(d)

    assert py["Name"] == "Cube"
    assert py["Color"] == [0.1, 0.2, 0.3, 1.0]
    assert py["Location"] == [1, 2, 3]
    assert py["Scale"] == [1, 1, 1]
    assert py["Rotation"] == [0, 0, 0]
    assert py["Dimensions"] == [4, 5, 6]
    assert py["zero"] == 0
    assert py["false_value"] in [False, 0]
    assert py["empty_string"] == ""
    assert "ignored_list" not in py


def test_by_object_properties_invalid_key_raises_exception():
    with pytest.raises(Exception):
        Dictionary.ByObjectProperties(_FakeBlenderObject(), ["does_not_exist"], False)






def test_adjacency_dictionary_rejects_invalid_inputs():
    assert Dictionary.AdjacencyDictionary(None, silent=True) is None
    assert Dictionary.AdjacencyDictionary({"not": "a topology"}, silent=True) is None
    assert Dictionary.AdjacencyDictionary(None, labelKey=5, silent=True) is None
    assert Dictionary.AdjacencyDictionary(None, subTopologyType=5, silent=True) is None
