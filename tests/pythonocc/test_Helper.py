# Auto-generated PythonOCC backend parity test
# Copied from test_Helper.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Helper."""

from __future__ import annotations

import importlib
import inspect
import random
import sys
import types

import pytest

from topologicpy.Helper import Helper
import topologicpy.Helper as helper_module


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class _FakeDictionary:
    @staticmethod
    def Keys(dictionary):
        if isinstance(dictionary, dict):
            return list(dictionary.keys())
        return []

    @staticmethod
    def ValueAtKey(dictionary, key):
        if isinstance(dictionary, dict):
            return dictionary.get(key)
        return None


@pytest.fixture
def fake_dictionary_module(monkeypatch):
    module = types.SimpleNamespace(Dictionary=_FakeDictionary)
    monkeypatch.setitem(sys.modules, "topologicpy.Dictionary", module)
    return module


def test_import_has_no_runtime_pip_install_side_effects():
    source = inspect.getsource(helper_module)
    assert "os.system" not in source
    assert "pip install" not in source


def test_private_numeric_helpers_are_defensive():
    assert Helper._Mantissa("bad") == 6
    assert Helper._Mantissa(-3) == 0
    assert Helper._Tolerance("bad") == pytest.approx(0.0001)
    assert Helper._Tolerance(-0.5) == pytest.approx(0.5)
    assert Helper._IsNumber(3.0) is True
    assert Helper._IsNumber(True) is False
    assert Helper._NumericList([1, 2.5], silent=True) == [1.0, 2.5]
    assert Helper._NumericList([1, "x"], silent=True) is None
    assert Helper._Hashable([1, 2]) == repr([1, 2])


def test_bin_and_average_groups_sorted_numeric_values():
    values = [2.0, 1.0, 1.04, 5.0, 5.02]
    result = Helper.BinAndAverage(values, mantissa=3, tolerance=0.05)
    assert result == [1.02, 2.0, 5.01]
    assert values == [2.0, 1.0, 1.04, 5.0, 5.02]
    assert Helper.BinAndAverage([7.12345], mantissa=2) == [7.12]
    assert Helper.BinAndAverage([], silent=True) is None
    assert Helper.BinAndAverage("bad", silent=True) is None
    assert Helper.BinAndAverage([1, object()], silent=True) is None


def test_check_version_with_fake_requests(monkeypatch):
    class FakeResponse:
        def __init__(self, latest):
            self._latest = latest

        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": self._latest}}

    calls = []

    def fake_get(url, timeout=10):
        calls.append((url, timeout))
        return FakeResponse("2.0.0")

    fake_requests = types.SimpleNamespace(get=fake_get)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    assert "OLDER" in Helper.CheckVersion("examplelib", "1.9.0", silent=True)
    assert "EQUAL" in Helper.CheckVersion("examplelib", "2.0.0", silent=True)
    assert "NEWER" in Helper.CheckVersion("examplelib", "2.1.0", silent=True)
    assert calls[0][0] == "https://pypi.org/pypi/examplelib/json"
    assert calls[0][1] == 10
    assert Helper.CheckVersion("", "1.0.0", silent=True) is None
    assert Helper.CheckVersion("examplelib", None, silent=True) is None


def test_check_version_handles_request_failure(monkeypatch):
    def fake_get(url, timeout=10):
        raise RuntimeError("network disabled")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    assert Helper.CheckVersion("examplelib", "1.0.0", silent=True) is None


def test_closest_match_numeric_and_string_are_deterministic():
    assert Helper.ClosestMatch(9.2, [1, "bad", 8.8, 10.0]) == 2
    assert Helper.ClosestMatch(True, [0, 5, 1]) == 2
    assert Helper.ClosestMatch("kithen", [12, "living", "kitchen", "bath"]) == 2
    assert Helper.ClosestMatch("room", [1, 2, 3]) is None
    assert Helper.ClosestMatch(object(), [1, "a"]) is None
    assert Helper.ClosestMatch(5, []) is None


def test_cluster_by_keys_groups_stably_and_does_not_mutate(fake_dictionary_module):
    elements = ["a", "b", "c", "d"]
    dictionaries = [
        {"zone": "public", "level": 1},
        {"zone": "private", "level": 1},
        {"zone": "public", "level": 1},
        {},
    ]
    original = [dict(d) for d in dictionaries]

    result = Helper.ClusterByKeys(elements, dictionaries, ["zone"], "level", silent=True)

    assert result["elements"] == [["a", "c"], ["b"], ["d"]]
    assert result["dictionaries"] == [
        [{"zone": "public", "level": 1}, {"zone": "public", "level": 1}],
        [{"zone": "private", "level": 1}],
        [{}],
    ]
    assert dictionaries == original
    assert Helper.ClusterByKeys(elements, dictionaries, silent=True) is None
    assert Helper.ClusterByKeys(elements, dictionaries[:-1], "zone", silent=True) is None
    assert Helper.ClusterByKeys("bad", dictionaries, "zone", silent=True) is None


def test_flatten_preserves_order_and_wraps_non_lists():
    assert Helper.Flatten([1, [2, [3, 4]], [], 5]) == [1, 2, 3, 4, 5]
    assert Helper.Flatten("x") == ["x"]


def test_grow_is_deterministic_under_patched_shuffle_and_does_not_mutate(monkeypatch):
    adjacency = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
    original = {key: list(value) for key, value in adjacency.items()}
    monkeypatch.setattr(random, "shuffle", lambda values: None)

    assert Helper.Grow(0, 3, adjacency, visited_global=set()) == [0, 1, 2]
    assert adjacency == original
    assert Helper.Grow(0, 5, adjacency, visited_global=set()) is None
    assert Helper.Grow(0, 2, adjacency, visited_global={0}) is None
    assert Helper.Grow(0, "bad", adjacency, visited_global=set()) is None


def test_iterate_repeat_trim_transpose_are_non_mutating_and_handle_empty_sublists():
    data = [[1, 2, 3], ["m", "n", "o", "p"], ["a", "b", "c", "d", "e"]]
    original = [list(row) for row in data]
    assert Helper.Iterate(data) == [[1, 2, 3, 1, 2], ["m", "n", "o", "p", "m"], ["a", "b", "c", "d", "e"]]
    assert Helper.Repeat(data) == [[1, 2, 3, 3, 3], ["m", "n", "o", "p", "p"], ["a", "b", "c", "d", "e"]]
    assert Helper.Trim(data) == [[1, 2, 3], ["m", "n", "o"], ["a", "b", "c"]]
    assert data == original

    assert Helper.Iterate([[], [1, 2]]) == [[None, None], [1, 2]]
    assert Helper.Repeat([[], [1, 2]]) == [[None, None], [1, 2]]
    assert Helper.Transpose([[1, 2, 3], [4, 5], [6, 7, 8]]) == [[1, 4, 6], [2, 5, 7]]
    assert Helper.Iterate("bad") is None
    assert Helper.Repeat(["bad"]) is None
    assert Helper.Trim([1, 2]) is None
    assert Helper.Transpose([[1], "bad"]) is None


def test_make_unique_converts_to_strings_and_suffixes_duplicates():
    assert Helper.MakeUnique(["A", "B", "A", "A", 7, 7]) == ["A", "B", "A_1", "A_2", "7", "7_1"]
    assert Helper.MakeUnique("bad") is None


def test_merge_by_threshold_uses_true_bin_averages_and_does_not_mutate():
    values = [1.0, 1.04, 1.08, 5.0, 5.02]
    assert Helper.MergeByThreshold(values, threshold=0.05) == pytest.approx([1.04, 5.01])
    assert values == [1.0, 1.04, 1.08, 5.0, 5.02]
    assert Helper.MergeByThreshold([], threshold=0.1) == []
    assert Helper.MergeByThreshold("bad", threshold=0.1) is None
    assert Helper.MergeByThreshold([1, "bad"], threshold=0.1) is None


def test_maximum_minimum_indices_and_normalize():
    assert Helper.MaximumIndices([7, 3, 4, 7, 5, 7]) == [0, 3, 5]
    assert Helper.MinimumIndices([1, 3, 4, 1, 5, 1]) == [0, 3, 5]
    assert Helper.MaximumIndices([]) == []
    assert Helper.MinimumIndices([]) == []
    assert Helper.MaximumIndices("bad", silent=True) is None
    assert Helper.MinimumIndices("bad", silent=True) is None
    assert Helper.Normalize([10, 15, "x", 20], mantissa=2) == [0.0, 0.5, 1.0]
    assert Helper.Normalize([5, 5, 5]) == [0, 0, 0]
    assert Helper.Normalize("bad") is None
    assert Helper.Normalize(["bad"]) is None


def test_position_and_remove_helpers():
    assert Helper.Position(5, [1, 3, 5, 7]) == 2
    assert Helper.Position(4, [1, 3, 5, 7]) == 2
    assert Helper.Position(0, [1, 3, 5]) == 0
    assert Helper.Position(9, [1, 3, 5]) == 3
    assert Helper.Position("bad", [1, 2]) is None
    assert Helper.Position(1, "bad") is None
    assert Helper.RemoveEven(["a", "b", "c", "d"]) == ["b", "d"]
    assert Helper.RemoveOdd(["a", "b", "c", "d"]) == ["a", "c"]
    assert Helper.RemoveEven("bad") is None
    assert Helper.RemoveOdd("bad") is None


def test_sort_supports_no_keys_multi_keys_reverse_flags_and_validation():
    assert Helper.Sort([3, 1, 2]) == [1, 2, 3]
    objects = ["a", "b", "c", "d"]
    primary = [2, 1, 2, 1]
    secondary = [1, 2, 0, 1]
    assert Helper.Sort(objects, primary, secondary, reverseFlags=[False, False]) == ["d", "b", "c", "a"]
    assert Helper.Sort(objects, primary, secondary, reverseFlags=[True, False]) == ["c", "a", "d", "b"]
    assert Helper.Sort(objects, primary, reverseFlags=[False, True], silent=True) is None
    assert Helper.Sort(objects, [1, 2], silent=True) is None
    assert Helper.Sort("bad", primary, silent=True) is None


def test_version_uses_topologicpy_version_and_optional_check(monkeypatch):
    monkeypatch.setattr(helper_module.topologicpy, "__version__", "9.8.7", raising=False)
    assert Helper.Version(check=False, silent=True) == "9.8.7"
    monkeypatch.setattr(Helper, "CheckVersion", staticmethod(lambda library, version, silent=False: f"{library}:{version}"))
    assert Helper.Version(check=True, silent=True) == "topologicpy:9.8.7"


def test_source_guards_for_corrected_regressions():
    source = inspect.getsource(Helper)
    assert "max_value = max(listA)" in source
    assert "min_value = min(listA)" in source
    assert "random.choice" not in source
    assert "os.system" not in source
