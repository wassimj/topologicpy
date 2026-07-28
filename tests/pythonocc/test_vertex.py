# Copyright (C) 2026
# PythonOCC backend Vertex parity tests.
#
# These tests mirror tests/test_Vertex.py but run against the PythonOCC backend.
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_vertex.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Vertex
Topology = pytest.importorskip("topologicpy.Topology").Topology
Dictionary = pytest.importorskip("topologicpy.Dictionary").Dictionary

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def _coords(vertex, mantissa=6):
    return Vertex.Coordinates(vertex, mantissa=mantissa)


def _assert_coords(vertex, expected, abs_tol=TOLERANCE, mantissa=6):
    actual = _coords(vertex, mantissa=mantissa)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert value == pytest.approx(target, abs=abs_tol)


# ===========================================================================
# S13.1 -- Primitive constructors
# ===========================================================================

class TestVertexConstructors:
    def test_by_coordinates_basic(self):
        v = _v(1.0, 2.0, 3.0)
        assert Topology.IsInstance(v, "Vertex")
        _assert_coords(v, [1.0, 2.0, 3.0])

    def test_by_coordinates_origin(self):
        v = _v(0, 0, 0)
        assert Topology.IsInstance(v, "Vertex")
        _assert_coords(v, [0.0, 0.0, 0.0])

    def test_by_coordinates_negative(self):
        v = _v(-1.5, -2.5, -3.5)
        assert Topology.IsInstance(v, "Vertex")
        _assert_coords(v, [-1.5, -2.5, -3.5])

    def test_by_coordinates_large_values(self):
        v = _v(1e6, 1e6, 1e6)
        assert Topology.IsInstance(v, "Vertex")
        _assert_coords(v, [1e6, 1e6, 1e6])

    def test_by_coordinates_small_values(self):
        v = _v(1e-6, 1e-6, 1e-6)
        assert Topology.IsInstance(v, "Vertex")
        _assert_coords(v, [1e-6, 1e-6, 1e-6])


# ===========================================================================
# Accessors
# ===========================================================================

class TestVertexAccessors:
    def test_x_y_z(self):
        v = _v(1.0, 2.0, 3.0)
        assert Vertex.X(v) == pytest.approx(1.0, abs=TOLERANCE)
        assert Vertex.Y(v) == pytest.approx(2.0, abs=TOLERANCE)
        assert Vertex.Z(v) == pytest.approx(3.0, abs=TOLERANCE)

    def test_coordinates_list(self):
        v = _v(1.0, 2.0, 3.0)
        coords = Vertex.Coordinates(v)
        assert len(coords) == 3
        assert coords[0] == pytest.approx(1.0, abs=TOLERANCE)
        assert coords[1] == pytest.approx(2.0, abs=TOLERANCE)
        assert coords[2] == pytest.approx(3.0, abs=TOLERANCE)

    def test_coordinates_with_mantissa(self):
        v = _v(1.123456789, 2.123456789, 3.123456789)
        coords = Vertex.Coordinates(v, mantissa=3)
        assert coords[0] == pytest.approx(1.123, abs=0.001)
        assert coords[1] == pytest.approx(2.123, abs=0.001)
        assert coords[2] == pytest.approx(3.123, abs=0.001)


# ===========================================================================
# Type checking
# ===========================================================================

class TestVertexType:
    def test_is_instance_vertex(self):
        v = _v(0, 0, 0)
        assert Topology.IsInstance(v, "Vertex") is True

    def test_is_not_edge(self):
        v = _v(0, 0, 0)
        assert Topology.IsInstance(v, "Edge") is False

    def test_type_returns_vertex(self):
        v = _v(0, 0, 0)
        assert Topology.Type(v) == 1  # Vertex type ID


# ===========================================================================
# Dictionary
# ===========================================================================

class TestVertexDictionary:
    def test_set_get_dictionary(self):
        v = _v(1, 2, 3)
        d = Dictionary.ByKeysValues(["key"], ["value"])
        v2 = Topology.SetDictionary(v, d)
        d2 = Topology.Dictionary(v2)
        assert Dictionary.ValueAtKey(d2, "key") == "value"

    def test_dictionary_preserves_geometry(self):
        v = _v(1, 2, 3)
        d = Dictionary.ByKeysValues(["key"], ["value"])
        v2 = Topology.SetDictionary(v, d)
        _assert_coords(v2, [1.0, 2.0, 3.0])


# ===========================================================================
# Serialization
# ===========================================================================

class TestVertexSerialization:
    def test_brep_roundtrip(self):
        v = _v(1, 2, 3)
        brep = Topology.BREPString(v)
        assert brep is not None
        assert len(brep) > 0
        v2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(v2, "Vertex")
        _assert_coords(v2, [1.0, 2.0, 3.0])

    def test_json_roundtrip(self):
        v = _v(1, 2, 3)
        d = Dictionary.ByKeysValues(["x", "y", "z"], [1, 2, 3])
        v2 = Topology.SetDictionary(v, d)
        json_str = Topology.JSON(v2)
        assert json_str is not None
        v3 = Topology.ByJSON(json_str)
        assert Topology.IsInstance(v3, "Vertex")


# ===========================================================================
# Operations
# ===========================================================================

class TestVertexOperations:
    def test_distance_between_vertices(self):
        v1 = _v(0, 0, 0)
        v2 = _v(1, 0, 0)
        dist = Topology.Distance(v1, v2)
        assert dist == pytest.approx(1.0, abs=TOLERANCE)

    def test_distance_3d(self):
        v1 = _v(0, 0, 0)
        v2 = _v(1, 1, 1)
        dist = Topology.Distance(v1, v2)
        assert dist == pytest.approx(math.sqrt(3), abs=TOLERANCE)

    def test_translate(self):
        v = _v(0, 0, 0)
        v2 = Topology.Translate(v, 1, 2, 3)
        assert Topology.IsInstance(v2, "Vertex")
        _assert_coords(v2, [1.0, 2.0, 3.0])

    def test_copy(self):
        v = _v(1, 2, 3)
        v2 = Topology.Copy(v)
        assert Topology.IsInstance(v2, "Vertex")
        _assert_coords(v2, [1.0, 2.0, 3.0])
        # Ensure it's a copy, not the same object
        v3 = Topology.Translate(v, 10, 0, 0)
        _assert_coords(v2, [1.0, 2.0, 3.0])  # v2 unchanged
