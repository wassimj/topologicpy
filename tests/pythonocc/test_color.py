# Copyright (C) 2026
# PythonOCC backend Color parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_color.py -v

import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
Topology = pytest.importorskip("topologicpy.Topology").Topology
Color = pytest.importorskip("topologicpy.Color").Color

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


# ===========================================================================
# Color constructors
# ===========================================================================

class TestColorConstructors:
    def test_by_rgb(self):
        c = Color.ByRGB(255, 0, 0)
        assert c is not None

    def test_by_name(self):
        c = Color.ByName("red")
        assert c is not None

    def test_by_hex(self):
        c = Color.ByHex("#FF0000")
        assert c is not None

    def test_white(self):
        c = Color.White()
        assert c is not None

    def test_black(self):
        c = Color.Black()
        assert c is not None

    def test_red(self):
        c = Color.Red()
        assert c is not None

    def test_green(self):
        c = Color.Green()
        assert c is not None

    def test_blue(self):
        c = Color.Blue()
        assert c is not None


# ===========================================================================
# Color accessors
# ===========================================================================

class TestColorAccessors:
    def test_rgb_values(self):
        c = Color.ByRGB(100, 150, 200)
        r, g, b = Color.RGB(c)
        assert r == 100
        assert g == 150
        assert b == 200

    def test_hex_value(self):
        c = Color.ByRGB(255, 0, 0)
        hex_val = Color.Hex(c)
        assert hex_val == "#FF0000"

    def test_name(self):
        c = Color.Red()
        name = Color.Name(c)
        assert name is not None


# ===========================================================================
# Color operations
# ===========================================================================

class TestColorOperations:
    def test_invert(self):
        c = Color.ByRGB(255, 0, 0)
        c2 = Color.Invert(c)
        r, g, b = Color.RGB(c2)
        assert r == 0
        assert g == 255
        assert b == 255

    def test_blend(self):
        c1 = Color.Red()
        c2 = Color.Blue()
        c3 = Color.Blend(c1, c2, 0.5)
        assert c3 is not None

    def test_lighter(self):
        c = Color.ByRGB(100, 100, 100)
        c2 = Color.Lighter(c)
        r1, g1, b1 = Color.RGB(c)
        r2, g2, b2 = Color.RGB(c2)
        assert r2 >= r1
        assert g2 >= g1
        assert b2 >= b1

    def test_darker(self):
        c = Color.ByRGB(200, 200, 200)
        c2 = Color.Darker(c)
        r1, g1, b1 = Color.RGB(c)
        r2, g2, b2 = Color.RGB(c2)
        assert r2 <= r1
        assert g2 <= g1
        assert b2 <= b1


# ===========================================================================
# Topology integration
# ===========================================================================

class TestColorTopology:
    def test_set_color_on_face(self):
        f = Face.Rectangle(1.0, 1.0)
        c = Color.Red()
        f2 = Topology.SetColor(f, c)
        assert f2 is not None

    def test_set_color_on_cell(self):
        c = Cell.Prism()
        color = Color.Blue()
        c2 = Topology.SetColor(c, color)
        assert c2 is not None
