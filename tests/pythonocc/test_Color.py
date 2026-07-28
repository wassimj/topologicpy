# Auto-generated PythonOCC backend parity test
# Copied from test_Color.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Color."""

from __future__ import annotations

import math

import pytest

color_module = pytest.importorskip("topologicpy.Color")
Color = color_module.Color


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _assert_rgb_channels(rgb, include_alpha=False):
    assert isinstance(rgb, list)
    expected_length = 4 if include_alpha else 3
    assert len(rgb) == expected_length

    for channel in rgb[:3]:
        assert isinstance(channel, int)
        assert 0 <= channel <= 255

    if include_alpha:
        assert isinstance(rgb[3], float)
        assert 0.0 <= rgb[3] <= 1.0


def test_add_hex_adds_channels_clips_and_accepts_nested_inputs():
    assert Color.AddHex("#010203", "#040506", silent=True) == "#050709"
    assert Color.AddHex(["#010203", ["#040506"]], silent=True) == "#050709"
    assert Color.AddHex("#F00010", "#300020", silent=True) == "#FF0030"
    assert Color.AddHex("#000000", silent=True) == "#000000"


def test_add_hex_ignores_invalid_colors_and_returns_none_for_no_valid_colors():
    assert Color.AddHex("not-a-color", silent=True) is None
    assert Color.AddHex([], None, 42, silent=True) is None
    assert Color.AddHex("#010203", "not-a-color", silent=True) == "#010203"


def test_any_to_hex_auto_converts_unambiguous_inputs():
    assert Color.AnyToHex("#0f1011", silent=True) == "#0F1011"
    assert Color.AnyToHex("0f1011", silent=True) == "#0F1011"
    assert Color.AnyToHex("#ABC", silent=True) == "#AABBCC"
    assert Color.AnyToHex("#ABCD", silent=True) == "#AABBCC"
    assert Color.AnyToHex("#0F101180", silent=True) == "#0F1011"

    assert Color.AnyToHex([255, 128, 0], silent=True) == "#FF8000"
    assert Color.AnyToHex((15, 16, 17), silent=True) == "#0F1011"
    assert Color.AnyToHex([1.0, 0.5, 0.0], silent=True) == "#FF8000"
    assert Color.AnyToHex([255, 0, 0, 0.5], silent=True) == "#FF0000"

    assert Color.AnyToHex(["#000000", "#FFFFFF"], silent=True) == "#808080"


def test_any_to_hex_explicit_color_models_resolve_four_channel_ambiguity():
    ambiguous = [0, 1, 1, 0]

    assert (
        Color.AnyToHex(
            ambiguous,
            colorModel="cmyk",
            silent=True,
        )
        == "#FF0000"
    )
    assert (
        Color.AnyToHex(
            ambiguous,
            colorModel="rgba",
            silent=True,
        )
        == "#00FFFF"
    )

    assert (
        Color.AnyToHex(
            [1, 0, 0],
            colorModel="rgb",
            silent=True,
        )
        == "#FF0000"
    )
    assert (
        Color.AnyToHex(
            [1, 0, 0, 0.25],
            colorModel="RGBA",
            silent=True,
        )
        == "#FF0000"
    )
    assert (
        Color.AnyToHex(
            [0, 1, 1, 0],
            colorModel="CMYK",
            silent=True,
        )
        == "#FF0000"
    )


def test_any_to_hex_auto_rejects_ambiguous_four_channel_inputs():
    assert Color.AnyToHex([0, 1, 1, 0], silent=True) is None
    assert Color.AnyToHex((1.0, 0.0, 0.0, 0.5), silent=True) is None


def test_any_to_hex_converts_css_function_strings():
    assert Color.AnyToHex("rgb(255, 0, 0)", silent=True) == "#FF0000"
    assert Color.AnyToHex("rgb(100% 0% 0%)", silent=True) == "#FF0000"
    assert Color.AnyToHex("rgba(255, 0, 0, 0.5)", silent=True) == "#FF0000"
    assert Color.AnyToHex("rgb(255 0 0 / 50%)", silent=True) == "#FF0000"

    assert Color.AnyToHex("hsl(0, 100%, 50%)", silent=True) == "#FF0000"
    assert Color.AnyToHex("hsla(120, 100%, 50%, 0.5)", silent=True) == "#00FF00"
    assert Color.AnyToHex("hsl(240deg 100% 50%)", silent=True) == "#0000FF"

    assert Color.AnyToHex("transparent", silent=True) == "#000000"


def test_any_to_hex_converts_css_named_color_when_webcolors_available():
    pytest.importorskip("webcolors")

    assert Color.AnyToHex("red", silent=True) == "#FF0000"
    assert Color.AnyToHex("Blue", silent=True) == "#0000FF"


def test_any_to_hex_rejects_invalid_color_models_and_incompatible_inputs():
    assert Color.AnyToHex([255, 0, 0], colorModel=None, silent=True) is None
    assert Color.AnyToHex([255, 0, 0], colorModel="xyz", silent=True) is None

    assert Color.AnyToHex("#FF0000", colorModel="rgb", silent=True) is None
    assert (
        Color.AnyToHex(
            ["#000000", "#FFFFFF"],
            colorModel="rgb",
            silent=True,
        )
        is None
    )

    assert Color.AnyToHex([255, 0, 0, 1], colorModel="rgb", silent=True) is None
    assert Color.AnyToHex([255, 0, 0], colorModel="rgba", silent=True) is None
    assert Color.AnyToHex([0, 1, 1], colorModel="cmyk", silent=True) is None

    assert Color.AnyToHex([256, 0, 0], colorModel="rgb", silent=True) is None
    assert Color.AnyToHex([255, 0, 0, 2], colorModel="rgba", silent=True) is None
    assert Color.AnyToHex([0, 1, 1, 2], colorModel="cmyk", silent=True) is None


def test_any_to_hex_returns_none_for_unrecognized_inputs():
    assert Color.AnyToHex(None, silent=True) is None
    assert Color.AnyToHex([1, 2], silent=True) is None
    assert Color.AnyToHex([0, 0, 0, 2], silent=True) is None
    assert Color.AnyToHex([True, 0, 0], silent=True) is None
    assert Color.AnyToHex("not-a-css-color", silent=True) is None
    assert Color.AnyToHex("rgba(255, 0, 0, 2)", silent=True) is None
    assert Color.AnyToHex("hsl(0, 120%, 50%)", silent=True) is None


def test_average_computes_arithmetic_mean_of_valid_hex_colors():
    assert Color.Average("#000000", "#FFFFFF", silent=True) == "#808080"
    assert Color.Average(["#000000", "#FFFFFF"], silent=True) == "#808080"
    assert (
        Color.Average(
            "#000000",
            "#FFFFFF",
            "#FF0000",
            silent=True,
        )
        == "#AA5555"
    )
    assert Color.Average("red", "blue", silent=True) == "#800080"


def test_average_returns_none_for_no_valid_colors_and_ignores_invalid_entries():
    assert Color.Average([], silent=True) is None
    assert Color.Average("not-a-color", silent=True) is None
    assert (
        Color.Average(
            "#000000",
            "not-a-color",
            "#FFFFFF",
            silent=True,
        )
        == "#808080"
    )


def test_by_css_named_color_returns_rgb_and_optional_alpha():
    pytest.importorskip("webcolors")

    assert Color.ByCSSNamedColor("red", silent=True) == [255, 0, 0]
    assert Color.ByCSSNamedColor(
        "red",
        alpha=0.25,
        silent=True,
    ) == [255, 0, 0, 0.25]
    assert Color.ByCSSNamedColor("not-a-color", silent=True) is None
    assert Color.ByCSSNamedColor("red", alpha=1.5, silent=True) is None
    assert Color.ByCSSNamedColor(123, silent=True) is None


def test_by_hex_converts_valid_hex_and_rejects_malformed_values():
    assert Color.ByHEX("#0F1011", silent=True) == [15, 16, 17]
    assert Color.ByHEX("0f1011", silent=True) == [15, 16, 17]
    assert Color.ByHEX(
        "#0F1011",
        alpha=0.5,
        silent=True,
    ) == [15, 16, 17, 0.5]

    assert Color.ByHEX("#GGGGGG", silent=True) is None
    assert Color.ByHEX("#12345", silent=True) is None
    assert Color.ByHEX(123, silent=True) is None
    assert Color.ByHEX("#0F1011", alpha=-0.1, silent=True) is None


def test_by_value_in_range_default_scale_clamps_swaps_and_adds_alpha():
    assert Color.ByValueInRange(
        0.0,
        colorScale="default",
        silent=True,
    ) == [0, 0, 255]
    assert Color.ByValueInRange(
        0.25,
        colorScale="default",
        silent=True,
    ) == [0, 255, 255]
    assert Color.ByValueInRange(
        0.5,
        colorScale="default",
        silent=True,
    ) == [0, 255, 0]
    assert Color.ByValueInRange(
        0.75,
        colorScale="default",
        silent=True,
    ) == [255, 255, 0]
    assert Color.ByValueInRange(
        1.0,
        colorScale="default",
        silent=True,
    ) == [255, 0, 0]
    assert Color.ByValueInRange(
        -10.0,
        colorScale="default",
        silent=True,
    ) == [0, 0, 255]
    assert Color.ByValueInRange(
        10.0,
        colorScale="default",
        silent=True,
    ) == [255, 0, 0]
    assert Color.ByValueInRange(
        0.75,
        minValue=1.0,
        maxValue=0.0,
        colorScale="default",
        silent=True,
    ) == [255, 255, 0]
    assert Color.ByValueInRange(
        0.5,
        colorScale="default",
        alpha=0.3,
        silent=True,
    ) == [0, 255, 0, 0.3]


def test_by_value_in_range_plotly_scale_returns_rgb_channels():
    rgb = Color.ByValueInRange(
        0.5,
        minValue=0,
        maxValue=1,
        colorScale="viridis",
        silent=True,
    )
    _assert_rgb_channels(rgb)

    rgba = Color.ByValueInRange(
        0.5,
        minValue=0,
        maxValue=1,
        alpha=0.25,
        colorScale="plasma",
        silent=True,
    )
    _assert_rgb_channels(rgba, include_alpha=True)
    assert math.isclose(rgba[3], 0.25)


def test_by_value_in_range_rejects_invalid_inputs():
    assert Color.ByValueInRange("x", silent=True) is None
    assert Color.ByValueInRange(0.5, minValue="x", silent=True) is None
    assert Color.ByValueInRange(0.5, maxValue="x", silent=True) is None
    assert Color.ByValueInRange(0.5, alpha=2, silent=True) is None
    assert Color.ByValueInRange(
        0.5,
        colorScale="not-a-colorscale",
        silent=True,
    ) is None


def test_cmyk_to_hex_converts_valid_values_and_rejects_invalid_inputs():
    assert Color.CMYKToHex([0, 0, 0, 0], silent=True) == "#FFFFFF"
    assert Color.CMYKToHex([0, 1, 1, 0], silent=True) == "#FF0000"
    assert Color.CMYKToHex([1, 0, 1, 0], silent=True) == "#00FF00"
    assert Color.CMYKToHex([1, 1, 0, 0], silent=True) == "#0000FF"

    assert Color.CMYKToHex([0, 0, 0], silent=True) is None
    assert Color.CMYKToHex([0, 0, 0, 2], silent=True) is None
    assert Color.CMYKToHex("invalid", silent=True) is None


def test_css_named_color_and_color_list_when_webcolors_available():
    pytest.importorskip("webcolors")

    assert Color.CSSNamedColor([255, 0, 0], silent=True) == "red"
    assert Color.CSSNamedColor([254, 1, 1], silent=True) == "red"
    assert Color.CSSNamedColor([256, 0, 0], silent=True) is None

    colors = Color.CSSNamedColors()
    assert isinstance(colors, list)
    assert "red" in colors
    assert "blue" in colors


def test_plotly_color_formats_rgb_and_rgba_strings():
    assert Color.PlotlyColor([1, 2, 3], silent=True) == "rgb(1,2,3)"
    assert Color.PlotlyColor(
        [1, 2, 3],
        alpha=0.5,
        silent=True,
    ) == "rgba(1,2,3,0.5)"
    assert Color.PlotlyColor(
        [1, 2, 3, 0.25],
        silent=True,
    ) == "rgba(1,2,3,0.25)"
    assert Color.PlotlyColor(
        [1, 2, 3],
        useAlpha=True,
        silent=True,
    ) == "rgba(1,2,3,1.0)"


def test_plotly_color_rejects_invalid_inputs():
    assert Color.PlotlyColor("red", silent=True) is None
    assert Color.PlotlyColor([1, 2], silent=True) is None
    assert Color.PlotlyColor([1, 2, 300], silent=True) is None
    assert Color.PlotlyColor([1, 2, 3], alpha=2, silent=True) is None


def test_rgb_to_hex_converts_valid_rgb_and_rejects_invalid_inputs():
    assert Color.RGBToHex([15, 16, 17], silent=True) == "#0F1011"
    assert Color.RGBToHex((255, 128, 0), silent=True) == "#FF8000"
    assert Color.RGBToHex([0.4, 1.6, 2.4], silent=True) == "#000202"

    assert Color.RGBToHex([1, 2], silent=True) is None
    assert Color.RGBToHex([1, 2, 3, 4], silent=True) == "#010203"
    assert Color.RGBToHex([1, 2, 300], silent=True) is None
    assert Color.RGBToHex([True, 0, 0], silent=True) is None
