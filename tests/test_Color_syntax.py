"""Depthmap Classic palette and Plotly integration regressions.

Reference: SpaceGroupUCL/depthmapX salalib/pafcolor.cpp,
makeDepthmapClassic/htmlByte, blue=0 and red=1.
"""
import json

import plotly.colors
import plotly.graph_objects as go
import pytest

from topologicpy.Color import Color
from topologicpy.Plotly import Plotly


@pytest.mark.parametrize("value, expected", [
    (0, [0, 0, 255]),
    (0.025, [0, 119, 255]),
    (0.05, [0, 255, 255]),
    (0.075, [0, 255, 119]),
    (0.1, [0, 255, 0]),
    (0.325, [119, 255, 0]),
    (0.55, [255, 255, 0]),
    (0.775, [255, 119, 0]),
    (1, [255, 0, 0]),
])
def test_classic_reference_colors(value, expected):
    assert Color.ByValueInRange(value, colorScale="syntax") == expected
    assert Color.ByValueInRange(value, colorScale=" SyNtAx ") == expected


def test_range_clamping_alpha_and_reversal():
    assert Color.ByValueInRange(12, 10, 30, alpha=0.4, colorScale="syntax") == [0, 255, 0, 0.4]
    assert Color.ByValueInRange(12, 30, 10, colorScale="syntax") == [0, 255, 0]
    assert Color.ByValueInRange(-2, colorScale="syntax") == [0, 0, 255]
    assert Color.ByValueInRange(2, colorScale="syntax") == [255, 0, 0]
    assert Color.ByValueInRange(5, 5, 5, colorScale="syntax") == [0, 0, 255]
    assert Color.ByValueInRange(0.9, colorScale="syntax_r") == [0, 255, 0]
    assert Color.ByValueInRange(float("nan"), colorScale="syntax", silent=True) is None


def test_plotly_scale_matches_exact_channels_and_serializes():
    scale = Plotly.ColorScale("syntax")
    assert scale == Color.ColorScale("syntax")
    assert scale[0] == [0.0, "rgb(0, 0, 255)"]
    assert scale[-1] == [1.0, "rgb(255, 0, 0)"]
    assert all(a[0] < b[0] for a, b in zip(scale, scale[1:]))
    # Plotly interpolation must keep the quantized channels between transitions.
    samples = [(i + 0.37) / 2000 for i in range(2000)]
    actual = plotly.colors.sample_colorscale(scale, samples)
    expected = [Color.PlotlyColor(Color.ByValueInRange(t, colorScale="syntax")) for t in samples]
    assert [Color.AnyToHex(c) for c in actual] == [Color.AnyToHex(c) for c in expected]
    figure = go.Figure(go.Heatmap(z=[[0, 0.1, 0.55, 1]], colorscale=scale))
    assert json.loads(figure.to_json())["data"][0]["colorscale"] == scale
    scale[0][1] = "black"
    assert Color.ColorScale("syntax")[0][1] == "rgb(0, 0, 255)"


def test_standard_scales_and_custom_lists_still_resolve():
    scale = Color.ColorScale("viridis")
    standard = plotly.colors.get_colorscale("viridis")
    assert [p for p, _ in scale] == pytest.approx([p for p, _ in standard])
    assert [c for _, c in scale] == [c for _, c in standard]
    custom = [[0, "#000000"], [1, "#FFFFFF"]]
    assert Color.ColorScale(custom) == custom
    assert Color.ColorScale("unknown", silent=True) is None
    reversed_scale = Plotly.ColorScale("syntax_r")
    samples = [(i + 0.37) / 1000 for i in range(1000)]
    actual = plotly.colors.sample_colorscale(reversed_scale, samples)
    expected = [Color.PlotlyColor(Color.ByValueInRange(t, colorScale="syntax_r")) for t in samples]
    assert [Color.AnyToHex(c) for c in actual] == [Color.AnyToHex(c) for c in expected]
    assert reversed_scale[0][1] == "rgb(255, 0, 0)"
    assert reversed_scale[-1][1] == "rgb(0, 0, 255)"


def test_tgraph_analysis_accepts_syntax_without_networkx():
    from topologicpy.TGraph import TGraph
    graph = TGraph()
    for i in range(4):
        graph.AddVertex({"x": i, "y": 0, "z": 0})
    for i in range(3):
        graph.AddEdge(i, i + 1)
    values = TGraph.ClosenessCentrality(graph, colorScale="syntax")
    assert values is not None
    for vertex, value in zip(graph._vertices, values):
        expected = Color.AnyToHex(Color.ByValueInRange(value, min(values), max(values), colorScale="syntax"))
        assert vertex["dictionary"]["cc_color"] == expected
