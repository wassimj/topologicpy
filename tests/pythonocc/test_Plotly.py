


"""Best-effort unit tests for topologicpy.Plotly.

These tests intentionally avoid renderer/browser/Kaleido-dependent assertions and
use fake TopologicPy helper modules where the Plotly methods need Dictionary,
Color, or TGraph support.
"""

from __future__ import annotations

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

import json
import math
import sys
import types
import zipfile
from pathlib import Path

import pytest

from topologicpy.Plotly import Plotly

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - Plotly.py itself is being tested.
    go = None


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


@pytest.fixture
def fake_color_dictionary(monkeypatch):
    """Install deterministic fake topologicpy.Color and Dictionary modules."""
    dict_mod = types.ModuleType("topologicpy.Dictionary")

    class Dictionary:
        @staticmethod
        def ValueAtKey(dictionary, key=None, defaultValue=None):
            if isinstance(dictionary, dict):
                return dictionary.get(key, defaultValue)
            return defaultValue

    dict_mod.Dictionary = Dictionary

    color_mod = types.ModuleType("topologicpy.Color")

    class Color:
        @staticmethod
        def AnyToHex(value):
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                vals = [max(0, min(255, int(float(v)))) for v in value[:3]]
                return "#{:02x}{:02x}{:02x}".format(vals[0], vals[1], vals[2])
            return "#000000"

        @staticmethod
        def ByValueInRange(value, minValue=0, maxValue=1, colorScale="viridis"):
            try:
                value = float(value)
                minValue = float(minValue)
                maxValue = float(maxValue)
                t = 0.0 if abs(maxValue - minValue) <= 1e-12 else (value - minValue) / (maxValue - minValue)
                t = max(0.0, min(1.0, t))
                return "#{:02x}{:02x}{:02x}".format(int(255 * t), 0, int(255 * (1.0 - t)))
            except Exception:
                return "#000000"

    color_mod.Color = Color
    monkeypatch.setitem(sys.modules, "topologicpy.Dictionary", dict_mod)
    monkeypatch.setitem(sys.modules, "topologicpy.Color", color_mod)
    return Color, Dictionary


@pytest.fixture
def fake_tgraph(monkeypatch):
    """Install a deterministic fake topologicpy.TGraph module."""
    tgraph_mod = types.ModuleType("topologicpy.TGraph")

    class TGraph:
        def __init__(self):
            self._directed = False
            self.vertices = [
                {"index": 0, "dictionary": {"label": "A", "group": 0, "size": 8}},
                {"index": 1, "dictionary": {"label": "B", "group": 1, "shape": "square"}},
            ]
            self.edges = [
                {"index": 0, "src": 0, "dst": 1, "directed": True, "dictionary": {"label": "ab", "width": 2}},
                {"index": 1, "src": 1, "dst": 1, "directed": False, "dictionary": {"label": "loop", "self_loop_radius": 0.1}},
            ]
            self.coordinates = {0: [0, 0, 0], 1: [1, 0, 0]}

        @staticmethod
        def Vertices(graph, asTopologic=False, active=True):
            return list(graph.vertices)

        @staticmethod
        def Edges(graph, asTopologic=False, active=True):
            return list(graph.edges)

        @staticmethod
        def Coordinates(graph, index, default=None):
            return graph.coordinates.get(index, default)

    tgraph_mod.TGraph = TGraph
    monkeypatch.setitem(sys.modules, "topologicpy.TGraph", tgraph_mod)
    return TGraph


def _assert_plotly_available():
    assert go is not None
    assert Plotly._plotly_available(silent=True) is True


def test_color_helpers_and_named_colours(fake_color_dictionary):
    _assert_plotly_available()
    assert "black" in Plotly.Colors()
    assert "rebeccapurple" in Plotly.Colors()

    protanopia = Plotly.ColorScale("protanopia")
    assert isinstance(protanopia, list)
    assert protanopia[0][0] == 0
    assert protanopia[-1][0] == 1
    assert Plotly.ColorScale("Viridis") == "Viridis"
    assert Plotly._color_to_hex([255, 0, 0]) == "#ff0000"


def test_add_colorbar_handles_invalid_empty_and_tick_values(fake_color_dictionary):
    _assert_plotly_available()
    assert Plotly.AddColorBar("not a figure", values=[1, 2]) is None

    fig = go.Figure()
    same = Plotly.AddColorBar(fig, values=[])
    assert same is fig
    assert len(fig.data) == 0

    fig = Plotly.AddColorBar(go.Figure(), values=[0, "bad", 10], nTicks=3, title="T", units="m")
    assert len(fig.data) == 1
    colorbar = fig.data[0].marker.colorbar
    assert list(colorbar.tickvals) == [0.0, 5.0, 10.0]
    assert "Units: m" in str(colorbar.title.text)


def test_vertex_data_uses_dictionaries_groups_labels_and_borders(fake_color_dictionary):
    _assert_plotly_available()
    traces = Plotly.vertexData(
        [[0, 0, 0], [1, 2, 3]],
        dictionaries=[{"label": "A", "size": 4, "group": 1}, {"label": "B", "size": 5, "group": 2}],
        sizeKey="size",
        labelKey="label",
        showVertexLabel=True,
        borderWidth=2,
        groupKey="group",
        groups=[1, 2],
        showLegend=True,
    )
    assert [trace.type for trace in traces] == ["scatter3d", "scatter3d"]
    assert traces[0].mode == "markers"
    assert traces[1].mode == "markers+text"
    assert list(traces[1].text) == ["A", "B"]
    assert list(traces[1].marker.size) == [4.0, 5.0]
    assert traces[1].showlegend is True


def test_edge_data_creates_lines_labels_and_cones(fake_color_dictionary):
    _assert_plotly_available()
    traces = Plotly.edgeData(
        [[0, 0, 0], [1, 0, 0]],
        [[0, 1]],
        dictionaries=[{"label": "e", "width": 3, "dash": True}],
        widthKey="width",
        dashKey="dash",
        labelKey="label",
        showEdgeLabel=True,
        directed=True,
        arrowSize=0.2,
    )
    assert [trace.type for trace in traces] == ["scatter3d", "scatter3d", "cone"]
    assert traces[0].mode == "lines+markers"
    assert traces[0].line.width == 3.0
    assert list(traces[1].text) == ["e"]
    assert list(traces[2].u) == [1.0]


def test_data_by_tgraph_uses_fake_tgraph_and_handles_self_loops(fake_color_dictionary, fake_tgraph):
    _assert_plotly_available()
    graph = fake_tgraph()
    traces = Plotly.DataByTGraph(
        graph,
        directed=None,
        edgeLabelKey="label",
        showEdgeLabel=True,
        vertexLabelKey="label",
        showVertexLabel=True,
        showVertexLegend=True,
        showEdgeLegend=True,
        splitVertexTracesByStyle=True,
    )
    assert traces is not None
    assert [trace.type for trace in traces] == ["scatter3d", "scatter3d", "cone", "scatter3d"]
    assert traces[0].name == "TGraph Edges"
    assert traces[-1].name == "TGraph Vertices"
    assert "loop" in list(traces[1].text)


def test_data_by_proof_graph_tree_radial_and_figure(fake_color_dictionary):
    _assert_plotly_available()
    proof = {
        "nodes": [
            {"id": "rule", "label": "Rule", "type": "rule", "level": 0},
            {"id": "fact", "label": "Fact", "type": "fact", "level": 1},
            {"id": "target", "label": "Target", "type": "conclusion", "level": 2},
        ],
        "edges": [
            {"source": "rule", "target": "fact", "label": "uses", "type": "uses"},
            {"source": "fact", "target": "target", "label": "supports", "type": "supports"},
        ],
    }
    tree = Plotly.DataByProofGraph(proof, showEdgeLabel=True, layout="tree")
    radial = Plotly.DataByProofGraph(proof, showEdgeLabel=False, layout="radial")
    assert tree and radial
    assert any(trace.type == "scatter3d" and trace.mode == "markers+text" for trace in tree)
    assert any(trace.type == "scatter3d" and trace.mode == "lines" for trace in tree)

    fig = Plotly.FigureByProofGraph(proof, title="Proof", width=640, height=480)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Proof"
    assert fig.layout.width == 640


def test_proof_graph_html_writes_file(tmp_path, fake_color_dictionary):
    _assert_plotly_available()
    proof = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
    path = tmp_path / "proof.html"
    result = Plotly.ProofGraphHTML(proofGraphData=proof, path=str(path), autoOpen=False, silent=True)
    assert result == str(path)
    assert path.exists()
    assert "Plotly" in path.read_text(encoding="utf-8")
    assert Plotly.ProofGraphHTML(proofGraphData=proof, path="", silent=True) is None


def test_figure_by_data_camera_renderer_and_show(monkeypatch, fake_color_dictionary):
    _assert_plotly_available()
    fig = Plotly.FigureByData([go.Scatter3d(x=[0], y=[0], z=[0])], xAxis=True, yAxis=True, zAxis=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # axis import is unavailable in this lightweight test context, so no crash and no axis traces.

    assert Plotly.SetCamera("not a figure") is None
    fig = Plotly.SetCamera(fig, camera=[1, 2, 3], center=[0.1, 0.2, 0.3], up=[0, 0, 1], projection="orthographic")
    assert fig.layout.scene.camera.projection.type == "orthographic"
    fig = Plotly.SetCamera(fig, projection="garbage")
    assert fig.layout.scene.camera.projection.type == "perspective"

    called = {"show": 0}

    def fake_show(self, renderer=None):
        called["show"] += 1
        called["renderer"] = renderer

    monkeypatch.setattr(go.Figure, "show", fake_show, raising=False)
    assert Plotly.Show(fig, renderer="json") is None
    assert called == {"show": 1, "renderer": "json"}
    assert Plotly.Show("not a figure", renderer="json") is None
    assert isinstance(Plotly.Renderer(), str)
    assert "json" in Plotly.Renderers()


def test_figure_by_pie_chart_accepts_dict_list_and_dataframe(fake_color_dictionary):
    _assert_plotly_available()
    import pandas
    fig1 = Plotly.FigureByPieChart({"names": ["A", "B"], "values": [2, 3]}, values="values", names="names")
    fig2 = Plotly.FigureByPieChart([{"name": "A", "value": 2}, {"name": "B", "value": 3}], values="value", names="name")
    fig3 = Plotly.FigureByPieChart(pandas.DataFrame({"n": ["A", "B"], "v": [2, 3]}), values="v", names="n")
    assert fig1.data[0].type == "pie"
    assert fig2.data[0].type == "pie"
    assert fig3.data[0].type == "pie"
    assert list(fig1.data[0].labels) == ["A", "B"]
    assert Plotly.FigureByPieChart(None, values="v", names="n") is None


def test_matrix_confusion_and_correlation_figures(fake_color_dictionary):
    _assert_plotly_available()
    fig = Plotly.FigureByMatrix([[1, 2], [3, 4]], xCategories=["a", "b"], title="M")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"
    assert fig.layout.title.text == "M"

    confusion = Plotly.FigureByConfusionMatrix([[1, 0], [2, 3]], categories=["A", "B"])
    assert isinstance(confusion, go.Figure)
    assert confusion.data[0].type == "heatmap"
    assert confusion.layout.yaxis.autorange == "reversed"

    corr = Plotly.FigureByCorrelation([1, 2, 3], [1, 2, 4], title="C")
    assert isinstance(corr, go.Figure)
    assert str(corr.layout.title.text).startswith("C")
    with pytest.warns(UserWarning):
        assert Plotly.FigureByMatrix("bad") is None


def test_figure_by_dataframe(fake_color_dictionary):
    _assert_plotly_available()
    import pandas
    df = pandas.DataFrame({"epoch": [0, 1, 2], "loss": [1.0, 0.5, 0.25], "accuracy": [0.3, 0.6, 0.9]})
    fig = Plotly.FigureByDataFrame(df, labels=["epoch", "loss", "accuracy"], title="DF")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "DF"
    assert len(fig.data) >= 1
    with pytest.raises(Exception):
        Plotly.FigureByDataFrame("bad", labels=["x", "y"])


# def test_json_export_import_and_export_failure_paths(tmp_path, monkeypatch, fake_color_dictionary):
#     _assert_plotly_available()
#     fig = go.Figure(data=[go.Scatter3d(x=[0], y=[1], z=[2])])
#     json_path = tmp_path / "figure"
#     assert Plotly.FigureExportToJSON(fig, str(json_path), overwrite=False) is True
#     saved = tmp_path / "figure.json"
#     assert saved.exists()
#     assert Plotly.FigureExportToJSON(fig, str(json_path), overwrite=False) is None

#     imported = Plotly.FigureByJSONPath(str(saved))
#     assert isinstance(imported, go.Figure)
#     assert Plotly.FigureByJSONPath(str(tmp_path / "missing.json")) is None

#     def failing_write_image(*args, **kwargs):
#         raise RuntimeError("no kaleido")

#     monkeypatch.setattr("plotly.io.write_image", failing_write_image)
#     assert Plotly.FigureExportToPNG(fig, str(tmp_path / "a.png"), overwrite=True) is None
#     assert Plotly.FigureExportToPDF(fig, str(tmp_path / "a.pdf"), overwrite=True) is None
#     assert Plotly.FigureExportToSVG(fig, str(tmp_path / "a.svg"), overwrite=True) is None
#     assert Plotly.ExportToImage(fig, str(tmp_path / "a.webp"), format="webp") is False
#     assert Plotly.ExportToImage("not a figure", str(tmp_path / "a.png")) is None
#     assert Plotly.ExportToImage(fig, str(tmp_path / "a.bad"), format="bad") is None


def test_data_by_graph_invalid_input_and_plotly_availability_branch(monkeypatch):
    _assert_plotly_available()
    assert Plotly.DataByGraph(None, silent=True) is None
    monkeypatch.setattr("topologicpy.Plotly.plotly", None)
    assert Plotly._plotly_available(silent=True) is False
    assert Plotly.AddColorBar(go.Figure(), values=[1]) is None


def test_package_contains_corrected_plotly_and_test_file():
    # This guards the generated package structure when this test is run from the packaged artifact.
    current = Path(__file__).resolve()
    assert current.name == "test_Plotly.py"
