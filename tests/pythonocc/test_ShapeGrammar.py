# Auto-generated PythonOCC backend parity test
# Copied from test_ShapeGrammar.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.ShapeGrammar.

These tests use small fake TopologicPy geometry modules so they can run without
TopologicCore. They focus on the ShapeGrammar control flow, operation dispatch,
validation, and regression fixes in the audited source.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeVertex:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"FakeVertex({self.x}, {self.y}, {self.z})"


class FakeTopo:
    def __init__(self, name, dictionary=None, signature=None, bbox=None, vertices=None, children=None, history=None):
        self.name = str(name)
        self.dictionary = dict(dictionary or {})
        self.signature = signature if signature is not None else self.name
        self.bbox = bbox if bbox is not None else [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        self.vertices = list(vertices) if vertices is not None else [
            FakeVertex(self.bbox[0], self.bbox[1], self.bbox[2]),
            FakeVertex(self.bbox[3], self.bbox[4], self.bbox[5]),
        ]
        self.children = list(children or [])
        self.history = list(history or [])

    def clone(self, name=None, history_item=None):
        history = list(self.history)
        if history_item is not None:
            history.append(history_item)
        return FakeTopo(
            name or self.name,
            dictionary=self.dictionary,
            signature=self.signature,
            bbox=list(self.bbox),
            vertices=list(self.vertices),
            children=list(self.children),
            history=history,
        )

    def __repr__(self):
        return f"FakeTopo({self.name!r})"


@pytest.fixture
def fake_topologicpy(monkeypatch):
    """Install fake TopologicPy submodules required by ShapeGrammar methods."""

    class Topology:
        @staticmethod
        def IsInstance(obj, type_name):
            return isinstance(obj, FakeTopo) and str(type_name).lower() in {"topology", "cluster", "cell", "face"}

        @staticmethod
        def Dictionary(obj):
            return getattr(obj, "dictionary", {}) if isinstance(obj, FakeTopo) else {}

        @staticmethod
        def IsSimilar(a, b):
            if getattr(a, "signature", None) == getattr(b, "signature", None):
                return True, [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
            return False, None

        @staticmethod
        def Transform(obj, matrix):
            return obj.clone(name=f"Transform({obj.name})", history_item=("Transform", matrix))

        @staticmethod
        def Union(a, b):
            return FakeTopo("Union", children=[a, b], history=[("Union", a, b)])

        @staticmethod
        def Difference(a, b):
            return FakeTopo("Difference", children=[a, b], history=[("Difference", a, b)])

        @staticmethod
        def SymmetricDifference(a, b):
            return FakeTopo("SymmetricDifference", children=[a, b], history=[("SymmetricDifference", a, b)])

        @staticmethod
        def Intersect(a, b):
            return FakeTopo("Intersect", children=[a, b], history=[("Intersect", a, b)])

        @staticmethod
        def Merge(a, b):
            return FakeTopo("Merge", children=[a, b], history=[("Merge", a, b)])

        @staticmethod
        def Slice(a, b, tolerance=0.0001):
            return FakeTopo("Slice", children=[a, b], history=[("Slice", a, b, tolerance)])

        @staticmethod
        def Impose(a, b):
            return FakeTopo("Impose", children=[a, b], history=[("Impose", a, b)])

        @staticmethod
        def Imprint(a, b):
            return FakeTopo("Imprint", children=[a, b], history=[("Imprint", a, b)])

        @staticmethod
        def Vertices(obj, silent=True):
            return list(getattr(obj, "vertices", []))

        @staticmethod
        def Translate(obj, x=0, y=0, z=0):
            return obj.clone(name=f"Translate({obj.name})", history_item=("Translate", x, y, z))

        @staticmethod
        def Scale(obj, x=1, y=1, z=1):
            return obj.clone(name=f"Scale({obj.name})", history_item=("Scale", x, y, z))

        @staticmethod
        def Rotate(obj, axis=None, angle=0):
            return obj.clone(name=f"Rotate({obj.name})", history_item=("Rotate", tuple(axis or []), angle))

        @staticmethod
        def BoundingBox(obj):
            bb = list(getattr(obj, "bbox", [0, 0, 0, 1, 1, 1]))
            return FakeTopo(
                f"BoundingBox({obj.name})",
                dictionary={"xmin": bb[0], "ymin": bb[1], "zmin": bb[2], "xmax": bb[3], "ymax": bb[4], "zmax": bb[5]},
                bbox=bb,
            )

        @staticmethod
        def Centroid(obj):
            bb = list(getattr(obj, "bbox", [0, 0, 0, 0, 0, 0]))
            return FakeVertex((bb[0] + bb[3]) * 0.5, (bb[1] + bb[4]) * 0.5, (bb[2] + bb[5]) * 0.5)

        @staticmethod
        def Place(obj, originA=None, originB=None):
            return obj.clone(name=f"Place({obj.name})", history_item=("Place", originA, originB))

    class Dictionary:
        @staticmethod
        def ValueAtKey(d, key, default=None):
            return d.get(key, default) if isinstance(d, dict) else default

    class Vertex:
        @staticmethod
        def X(v, mantissa=None):
            return round(v.x, mantissa) if mantissa is not None else v.x

        @staticmethod
        def Y(v, mantissa=None):
            return round(v.y, mantissa) if mantissa is not None else v.y

        @staticmethod
        def Z(v, mantissa=None):
            return round(v.z, mantissa) if mantissa is not None else v.z

        @staticmethod
        def ByCoordinates(x=0, y=0, z=0):
            return FakeVertex(x, y, z)

        @staticmethod
        def Origin():
            return FakeVertex(0, 0, 0)

    class Face:
        @staticmethod
        def Rectangle(origin=None, width=1, length=1, direction=None):
            origin = origin or FakeVertex()
            return FakeTopo(
                "Rectangle",
                bbox=[origin.x, origin.y, origin.z, origin.x + float(width), origin.y + float(length), origin.z],
                history=[("Rectangle", width, length, tuple(direction or [0, 0, 1]))],
            )

    class Cluster:
        @staticmethod
        def ByTopologies(items):
            return FakeTopo("Cluster", children=list(items or []), history=[("Cluster", len(list(items or [])))])

    class Cell:
        @staticmethod
        def Cylinder(radius=1, height=1, placement="center"):
            return FakeTopo("Cylinder", history=[("Cylinder", radius, height, placement)])

        @staticmethod
        def Cone(baseRadius=1, topRadius=0, height=1, placement="center"):
            return FakeTopo("Cone", history=[("Cone", baseRadius, topRadius, height, placement)])

    class Plotly:
        @staticmethod
        def DataByTopology(topology):
            return [{"topology": getattr(topology, "name", "unknown")}]

        @staticmethod
        def FigureByData(data):
            return {"figure_data": data}

    for mod_name, cls_name, cls in [
        ("topologicpy.Topology", "Topology", Topology),
        ("topologicpy.Dictionary", "Dictionary", Dictionary),
        ("topologicpy.Vertex", "Vertex", Vertex),
        ("topologicpy.Face", "Face", Face),
        ("topologicpy.Cluster", "Cluster", Cluster),
        ("topologicpy.Cell", "Cell", Cell),
        ("topologicpy.Plotly", "Plotly", Plotly),
    ]:
        module = types.ModuleType(mod_name)
        setattr(module, cls_name, cls)
        monkeypatch.setitem(sys.modules, mod_name, module)

    return types.SimpleNamespace(
        Topology=Topology,
        Dictionary=Dictionary,
        Vertex=Vertex,
        Face=Face,
        Cluster=Cluster,
        Cell=Cell,
        Plotly=Plotly,
        topo=lambda name, **kwargs: FakeTopo(name, **kwargs),
    )


@pytest.fixture
def sg(fake_topologicpy):
    from topologicpy.ShapeGrammar import ShapeGrammar

    return ShapeGrammar()


def matrix(dx=0):
    return [[1, 0, 0, dx], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_operation_titles_and_lookup_are_case_insensitive(sg):
    titles = sg.OperationTitles()
    assert titles[0] == "Replace"
    assert "Symmetric Difference" in titles
    assert sg.OperationByTitle("replace")["title"] == "Replace"
    assert sg.OperationByTitle("sym")["title"] == "Symmetric Difference"
    assert sg.OperationByTitle("DIFFERENCE")["title"] == "Difference"
    assert sg.OperationByTitle(None) is None
    assert sg.OperationByTitle("") is None


def test_matrix_validation_accepts_numeric_tuples_and_rejects_bad_values(sg):
    tuple_matrix = ((1, "0", 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    assert sg._is_4x4_matrix(tuple_matrix)
    copied = sg._copy_matrix(tuple_matrix)
    assert copied == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    assert not sg._is_4x4_matrix([[1, 0], [0, 1]])
    assert not sg._is_4x4_matrix([[1, 0, 0, 0], [0, "bad", 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def test_add_rule_default_replace_and_validation_paths(sg, fake_topologicpy):
    a = fake_topologicpy.topo("input", signature="room")
    b = fake_topologicpy.topo("output", signature="room")

    assert sg.AddRule(a, b, matrix=matrix(2), title="r1") is None
    assert len(sg.rules) == 1
    rule = sg.rules[0]
    assert rule["title"] == "r1"
    assert rule["operation"]["title"] == "Replace"
    assert rule["matrix"] == matrix(2)
    assert rule["matrix"] is not matrix(2)

    assert sg.AddRule("not topology", b, silent=True) is None
    assert sg.AddRule(a, "not topology", silent=True) is None
    assert sg.AddRule(a, b, operation="not an operation", silent=True) is None
    assert sg.AddRule(a, b, matrix=[[1]], silent=True) is None
    assert len(sg.rules) == 1


def test_add_rule_accepts_operation_titles_and_copied_divide_overrides(sg, fake_topologicpy):
    a = fake_topologicpy.topo("input")
    b = fake_topologicpy.topo("output")
    copied_divide = {"title": "Divide", "uSides": 3, "vSides": 4, "wSides": 5}

    sg.AddRule(a, b, operation="Union")
    sg.AddRule(a, None, operation=copied_divide)

    assert sg.rules[0]["operation"]["title"] == "Union"
    assert sg.rules[1]["operation"]["title"] == "Divide"
    assert sg.rules[1]["operation"]["uSides"] == 3
    assert sg.rules[1]["operation"]["vSides"] == 4
    assert sg.rules[1]["operation"]["wSides"] == 5


def test_applicable_rules_use_dictionary_keys_and_similarity_tuple(sg, fake_topologicpy):
    pattern = fake_topologicpy.topo("pattern", signature="cell", dictionary={"type": "A", "level": 2})
    other = fake_topologicpy.topo("other", signature="cell", dictionary={"type": "B", "level": 2})
    output = fake_topologicpy.topo("output")
    target = fake_topologicpy.topo("target", signature="cell", dictionary={"type": "A", "level": 2})

    sg.AddRule(pattern, output, title="match")
    sg.AddRule(other, output, title="no semantic match")

    rules, transforms = sg.ApplicableRules(target, keys=["type", "level"])
    assert [r["title"] for r in rules] == ["match"]
    assert transforms[0][0][3] == 10
    assert sg.ApplicableRules("bad", silent=True) is None


def test_apply_rule_none_returns_topology_or_final_transform(sg, fake_topologicpy):
    topo = fake_topologicpy.topo("target")
    assert sg.ApplyRule(topo) is topo
    transformed = sg.ApplyRule(topo, matrix=matrix(9))
    assert transformed.name == "Transform(target)"
    assert transformed.history[-1] == ("Transform", matrix(9))
    assert sg.ApplyRule("bad", silent=True) is None
    assert sg.ApplyRule(topo, matrix=[[1]], silent=True) is None


def test_apply_rule_default_replace_and_rule_matrix(sg, fake_topologicpy):
    input_topo = fake_topologicpy.topo("input")
    output_topo = fake_topologicpy.topo("output")
    sg.AddRule(input_topo, output_topo, matrix=matrix(4))

    result = sg.ApplyRule(fake_topologicpy.topo("target"), sg.rules[0])
    assert result.name == "Transform(output)"
    assert result.history[-1] == ("Transform", matrix(4))

    final = sg.ApplyRule(fake_topologicpy.topo("target"), sg.rules[0], matrix=matrix(8))
    assert final.name == "Transform(Transform(output))"
    assert final.history[-1] == ("Transform", matrix(8))


def test_apply_rule_dispatches_binary_operations_and_symmetric_difference_first(sg, fake_topologicpy):
    input_topo = fake_topologicpy.topo("input")
    output_topo = fake_topologicpy.topo("output")
    expectations = {
        "Union": "Union",
        "Difference": "Difference",
        "Symmetric Difference": "SymmetricDifference",
        "Intersect": "Intersect",
        "Merge": "Merge",
        "Slice": "Slice",
        "Impose": "Impose",
        "Imprint": "Imprint",
    }
    for title, expected_name in expectations.items():
        rule = {"input": input_topo, "output": output_topo, "operation": title, "matrix": None}
        result = sg.ApplyRule(input_topo, rule)
        assert result.name == expected_name


def test_apply_rule_transform_operation_uses_rule_matrix_or_identity_behaviour(sg, fake_topologicpy):
    input_topo = fake_topologicpy.topo("input")
    rule = {"input": input_topo, "output": None, "operation": "Transform", "matrix": matrix(3)}
    result = sg.ApplyRule(input_topo, rule)
    assert result.name == "Transform(input)"
    assert result.history[-1] == ("Transform", matrix(3))

    no_matrix_rule = {"input": input_topo, "output": None, "operation": "Transform", "matrix": None}
    assert sg.ApplyRule(input_topo, no_matrix_rule) is input_topo


def test_apply_rule_divide_dispatches_slice_when_sides_create_cutters(sg, fake_topologicpy):
    input_topo = fake_topologicpy.topo("box", bbox=[0, 0, 0, 10, 8, 6])
    divide_op = {"title": "Divide", "uSides": 2, "vSides": 2, "wSides": 2}
    rule = {"input": input_topo, "output": None, "operation": divide_op, "matrix": None}
    result = sg.ApplyRule(input_topo, rule)
    assert result.name == "Slice"
    assert result.history[-1][0] == "Slice"

    degenerate = fake_topologicpy.topo("point", bbox=[0, 0, 0, 0, 0, 0], vertices=[FakeVertex(0, 0, 0)])
    deg_rule = {"input": degenerate, "output": None, "operation": divide_op, "matrix": None}
    assert sg.ApplyRule(degenerate, deg_rule) is degenerate


def test_apply_rule_rejects_invalid_rule_inputs_outputs_operations_and_matrices(sg, fake_topologicpy):
    topo = fake_topologicpy.topo("input")
    output = fake_topologicpy.topo("output")
    assert sg.ApplyRule(topo, rule="bad", silent=True) is None
    assert sg.ApplyRule(topo, {"input": "bad", "output": output, "operation": "Replace"}, silent=True) is None
    assert sg.ApplyRule(topo, {"input": topo, "output": "bad", "operation": "Replace"}, silent=True) is None
    assert sg.ApplyRule(topo, {"input": topo, "output": output, "operation": "Invalid"}, silent=True) is None
    assert sg.ApplyRule(topo, {"input": topo, "output": output, "operation": "Replace", "matrix": [[1]]}, silent=True) is None


def test_cluster_and_figure_helpers(fake_topologicpy, sg):
    input_topo = fake_topologicpy.topo("input", bbox=[0, 0, 0, 2, 4, 6])
    output_topo = fake_topologicpy.topo("output", bbox=[0, 0, 0, 1, 1, 1])

    cluster = sg.ClusterByInputOutput(input_topo, output_topo)
    assert isinstance(cluster, FakeTopo)
    assert cluster.name.startswith("Place(")
    assert sg.ClusterByInputOutput("bad", output_topo, silent=True) is None
    assert sg.ClusterByInputOutput(input_topo, "bad", silent=True) is None

    figure = sg.FigureByInputOutput(input_topo, output_topo)
    assert "figure_data" in figure
    assert sg.FigureByInputOutput("bad", output_topo, silent=True) is None


def test_cluster_by_rule_and_figure_by_rule(fake_topologicpy, sg):
    input_topo = fake_topologicpy.topo("input")
    output_topo = fake_topologicpy.topo("output")
    sg.AddRule(input_topo, output_topo, title="replace")
    rule = sg.rules[0]

    cluster = sg.ClusterByRule(rule)
    assert isinstance(cluster, FakeTopo)
    figure = sg.FigureByRule(rule)
    assert "figure_data" in figure

    assert sg.ClusterByRule("bad", silent=True) is None
    assert sg.FigureByRule("bad", silent=True) is None


def test_public_exports():
    module = importlib.import_module("topologicpy.ShapeGrammar")
    assert module.__all__ == ["ShapeGrammar"]
