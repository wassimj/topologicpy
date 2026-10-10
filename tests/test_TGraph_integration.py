"""Topological HH integration, checked against Depthmap's published D/RA definitions."""
import math
import pytest
from topologicpy.TGraph import TGraph


def graph(n, pairs):
    g = TGraph()
    for i in range(n):
        g.AddVertex({"x": i, "y": 0, "z": 0})
    for a, b in pairs:
        g.AddEdge(a, b)
    return g


def test_path_four_hh_and_default_compatibility():
    g = graph(4, [(0, 1), (1, 2), (2, 3)])
    assert TGraph.Integration(g) == [0, 1, 1, 0]
    # n=4: D=1/3. RA at the ends is 1; at the middle it is 1/3.
    assert TGraph.Integration(g, method="depthmap", colorScale="syntax") == pytest.approx([1/3, 1, 1, 1/3], abs=1e-6)
    assert [v["dictionary"]["integration"] for v in g._vertices] == pytest.approx([1/3, 1, 1, 1/3], abs=1e-6)
    assert all(v["dictionary"]["cc_color"].startswith("#") for v in g._vertices)
    assert TGraph.Integration(g, method=" HH ", normalize=False) == TGraph.Integration(g, method="depthmap")


def test_local_radius_disconnected_and_inactive_nodes():
    g = graph(5, [(0, 1), (1, 2), (2, 3)])
    # Endpoints reach a three-node path at R2: RA=1 and D(3).
    d3 = 3 * math.log2(5/3) - 2
    assert TGraph.Integration(g, method="hh", radius=2) == pytest.approx([d3, 1, 1, d3, -1], abs=1e-6)
    assert TGraph.Integration(g, method="hh", radius=2.9) == TGraph.Integration(g, method="hh", radius=2)
    assert TGraph.Integration(g, method="hh", radius=1) == [-1]*5
    assert TGraph.Integration(g, method="hh", radius=0) == [-1]*5
    assert TGraph.Integration(g, method="hh")[-1] == -1
    assert g._vertices[4]["dictionary"]["cc_color"] == "#7F7F7F"
    g._vertices[4]["active"] = False
    assert TGraph.Integration(g, method="hh", colorKey=None) == pytest.approx([1/3, 1, 1, 1/3], abs=1e-6)


def test_complete_graph_and_small_components_are_undefined():
    assert TGraph.Integration(graph(3, [(0,1),(1,2),(0,2)]), method="hh") == [-1]*3
    assert TGraph.Integration(graph(2, [(0,1)]), method="hh") == [-1]*2
    assert TGraph.Integration(TGraph(), method="hh") == []


def test_graph_cycle_hh_and_no_result_storage():
    g = graph(4, [(0,1),(1,2),(2,3),(3,0)])
    assert TGraph.Integration(g, method="hh", key=None, colorKey=None, mantissa=None) == pytest.approx([1]*4)
    assert all("integration" not in v["dictionary"] for v in g._vertices)


@pytest.mark.parametrize("radius", [-1, float("inf"), float("nan"), True, "3"])
def test_bad_radius(radius):
    assert TGraph.Integration(graph(1, []), method="hh", radius=radius, silent=True) is None


def test_bad_method_and_graph():
    assert TGraph.Integration(graph(1, []), method="unknown", silent=True) is None
    assert TGraph.Integration(None, method="hh", silent=True) is None


def test_hh_matches_stored_depthmap_barnsbury_output():
    import json
    from pathlib import Path
    reference = json.loads((Path(__file__).parent / "fixtures/depthmap/barnsbury_hh.json").read_text())
    g = graph(len(reference["vertex_ids"]), reference["pairs"])
    actual = TGraph.Integration(g, method="hh", colorKey=None, mantissa=None)
    # Original graph stores float32 values; compare before TopologicPy rounding.
    assert actual == pytest.approx(reference["hh"], rel=2e-6, abs=1e-6)
