# Auto-generated PythonOCC backend parity test
# Copied from test_Grid.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Grid."""

import math

import pytest

from topologicpy.Cluster import Cluster
from topologicpy.Dictionary import Dictionary
from topologicpy.Face import Face
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _rectangle(width=4, length=2):
    try:
        return Face.Rectangle(width=width, length=length, placement="lowerleft", silent=True)
    except TypeError:
        try:
            return Face.Rectangle(width=width, length=length, silent=True)
        except TypeError:
            return Face.Rectangle(width=width, length=length)


def _cluster_topologies(cluster):
    if cluster is None:
        return []
    try:
        topologies = Cluster.Topologies(cluster, silent=True)
    except TypeError:
        try:
            topologies = Cluster.Topologies(cluster)
        except Exception:
            topologies = None
    except Exception:
        topologies = None
    return topologies or []


def _edges(topology):
    if topology is None:
        return []
    try:
        if Topology.IsInstance(topology, "Edge"):
            return [topology]
    except Exception:
        pass
    try:
        edges = Topology.Edges(topology, silent=True)
    except TypeError:
        try:
            edges = Topology.Edges(topology)
        except Exception:
            edges = None
    except Exception:
        edges = None
    if edges:
        return edges
    return [t for t in _cluster_topologies(topology) if Topology.IsInstance(t, "Edge")]


def _vertices(topology):
    if topology is None:
        return []
    try:
        if Topology.IsInstance(topology, "Vertex"):
            return [topology]
    except Exception:
        pass
    try:
        vertices = Topology.Vertices(topology, silent=True)
    except TypeError:
        try:
            vertices = Topology.Vertices(topology)
        except Exception:
            vertices = None
    except Exception:
        vertices = None
    if vertices:
        return vertices
    return [t for t in _cluster_topologies(topology) if Topology.IsInstance(t, "Vertex")]


def _dictionary_value(topology, key, default=None):
    try:
        dictionary = Topology.Dictionary(topology)
    except Exception:
        dictionary = None
    try:
        return Dictionary.ValueAtKey(dictionary, key, defaultValue=default, silent=True)
    except TypeError:
        try:
            return Dictionary.ValueAtKey(dictionary, key, default)
        except TypeError:
            try:
                value = Dictionary.ValueAtKey(dictionary, key)
                return default if value is None else value
            except Exception:
                return default
    except Exception:
        return default


def _xyz(vertex):
    def _coord(func):
        try:
            return func(vertex, mantissa=6)
        except TypeError:
            return func(vertex)

    return (_coord(Vertex.X), _coord(Vertex.Y), _coord(Vertex.Z))


def _value_multiset(topologies, key):
    return sorted(_dictionary_value(t, key) for t in topologies)


def test_private_range_helpers_are_defensive():
    assert Grid._Tolerance(None) == pytest.approx(0.0001)
    assert Grid._Tolerance(-0.25) == pytest.approx(0.25)
    assert Grid._Tolerance(0) > 0

    assert Grid._FloatList(0.25) == [0.25]
    assert Grid._FloatList(["2", 1, 0.5]) == [0.5, 1.0, 2.0]
    assert Grid._FloatList([object()]) is None

    assert Grid._ParameterList([0, 0.5, 1]) == [0.0, 0.5, 1.0]
    assert Grid._ParameterList([-0.01, 0.5]) is None
    assert Grid._ParameterList([0.5, 1.01]) is None

    assert Grid._Span([2]) == [1.5, 2.5]
    assert Grid._Normalize([0, 0, 0], tolerance=0.0001) is None
    assert Grid._Normalize([3, 0, 0], tolerance=0.0001) == [1.0, 0.0, 0.0]


def test_edges_by_parameters_rectangular_face_counts_and_metadata():
    face = _rectangle(width=4, length=2)
    grid = Grid.EdgesByParameters(face, uRange=[0, 0.5, 1], vRange=[0, 1], clip=False)

    edges = _edges(grid)
    assert len(edges) == 5
    assert _value_multiset(edges, "dir") == ["u", "u", "u", "v", "v"]
    assert _value_multiset(edges, "offset") == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_edges_by_parameters_rejects_invalid_inputs():
    face = _rectangle(width=4, length=2)

    assert Grid.EdgesByParameters(None) is None
    assert Grid.EdgesByParameters(face, uRange=[-0.1], vRange=[0], clip=False) is None
    assert Grid.EdgesByParameters(face, uRange=[0], vRange=[1.1], clip=False) is None
    assert Grid.EdgesByParameters(face, uRange="not-a-number", vRange=[0], clip=False) is None


def test_vertices_by_parameters_rectangular_face_counts_coordinates_and_metadata():
    face = _rectangle(width=4, length=2)
    grid = Grid.VerticesByParameters(face, uRange=[0, 0.5, 1], vRange=[0, 1], clip=False, silent=True)

    vertices = _vertices(grid)
    assert len(vertices) == 6
    assert _value_multiset(vertices, "u") == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    assert _value_multiset(vertices, "v") == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    coords = {_xyz(v) for v in vertices}
    assert len(coords) == 6


def test_vertices_by_parameters_rejects_invalid_inputs():
    face = _rectangle(width=4, length=2)

    assert Grid.VerticesByParameters(None, silent=True) is None
    assert Grid.VerticesByParameters(face, uRange=[-0.1], vRange=[0], silent=True) is None
    assert Grid.VerticesByParameters(face, uRange=[0], vRange=[1.1], silent=True) is None
    assert Grid.VerticesByParameters(face, uRange=[0], vRange="bad", silent=True) is None


def test_vertices_by_distances_world_grid_coordinates_and_metadata():
    grid = Grid.VerticesByDistances(face=None, uRange=[0, 1], vRange=[0, 2], clip=False, silent=True)

    vertices = _vertices(grid)
    assert len(vertices) == 4
    assert _value_multiset(vertices, "u") == [0.0, 0.0, 1.0, 1.0]
    assert _value_multiset(vertices, "v") == [0.0, 0.0, 2.0, 2.0]
    assert {_xyz(v) for v in vertices} == {
        (0.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 2.0, 0.0),
    }


def test_vertices_by_distances_rejects_invalid_ranges_and_preserves_old_alias():
    assert Grid.VerticesByDistances(face=None, uRange="bad", vRange=[0], silent=True) is None
    assert Grid.VerticesByDistances(face=None, uRange=[], vRange=[0], silent=True) is None

    current = Grid.VerticesByDistances(face=None, uRange=[0, 1], vRange=[0, 1], silent=True)
    legacy = Grid.VerticesByDistances_old(face=None, uRange=[0, 1], vRange=[0, 1])
    assert len(_vertices(current)) == len(_vertices(legacy)) == 4


def test_edges_by_distances_world_grid_no_face_creates_edges_and_metadata():
    grid = Grid.EdgesByDistances(face=None, uRange=[0, 1], vRange=[0, 2], clip=False)

    edges = _edges(grid)
    assert len(edges) == 4
    assert _value_multiset(edges, "dir") == ["u", "u", "v", "v"]
    assert _value_multiset(edges, "offset") == [0.0, 0.0, 1.0, 2.0]


def test_edges_by_distances_world_grid_accepts_scalar_ranges():
    grid = Grid.EdgesByDistances(face=None, uRange=0, vRange=1, clip=False)

    edges = _edges(grid)
    assert len(edges) == 2
    assert _value_multiset(edges, "dir") == ["u", "v"]
    assert _value_multiset(edges, "offset") == [0.0, 1.0]


def test_edges_by_distances_face_converts_physical_distances_to_parameters():
    face = _rectangle(width=4, length=2)
    grid = Grid.EdgesByDistances(face=face, uRange=[0, 2, 4], vRange=[0, 1, 2], clip=False, mantissa=6)

    edges = _edges(grid)
    assert len(edges) == 6
    assert _value_multiset(edges, "dir") == ["u", "u", "u", "v", "v", "v"]
    assert _value_multiset(edges, "offset") == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]


def test_edges_by_distances_face_ignores_out_of_domain_distances():
    face = _rectangle(width=4, length=2)
    grid = Grid.EdgesByDistances(face=face, uRange=[-1, 0, 2, 99], vRange=[], clip=False, mantissa=6)

    edges = _edges(grid)
    assert len(edges) == 2
    assert _value_multiset(edges, "dir") == ["u", "u"]
    assert _value_multiset(edges, "offset") == [0.0, 0.5]


def test_vertices_by_distances_clip_keeps_boundary_vertices():
    face = _rectangle(width=4, length=2)
    grid = Grid.VerticesByDistances(face=face, uRange=[0, 4], vRange=[0, 2], clip=True, silent=True)

    vertices = _vertices(grid)
    assert len(vertices) == 4
    assert _value_multiset(vertices, "u") == [0.0, 0.0, 4.0, 4.0]
    assert _value_multiset(vertices, "v") == [0.0, 0.0, 2.0, 2.0]


def test_source_replacement_guard_for_corrected_distance_grid_path():
    """Fail clearly if an older Grid.py without the world-grid fix is imported."""
    assert Grid.EdgesByDistances(face=None, uRange=[0], vRange=[0], clip=False) is not None
