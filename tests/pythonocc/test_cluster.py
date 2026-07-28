# Copyright (C) 2026
# PythonOCC backend Cluster parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_cluster.py -v

import math
import os
import pytest

os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge
Wire = pytest.importorskip("topologicpy.Wire").Wire
Face = pytest.importorskip("topologicpy.Face").Face
Shell = pytest.importorskip("topologicpy.Shell").Shell
Cell = pytest.importorskip("topologicpy.Cell").Cell
Cluster = pytest.importorskip("topologicpy.Cluster").Cluster
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


def _simple_cluster():
    """Create a cluster of two separate cubes."""
    c1 = Cell.Prism(1.0, 1.0, 1.0)
    c2 = Cell.Prism(1.0, 1.0, 1.0)
    c2 = Topology.Translate(c2, 5.0, 0, 0)  # Far apart
    return Cluster.ByTopologies([c1, c2])


def _mixed_cluster():
    """Create a cluster with different topology types."""
    v = _v(0, 0, 0)
    e = Edge.ByStartVertexEndVertex(_v(2, 0), _v(3, 0))
    f = Face.Rectangle(1.0, 1.0)
    f = Topology.Translate(f, 0, 0, 2)
    return Cluster.ByTopologies([v, e, f])


# ===========================================================================
# Constructors
# ===========================================================================

class TestClusterConstructors:
    def test_by_topologies(self):
        cl = _simple_cluster()
        assert Topology.IsInstance(cl, "Cluster")

    def test_by_topologies_mixed(self):
        cl = _mixed_cluster()
        assert Topology.IsInstance(cl, "Cluster")

    def test_by_vertices(self):
        v1 = _v(0, 0, 0)
        v2 = _v(1, 0, 0)
        v3 = _v(0, 1, 0)
        cl = Cluster.ByVertices([v1, v2, v3])
        assert Topology.IsInstance(cl, "Cluster")

    def test_by_edges(self):
        e1 = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        e2 = Edge.ByStartVertexEndVertex(_v(2, 0), _v(3, 0))
        cl = Cluster.ByEdges([e1, e2])
        assert Topology.IsInstance(cl, "Cluster")

    def test_by_faces(self):
        f1 = Face.Rectangle(1.0, 1.0)
        f2 = Face.Rectangle(1.0, 1.0)
        f2 = Topology.Translate(f2, 5, 0, 0)
        cl = Cluster.ByFaces([f1, f2])
        assert Topology.IsInstance(cl, "Cluster")

    def test_by_cells(self):
        c1 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Topology.Translate(c2, 5, 0, 0)
        cl = Cluster.ByCells([c1, c2])
        assert Topology.IsInstance(cl, "Cluster")


# ===========================================================================
# Accessors
# ===========================================================================

class TestClusterAccessors:
    def test_vertices(self):
        cl = _simple_cluster()
        verts = Cluster.Vertices(cl)
        assert len(verts) >= 8  # At least 8 vertices (2 cubes)

    def test_edges(self):
        cl = _simple_cluster()
        edges = Cluster.Edges(cl)
        assert len(edges) >= 12  # At least 12 edges

    def test_faces(self):
        cl = _simple_cluster()
        faces = Cluster.Faces(cl)
        assert len(faces) >= 6  # At least 6 faces

    def test_cells(self):
        cl = _simple_cluster()
        cells = Cluster.Cells(cl)
        assert len(cells) == 2

    def test_shells(self):
        cl = _simple_cluster()
        shells = Cluster.Shells(cl)
        assert len(shells) >= 2

    def test_contents(self):
        cl = _simple_cluster()
        contents = Cluster.Contents(cl)
        assert len(contents) == 2


# ===========================================================================
# Type checking
# ===========================================================================

class TestClusterType:
    def test_is_instance_cluster(self):
        cl = _simple_cluster()
        assert Topology.IsInstance(cl, "Cluster") is True

    def test_is_not_cell(self):
        cl = _simple_cluster()
        assert Topology.IsInstance(cl, "Cell") is False

    def test_type_returns_cluster(self):
        cl = _simple_cluster()
        assert Topology.Type(cl) == 128  # Cluster type ID


# ===========================================================================
# Geometry
# ===========================================================================

class TestClusterGeometry:
    def test_volume(self):
        cl = _simple_cluster()
        vol = Cluster.Volume(cl)
        assert vol == pytest.approx(2.0, abs=TOLERANCE)  # 2 cubes of volume 1

    def test_area(self):
        cl = _simple_cluster()
        area = Cluster.Area(cl)
        assert area > 0


# ===========================================================================
# Operations
# ===========================================================================

class TestClusterOperations:
    def test_by_analyze(self):
        cl = _simple_cluster()
        analyzed = Cluster.ByAnalyze(cl)
        assert analyzed is not None

    def test_contents_type(self):
        cl = _mixed_cluster()
        contents = Cluster.Contents(cl)
        types = [Topology.Type(c) for c in contents]
        assert 1 in types  # Vertex
        assert 2 in types  # Edge
        assert 8 in types  # Face


# ===========================================================================
# Serialization
# ===========================================================================

class TestClusterSerialization:
    def test_brep_roundtrip(self):
        cl = _simple_cluster()
        brep = Topology.BREPString(cl)
        assert brep is not None
        cl2 = Topology.ByBREPString(brep)
        assert Topology.IsInstance(cl2, "Cluster")
        assert Cluster.Volume(cl2) == pytest.approx(Cluster.Volume(cl), abs=TOLERANCE)
