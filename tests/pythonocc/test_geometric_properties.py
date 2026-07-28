# Copyright (C) 2026
# PythonOCC backend GeometricProperties parity tests.
#
# Run with: TOPOLOGICPY_CORE_BACKEND=pythonocc pytest tests/pythonocc/test_geometric_properties.py -v

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
CellComplex = pytest.importorskip("topologicpy.CellComplex").CellComplex
Topology = pytest.importorskip("topologicpy.Topology").Topology
GeometricProperties = pytest.importorskip("topologicpy.GeometricProperties").GeometricProperties

TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _suppress_output(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


# ===========================================================================
# Edge properties
# ===========================================================================

class TestEdgeProperties:
    def test_edge_length(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(3, 4))
        length = GeometricProperties.Length(e)
        assert length == pytest.approx(5.0, abs=TOLERANCE)

    def test_edge_direction(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(1, 0))
        direction = GeometricProperties.Direction(e)
        assert direction is not None
        assert len(direction) == 3

    def test_edge_tangent(self):
        e = Edge.ByStartVertexEndVertex(_v(0, 0), _v(0, 1))
        tangent = GeometricProperties.Tangent(e, 0.5)
        assert tangent is not None


# ===========================================================================
# Wire properties
# ===========================================================================

class TestWireProperties:
    def test_wire_length(self):
        w = Wire.Rectangle(2.0, 3.0)
        length = GeometricProperties.Length(w)
        assert length == pytest.approx(10.0, abs=TOLERANCE)  # 2*(2+3)

    def test_wire_perimeter(self):
        w = Wire.Rectangle(1.0, 1.0)
        perimeter = GeometricProperties.Perimeter(w)
        assert perimeter == pytest.approx(4.0, abs=TOLERANCE)


# ===========================================================================
# Face properties
# ===========================================================================

class TestFaceProperties:
    def test_face_area_rectangle(self):
        f = Face.Rectangle(2.0, 3.0)
        area = GeometricProperties.Area(f)
        assert area == pytest.approx(6.0, abs=TOLERANCE)

    def test_face_area_circle(self):
        f = Face.Circle(radius=1.0)
        area = GeometricProperties.Area(f)
        assert area == pytest.approx(math.pi, abs=TOLERANCE)

    def test_face_perimeter(self):
        f = Face.Rectangle(1.0, 1.0)
        perimeter = GeometricProperties.Perimeter(f)
        assert perimeter == pytest.approx(4.0, abs=TOLERANCE)

    def test_face_compactness(self):
        f = Face.Circle(radius=1.0)
        compactness = GeometricProperties.Compactness(f)
        assert compactness == pytest.approx(1.0, abs=TOLERANCE)  # Circle is most compact

    def test_face_compactness_square(self):
        f = Face.Rectangle(1.0, 1.0)
        compactness = GeometricProperties.Compactness(f)
        # Square compactness = 4*pi*area / perimeter^2 = 4*pi*1 / 16 = pi/4
        assert compactness == pytest.approx(math.pi / 4, abs=TOLERANCE)

    def test_face_normal_vector(self):
        f = Face.Rectangle(1.0, 1.0)
        normal = GeometricProperties.NormalVector(f)
        assert normal is not None
        assert len(normal) == 3
        # For XY plane face, normal should be along Z
        assert abs(normal[2]) > 0.9

    def test_face_centroid(self):
        f = Face.Rectangle(2.0, 2.0)
        centroid = GeometricProperties.Centroid(f)
        assert centroid is not None
        assert Vertex.X(centroid) == pytest.approx(1.0, abs=TOLERANCE)
        assert Vertex.Y(centroid) == pytest.approx(1.0, abs=TOLERANCE)

    def test_face_internal_vertex(self):
        f = Face.Rectangle(2.0, 2.0)
        iv = GeometricProperties.InternalVertex(f)
        assert iv is not None
        # Should be inside the face
        assert 0 < Vertex.X(iv) < 2.0
        assert 0 < Vertex.Y(iv) < 2.0


# ===========================================================================
# Shell properties
# ===========================================================================

class TestShellProperties:
    def test_shell_area(self):
        c = Cell.Prism(1.0, 1.0, 1.0)
        s = Cell.Shells(c)[0]
        area = GeometricProperties.Area(s)
        assert area == pytest.approx(6.0, abs=TOLERANCE)  # 6 faces of 1x1

    def test_shell_external_boundary_length(self):
        c = Cell.Prism(1.0, 1.0, 1.0)
        s = Cell.Shells(c)[0]
        eb = Shell.ExternalBoundary(s)
        length = GeometricProperties.Length(eb)
        assert length > 0


# ===========================================================================
# Cell properties
# ===========================================================================

class TestCellProperties:
    def test_cell_volume_cube(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        volume = GeometricProperties.Volume(c)
        assert volume == pytest.approx(8.0, abs=TOLERANCE)

    def test_cell_volume_sphere(self):
        c = Cell.Sphere(radius=1.0)
        volume = GeometricProperties.Volume(c)
        expected = (4/3) * math.pi * (1.0 ** 3)
        assert volume == pytest.approx(expected, abs=0.01)

    def test_cell_surface_area_cube(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        area = GeometricProperties.SurfaceArea(c)
        assert area == pytest.approx(24.0, abs=TOLERANCE)  # 6 * 4

    def test_cell_centroid(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        centroid = GeometricProperties.Centroid(c)
        assert centroid is not None
        assert Vertex.X(centroid) == pytest.approx(1.0, abs=TOLERANCE)
        assert Vertex.Y(centroid) == pytest.approx(1.0, abs=TOLERANCE)
        assert Vertex.Z(centroid) == pytest.approx(1.0, abs=TOLERANCE)

    def test_cell_internal_vertex(self):
        c = Cell.Prism(2.0, 2.0, 2.0)
        iv = GeometricProperties.InternalVertex(c)
        assert iv is not None
        # Should be inside the cell
        assert 0 < Vertex.X(iv) < 2.0
        assert 0 < Vertex.Y(iv) < 2.0
        assert 0 < Vertex.Z(iv) < 2.0

    def test_cell_compactness(self):
        c = Cell.Sphere(radius=1.0)
        compactness = GeometricProperties.Compactness(c)
        assert compactness == pytest.approx(1.0, abs=TOLERANCE)  # Sphere is most compact


# ===========================================================================
# CellComplex properties
# ===========================================================================

class TestCellComplexProperties:
    def test_cellcomplex_volume(self):
        c1 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Topology.Translate(c2, 1, 0, 0)
        cc = CellComplex.ByCells([c1, c2])
        volume = GeometricProperties.Volume(cc)
        assert volume == pytest.approx(2.0, abs=TOLERANCE)

    def test_cellcomplex_centroid(self):
        c1 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Cell.Prism(1.0, 1.0, 1.0)
        c2 = Topology.Translate(c2, 2, 0, 0)
        cc = CellComplex.ByCells([c1, c2])
        centroid = GeometricProperties.Centroid(cc)
        # Centroid should be at x=1.5 (midpoint between 0.5 and 2.5)
        assert Vertex.X(centroid) == pytest.approx(1.5, abs=TOLERANCE)
