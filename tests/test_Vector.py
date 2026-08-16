import importlib
import math
import sys
import types

import pytest

from topologicpy.Vector import Vector


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _apply_matrix(matrix, vector):
    x, y, z = vector
    values = [x, y, z, 1.0]
    return [sum(matrix[i][j] * values[j] for j in range(4)) for i in range(3)]




def test_arithmetic_sum_average_reverse_and_multiply():
    assert Vector.Add([1, 2, 3], (4, 5, 6)) == [5.0, 7.0, 9.0]
    assert Vector.Subtract((5, 7, 9), [1, 2, 3]) == [4.0, 5.0, 6.0]
    assert Vector.Multiply([1, -2, 3], 2.5) == [2.5, -5.0, 7.5]
    assert Vector.Multiply([1, -2, 3], 0.0) == [0.0, 0.0, 0.0]
    assert Vector.Reverse((1, -2, 3)) == [-1.0, 2.0, -3.0]
    assert Vector.Sum([[1, 2, 3], (4, 5, 6), [7, 8, 9]]) == [12.0, 15.0, 18.0]
    assert Vector.Average([[1, 2, 3], (4, 5, 6), [7, 8, 9]]) == [4.0, 5.0, 6.0]
    assert Vector.Add(None, [1, 2, 3]) is None
    assert Vector.Subtract([1, 2, 3], None) is None


def test_coordinates_construction_and_output_ordering():
    assert Vector.ByCoordinates(1, 2, 3) == [1, 2, 3]
    assert Vector.Coordinates([1.23456, 2.34567, 3.45678], mantissa=2) == [1.23, 2.35, 3.46]
    assert Vector.Coordinates([1, 2, 3], outputType="zyx") == [3.0, 2.0, 1.0]
    assert Vector.Coordinates([1, 2, 3], outputType="matrix") == [
        [1, 0, 0, 1.0],
        [0, 1, 0, 2.0],
        [0, 0, 1, 3.0],
        [0, 0, 0, 1],
    ]
    assert Vector.Coordinates([1, 2]) == [1.0, 2.0, 0.0]
    assert Vector.Coordinates("not a vector", silent=True) is None


def test_magnitude_quadrance_normalize_and_set_magnitude():
    assert Vector.Quadrance([3, 4, 12]) == pytest.approx(169.0)
    assert Vector.Magnitude([3, 4, 12]) == pytest.approx(13.0)
    assert Vector.Length([3, 4, 12]) == pytest.approx(13.0)
    assert Vector.Normalize([3, 4, 0]) == pytest.approx([0.6, 0.8, 0.0])
    assert Vector.SetMagnitude([3, 4, 0], 10) == pytest.approx([6.0, 8.0, 0.0])
    assert Vector.Normalize([0, 0, 0]) is None
    assert Vector.SetMagnitude([0, 0, 0], 5) is None
    assert Vector.Magnitude("not a vector") is None


def test_dot_cross_spread_and_angle():
    assert Vector.Dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)
    assert Vector.Cross([1, 0, 0], [0, 1, 0]) == [0.0, 0.0, 1.0]
    assert Vector.Cross([1, 0, 0], [2, 0, 0]) == [0, 0, 0]
    assert Vector.Cross([0, 0, 0], [1, 0, 0], silent=True) is None
    assert Vector.Dot([0, 0, 0], [1, 0, 0], silent=True) is None
    assert Vector.Spread([1, 0], [0, 1]) == pytest.approx(1.0)
    assert Vector.Spread([1, 0, 0], [2, 0, 0]) == pytest.approx(0.0)
    assert Vector.Angle([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert Vector.Angle([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert Vector.Angle([1, 0, 0], [-1, 0, 0], bracket=True) == pytest.approx(0.0)


def test_azimuth_altitude_round_trip_and_reverse():
    assert Vector.ByAzimuthAltitude(0, 0) == pytest.approx([0.0, 1.0, 0.0])
    assert Vector.ByAzimuthAltitude(90, 0) == pytest.approx([1.0, 0.0, 0.0])
    assert Vector.ByAzimuthAltitude(0, 90) == pytest.approx([0.0, 0.0, 1.0])
    assert Vector.ByAzimuthAltitude(0, 0, reverse=True) == pytest.approx([0.0, -1.0, 0.0])
    assert Vector.AzimuthAltitude([0, 1, 0]) == {"azimuth": 0.0, "altitude": 0.0}
    assert Vector.AzimuthAltitude([1, 0, 0]) == {"azimuth": 90.0, "altitude": 0.0}
    assert Vector.AzimuthAltitude([0, 0, 1]) == {"azimuth": 0, "altitude": 90}
    assert Vector.AzimuthAltitude([0, 0, 0]) is None


def test_compass_angle_and_compass_direction():
    assert Vector.CompassAngle([0, 1, 0], [1, 0, 0]) == pytest.approx(90.0)
    assert Vector.CompassAngle([1, 0, 0], [0, 1, 0]) == pytest.approx(270.0)
    assert Vector.CompassAngle([0, 0, 1], [1, 0, 0], silent=True) is None
    assert Vector.CompassDirection([0, 0, 0]) == "Origin"
    assert Vector.CompassDirection([0, 1, 0]) == "North"
    assert Vector.CompassDirection([1, 1, 1]) == "Up_Northeast"
    assert Vector.CompassDirection([-1, -1, -1]) == "Down_Southwest"
    assert "Down_Northwest" in Vector.CompassDirections()


def test_direction_convenience_vectors():
    assert Vector.North() == [0, 1, 0]
    assert Vector.South() == [0, -1, 0]
    assert Vector.East() == [1, 0, 0]
    assert Vector.West() == [-1, 0, 0]
    assert Vector.Up() == [0, 0, 1]
    assert Vector.Down() == [0, 0, -1]
    assert Vector.NorthEast() == [1, 1, 0]
    assert Vector.NorthWest() == [-1, 1, 0]
    assert Vector.SouthEast() == [1, -1, 0]
    assert Vector.SouthWest() == [-1, -1, 0]
    assert Vector.XAxis() == [1, 0, 0]
    assert Vector.YAxis() == [0, 1, 0]
    assert Vector.ZAxis() == [0, 0, 1]


def test_bisect_parallel_antiparallel_and_invalid_inputs():
    assert Vector.Bisect([1, 0, 0], [0, 1, 0]) == pytest.approx([math.sqrt(0.5), math.sqrt(0.5), 0.0])
    assert Vector.Bisect([2, 0, 0], [4, 0, 0]) == pytest.approx([1.0, 0.0, 0.0])
    b = Vector.Bisect([1, 0, 0], [-1, 0, 0])
    assert Vector.Magnitude(b) == pytest.approx(1.0)
    assert Vector.Dot(b, [1, 0, 0]) == pytest.approx(0.0)
    assert Vector.Bisect([0, 0, 0], [1, 0, 0]) is None


def test_parallel_antiparallel_collinear_and_same():
    assert Vector.IsParallel([1, 0, 0], [10, 0, 0]) is True
    assert Vector.IsParallel([1, 0, 0], [-10, 0, 0]) is False
    assert Vector.IsAntiParallel([1, 0, 0], [-10, 0, 0]) is True
    assert Vector.IsAntiParallel([1, 0, 0], [10, 0, 0]) is False
    assert Vector.IsCollinear([1, 0, 0], [-10, 0, 0]) is True
    assert Vector.IsCollinear([1, 0, 0], [0, 1, 0]) is False
    assert Vector.IsParallel([0, 0, 0], [1, 0, 0]) is False
    assert Vector.IsAntiParallel([0, 0, 0], [1, 0, 0]) is False
    assert Vector.IsCollinear([0, 0, 0], [1, 0, 0]) is False
    assert Vector.IsSame([1, 2, 3], [1.00001, 2.00001, 2.99999], tolerance=0.0001) is True
    assert Vector.IsSame([1, 2, 3], [1, 2], tolerance=0.0001) is False


def test_transformation_matrix_aligns_vectors():
    matrix = Vector.TransformationMatrix([1, 0, 0], [0, 1, 0])
    assert matrix is not None and len(matrix) == 4 and all(len(row) == 4 for row in matrix)
    mapped = _apply_matrix(matrix, [1, 0, 0])
    assert mapped == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    anti = Vector.TransformationMatrix([1, 0, 0], [-1, 0, 0])
    mapped_anti = _apply_matrix(anti, [1, 0, 0])
    assert mapped_anti == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)

    assert Vector.TransformationMatrix([0, 0, 0], [1, 0, 0]) is None


def test_by_vertices_with_fake_topologicpy_modules(monkeypatch):
    vertex_module = types.ModuleType("topologicpy.Vertex")
    topology_module = types.ModuleType("topologicpy.Topology")
    helper_module = types.ModuleType("topologicpy.Helper")

    class FakeVertex:
        @staticmethod
        def X(vertex, mantissa=6):
            return round(float(vertex[0]), mantissa)

        @staticmethod
        def Y(vertex, mantissa=6):
            return round(float(vertex[1]), mantissa)

        @staticmethod
        def Z(vertex, mantissa=6):
            return round(float(vertex[2]), mantissa)

    class FakeTopology:
        @staticmethod
        def IsInstance(value, type_name):
            return type_name == "Vertex" and isinstance(value, tuple) and len(value) == 3

    class FakeHelper:
        @staticmethod
        def Flatten(values):
            out = []
            for value in values:
                if isinstance(value, list):
                    out.extend(FakeHelper.Flatten(value))
                else:
                    out.append(value)
            return out

    vertex_module.Vertex = FakeVertex
    topology_module.Topology = FakeTopology
    helper_module.Helper = FakeHelper
    monkeypatch.setitem(sys.modules, "topologicpy.Vertex", vertex_module)
    monkeypatch.setitem(sys.modules, "topologicpy.Topology", topology_module)
    monkeypatch.setitem(sys.modules, "topologicpy.Helper", helper_module)

    assert Vector.ByVertices((0, 0, 0), (3, 4, 0), normalize=False) == [3.0, 4.0, 0.0]
    assert Vector.ByVertices([(0, 0, 0), (3, 4, 0)], normalize=True) == pytest.approx([0.6, 0.8, 0.0])
    assert Vector.ByVertices((0, 0, 0), silent=True) is None
