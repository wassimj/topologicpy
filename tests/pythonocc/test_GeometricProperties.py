# Auto-generated PythonOCC backend parity test
# Copied from test_GeometricProperties.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# Wassim Jabi
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

"""Regression tests for TopologicPy geometric properties.

The suite checks:
- geometric centroid and centre of mass;
- vertex-average centroid as a distinct quantity;
- edge U and face UV evaluation and round-trips;
- 2D and 3D normal direction;
- length, perimeter, area, surface area, and volume;
- non-convex geometry and geometry containing holes.

All expected values are calculated analytically, independently of the backend.
"""

from functools import lru_cache
import math

import pytest

from topologicpy.Cell import Cell
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


TOPOLOGY_TOLERANCE = 0.0001
ABSOLUTE_TOLERANCE = 0.0001
PARAMETER_TOLERANCE = 0.000001
NORMAL_TOLERANCE = 0.00001
MANTISSA = 12


def _vertex(x, y, z):
    vertex = Vertex.ByCoordinates(float(x), float(y), float(z))
    assert Topology.IsInstance(vertex, "Vertex")
    return vertex


def _xyz(vertex):
    values = Vertex.Coordinates(vertex, outputType="xyz", mantissa=MANTISSA)
    assert isinstance(values, list) and len(values) == 3
    return tuple(float(value) for value in values)


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _mul(vector, scalar):
    return tuple(value * scalar for value in vector)


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _magnitude(vector):
    return math.sqrt(_dot(vector, vector))


def _normalize(vector):
    magnitude = _magnitude(vector)
    assert magnitude > 0.0
    return tuple(value / magnitude for value in vector)


def _distance(a, b):
    return _magnitude(_sub(a, b))


def _assert_scalar(actual, expected, tolerance=ABSOLUTE_TOLERANCE, label="value"):
    assert isinstance(actual, (int, float)), f"{label} is not numeric: {actual!r}"
    assert math.isclose(
        float(actual), float(expected), abs_tol=tolerance, rel_tol=1.0e-9
    ), f"{label}: expected {expected:.12g}, observed {float(actual):.12g}"


def _assert_xyz(actual, expected, tolerance=ABSOLUTE_TOLERANCE, label="XYZ"):
    assert len(actual) == 3 and len(expected) == 3
    for index, axis in enumerate("XYZ"):
        _assert_scalar(actual[index], expected[index], tolerance, f"{label} {axis}")


def _normal(face):
    values = Face.Normal(face, outputType="xyz", mantissa=MANTISSA)
    assert isinstance(values, list) and len(values) == 3
    return _normalize(tuple(float(value) for value in values))


def _centroid_xyz(topology):
    centroid = Topology.Centroid(topology, silent=True)
    assert Topology.IsInstance(centroid, "Vertex")
    return _xyz(centroid)


def _center_of_mass_xyz(topology):
    center = Topology.CenterOfMass(topology, silent=True)
    assert Topology.IsInstance(center, "Vertex")
    return _xyz(center)


def _vertices_centroid_xyz(topology):
    centroid = Topology.VerticesCentroid(topology, mantissa=MANTISSA, silent=True)
    assert Topology.IsInstance(centroid, "Vertex")
    return _xyz(centroid)


def _assert_centres(topology, expected, label):
    centroid = _centroid_xyz(topology)
    center_of_mass = _center_of_mass_xyz(topology)
    _assert_xyz(centroid, expected, label=f"{label} centroid")
    _assert_xyz(center_of_mass, expected, label=f"{label} centre of mass")
    _assert_xyz(centroid, center_of_mass, label=f"{label} centroid agreement")


def _face(points):
    face = Face.ByVertices(
        [_vertex(*point) for point in points],
        tolerance=TOPOLOGY_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


def _regular_polygon_area(radius, sides):
    return 0.5 * sides * radius * radius * math.sin(2.0 * math.pi / sides)


@lru_cache(maxsize=1)
def _geometry():
    edge = Edge.ByVertices(
        [_vertex(1, -2, 3), _vertex(7, 6, 3)],
        tolerance=TOPOLOGY_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(edge, "Edge")

    l_wire = Wire.ByVertices(
        [_vertex(0, 0, 0), _vertex(4, 0, 0), _vertex(4, 3, 0)],
        close=False,
        tolerance=TOPOLOGY_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(l_wire, "Wire")

    l_face = _face(
        [(0, 0, 0), (4, 0, 0), (4, 1, 0),
         (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    )

    outer_wire = Wire.Rectangle(
        origin=_vertex(0, 0, 0), width=8, length=6,
        direction=[0, 0, 1], placement="lowerleft",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    hole_wire = Wire.Rectangle(
        origin=_vertex(5, 1, 0), width=2, length=2,
        direction=[0, 0, 1], placement="lowerleft",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    assert Topology.IsInstance(outer_wire, "Wire")
    assert Topology.IsInstance(hole_wire, "Wire")
    holed_face = Face.ByWires(
        outer_wire, [hole_wire], tolerance=TOPOLOGY_TOLERANCE, silent=True
    )
    assert Topology.IsInstance(holed_face, "Face")

    a, b, c, d = (0, 0, 0), (4, 0, 0), (0, 3, 0), (0, 0, 6)
    tetrahedron = Cell.ByFaces(
        [
            _face([a, c, b]),
            _face([a, b, d]),
            _face([a, d, c]),
            _face([b, c, d]),
        ],
        tolerance=TOPOLOGY_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(tetrahedron, "Cell")

    prism = Cell.Prism(
        origin=_vertex(2, -1, 4), width=4, length=6, height=8,
        uSides=1, vSides=1, wSides=1,
        direction=[0, 0, 1], placement="center",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    assert Topology.IsInstance(prism, "Cell")

    chs_origin = (3.0, -4.0, 2.0)
    chs_radius, chs_thickness, chs_height, chs_sides = 3.0, 0.5, 5.0, 12
    chs = Cell.CHS(
        origin=_vertex(*chs_origin), radius=chs_radius,
        thickness=chs_thickness, height=chs_height, sides=chs_sides,
        direction=[0, 0, 1], placement="center",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    assert Topology.IsInstance(chs, "Cell")

    return {
        "edge": edge,
        "l_wire": l_wire,
        "l_face": l_face,
        "outer_wire": outer_wire,
        "hole_wire": hole_wire,
        "holed_face": holed_face,
        "tetrahedron": tetrahedron,
        "prism": prism,
        "chs": chs,
        "chs_origin": chs_origin,
        "chs_radius": chs_radius,
        "chs_thickness": chs_thickness,
        "chs_height": chs_height,
        "chs_sides": chs_sides,
    }


# Centroid and centre of mass -------------------------------------------------

def test_edge_centroid_and_center_of_mass():
    _assert_centres(_geometry()["edge"], (4, 2, 3), "edge")


def test_length_weighted_wire_centroid_and_center_of_mass():
    expected = (20.0 / 7.0, 9.0 / 14.0, 0.0)
    _assert_centres(_geometry()["l_wire"], expected, "open L-wire")


def test_non_convex_face_centroid_and_center_of_mass():
    _assert_centres(_geometry()["l_face"], (1.5, 1.0, 0.0), "L-face")


def test_holed_face_centroid_and_center_of_mass():
    _assert_centres(
        _geometry()["holed_face"], (42.0 / 11.0, 34.0 / 11.0, 0.0),
        "holed face",
    )


def test_tetrahedron_centroid_and_center_of_mass():
    _assert_centres(_geometry()["tetrahedron"], (1.0, 0.75, 1.5), "tetrahedron")


def test_hollow_cell_centroid_and_center_of_mass():
    geometry = _geometry()
    _assert_centres(geometry["chs"], geometry["chs_origin"], "CHS")


def test_vertices_centroid_differs_from_wire_center_of_mass():
    wire = _geometry()["l_wire"]
    geometric = _centroid_xyz(wire)
    vertex_average = _vertices_centroid_xyz(wire)
    _assert_xyz(geometric, (20 / 7, 9 / 14, 0), label="wire geometric centroid")
    _assert_xyz(vertex_average, (8 / 3, 1, 0), label="wire vertex centroid")
    assert _distance(geometric, vertex_average) > 0.1


def test_vertices_centroid_differs_from_face_center_of_mass():
    face = _geometry()["l_face"]
    geometric = _centroid_xyz(face)
    vertex_average = _vertices_centroid_xyz(face)
    _assert_xyz(geometric, (1.5, 1, 0), label="face geometric centroid")
    _assert_xyz(vertex_average, (5 / 3, 4 / 3, 0), label="face vertex centroid")
    assert _distance(geometric, vertex_average) > 0.1


# Edge U parameters ----------------------------------------------------------

EDGE_U_CASES = (
    (0.0, (1.0, -2.0, 3.0)),
    (0.2, (2.2, -0.4, 3.0)),
    (0.5, (4.0, 2.0, 3.0)),
    (0.85, (6.1, 4.8, 3.0)),
    (1.0, (7.0, 6.0, 3.0)),
)


@pytest.mark.parametrize("u, expected_xyz", EDGE_U_CASES)
def test_edge_vertex_by_parameter_xyz(u, expected_xyz):
    vertex = Edge.VertexByParameter(_geometry()["edge"], u=u)
    assert Topology.IsInstance(vertex, "Vertex")
    _assert_xyz(_xyz(vertex), expected_xyz, PARAMETER_TOLERANCE, f"edge U={u}")


@pytest.mark.parametrize("u, expected_xyz", EDGE_U_CASES)
def test_edge_parameter_at_vertex(u, expected_xyz):
    actual = Edge.ParameterAtVertex(
        _geometry()["edge"], _vertex(*expected_xyz), mantissa=MANTISSA, silent=True
    )
    _assert_scalar(actual, u, PARAMETER_TOLERANCE, f"edge parameter U={u}")


@pytest.mark.parametrize("u, expected_xyz", EDGE_U_CASES)
def test_edge_parameter_round_trip(u, expected_xyz):
    edge = _geometry()["edge"]
    vertex = Edge.VertexByParameter(edge, u=u)
    recovered = Edge.ParameterAtVertex(edge, vertex, mantissa=MANTISSA, silent=True)
    _assert_xyz(_xyz(vertex), expected_xyz, PARAMETER_TOLERANCE, "edge round-trip XYZ")
    _assert_scalar(recovered, u, PARAMETER_TOLERANCE, "edge round-trip U")


@pytest.mark.parametrize("u, expected_xyz", EDGE_U_CASES)
def test_reversed_edge_parameter_is_one_minus_u(u, expected_xyz):
    reversed_edge = Edge.Reverse(_geometry()["edge"], silent=True)
    assert Topology.IsInstance(reversed_edge, "Edge")
    actual = Edge.ParameterAtVertex(
        reversed_edge, _vertex(*expected_xyz), mantissa=MANTISSA, silent=True
    )
    _assert_scalar(actual, 1.0 - u, PARAMETER_TOLERANCE, "reversed-edge U")


# Face UV parameters ---------------------------------------------------------

UV_CASES = (
    (0.0, 0.0, (10.0, 20.0, 30.0)),
    (1.0, 0.0, (18.0, 20.0, 30.0)),
    (0.0, 1.0, (10.0, 26.0, 30.0)),
    (1.0, 1.0, (18.0, 26.0, 30.0)),
    (0.25, 0.75, (12.0, 24.5, 30.0)),
    (0.6, 0.35, (14.8, 22.1, 30.0)),
)


@lru_cache(maxsize=1)
def _uv_face():
    face = Face.Rectangle(
        origin=_vertex(10, 20, 30), width=8, length=6,
        direction=[0, 0, 1], placement="lowerleft",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    return face


@pytest.mark.parametrize("u, v, expected_xyz", UV_CASES)
def test_face_vertex_by_parameters_xyz(u, v, expected_xyz):
    vertex = Face.VertexByParameters(_uv_face(), u=u, v=v)
    assert Topology.IsInstance(vertex, "Vertex")
    _assert_xyz(_xyz(vertex), expected_xyz, PARAMETER_TOLERANCE, f"face UV=({u},{v})")


@pytest.mark.parametrize("u, v, expected_xyz", UV_CASES)
def test_face_vertex_parameters(u, v, expected_xyz):
    parameters = Face.VertexParameters(
        _uv_face(), _vertex(*expected_xyz), outputType="uv", mantissa=MANTISSA
    )
    assert isinstance(parameters, list) and len(parameters) == 2
    _assert_scalar(parameters[0], u, PARAMETER_TOLERANCE, "face U")
    _assert_scalar(parameters[1], v, PARAMETER_TOLERANCE, "face V")


@pytest.mark.parametrize("u, v, expected_xyz", UV_CASES)
def test_face_parameter_round_trip(u, v, expected_xyz):
    face = _uv_face()
    vertex = Face.VertexByParameters(face, u=u, v=v)
    parameters = Face.VertexParameters(face, vertex, outputType="uv", mantissa=MANTISSA)
    _assert_xyz(_xyz(vertex), expected_xyz, PARAMETER_TOLERANCE, "face UV XYZ")
    _assert_scalar(parameters[0], u, PARAMETER_TOLERANCE, "round-trip U")
    _assert_scalar(parameters[1], v, PARAMETER_TOLERANCE, "round-trip V")


# Normals --------------------------------------------------------------------

NORMAL_CASES = (
    ([0, 0, 1], (0, 0, 1)),
    ([0, 0, -1], (0, 0, -1)),
    ([1, 0, 0], (1, 0, 0)),
    ([0, -1, 0], (0, -1, 0)),
    ([1, 2, 3], _normalize((1, 2, 3))),
)


@pytest.mark.parametrize("direction, expected", NORMAL_CASES)
def test_2d_primitive_normal(direction, expected):
    face = Face.Rectangle(
        origin=_vertex(2, -3, 5), width=4, length=7,
        direction=direction, placement="center",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    actual = _normal(face)
    _assert_xyz(actual, expected, NORMAL_TOLERANCE, "face normal")
    _assert_scalar(_magnitude(actual), 1.0, NORMAL_TOLERANCE, "normal magnitude")


def test_face_invert_reverses_normal():
    face = Face.Rectangle(
        origin=_vertex(1, 2, 3), width=5, length=2,
        direction=[1, 2, 3], placement="center",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    inverted = Face.Invert(face, tolerance=TOPOLOGY_TOLERANCE, silent=True)
    assert Topology.IsInstance(inverted, "Face")
    _assert_xyz(_normal(inverted), _mul(_normal(face), -1), NORMAL_TOLERANCE, "inverted normal")


def test_prism_face_normals_point_outward():
    cell = _geometry()["prism"]
    cell_center = _centroid_xyz(cell)
    faces = Topology.Faces(cell, silent=True)
    assert isinstance(faces, list) and len(faces) == 6
    for index, face in enumerate(faces):
        expected = _normalize(_sub(_centroid_xyz(face), cell_center))
        actual = _normal(face)
        _assert_xyz(actual, expected, NORMAL_TOLERANCE, f"prism face {index} normal")


def test_tetrahedron_face_normals_point_outward():
    cell = _geometry()["tetrahedron"]
    cell_center = _centroid_xyz(cell)
    faces = Topology.Faces(cell, silent=True)
    assert isinstance(faces, list) and len(faces) == 4
    for index, face in enumerate(faces):
        outward_hint = _normalize(_sub(_centroid_xyz(face), cell_center))
        assert _dot(_normal(face), outward_hint) > 0.0, f"face {index} points inward"


def test_hollow_cell_outer_and_inner_side_normals():
    geometry = _geometry()
    cell = geometry["chs"]
    origin = geometry["chs_origin"]
    outer_radius = geometry["chs_radius"]
    inner_radius = outer_radius - geometry["chs_thickness"]
    threshold = 0.5 * (outer_radius + inner_radius)
    outer_count = inner_count = 0

    for face in Topology.Faces(cell, silent=True):
        normal = _normal(face)
        if abs(normal[2]) > 0.5:
            continue
        center = _centroid_xyz(face)
        radial = (center[0] - origin[0], center[1] - origin[1], 0.0)
        radial_length = _magnitude(radial)
        radial_unit = _normalize(radial)
        if radial_length > threshold:
            outer_count += 1
            assert _dot(normal, radial_unit) > 0.999
        else:
            inner_count += 1
            assert _dot(normal, radial_unit) < -0.999

    assert outer_count == geometry["chs_sides"]
    assert inner_count == geometry["chs_sides"]


# Length, perimeter, and area -------------------------------------------------

def test_edge_length():
    _assert_scalar(Edge.Length(_geometry()["edge"], mantissa=MANTISSA), 10.0, label="edge length")


def test_open_wire_length():
    _assert_scalar(Wire.Length(_geometry()["l_wire"], mantissa=MANTISSA), 7.0, label="wire length")


def test_non_convex_face_perimeter_and_area():
    face = _geometry()["l_face"]
    boundary = Face.ExternalBoundary(face)
    _assert_scalar(Wire.Length(boundary, mantissa=MANTISSA), 14.0, label="L-face perimeter")
    _assert_scalar(Face.Area(face, mantissa=MANTISSA), 6.0, label="L-face area")


def test_holed_face_perimeters_area_and_boundary_count():
    face = _geometry()["holed_face"]
    external = Face.ExternalBoundary(face)
    internal = Face.InternalBoundaries(face)
    assert isinstance(internal, list) and len(internal) == 1
    assert Topology.IsInstance(internal[0], "Wire")
    _assert_scalar(Wire.Length(external, mantissa=MANTISSA), 28.0, label="outer perimeter")
    _assert_scalar(Wire.Length(internal[0], mantissa=MANTISSA), 8.0, label="hole perimeter")
    _assert_scalar(Face.Area(face, mantissa=MANTISSA), 44.0, label="holed area")


def test_holed_face_area_is_outer_minus_hole():
    geometry = _geometry()
    outer = Face.ByWire(geometry["outer_wire"], tolerance=TOPOLOGY_TOLERANCE, silent=True)
    hole = Face.ByWire(geometry["hole_wire"], tolerance=TOPOLOGY_TOLERANCE, silent=True)
    expected = Face.Area(outer, mantissa=MANTISSA) - Face.Area(hole, mantissa=MANTISSA)
    _assert_scalar(Face.Area(geometry["holed_face"], mantissa=MANTISSA), expected, label="outer-minus-hole area")


@pytest.mark.parametrize("radius, sides", ((2.0, 7), (3.5, 12), (1.25, 32)))
def test_faceted_circle_area(radius, sides):
    face = Face.Circle(
        origin=_vertex(0, 0, 0), radius=radius, sides=sides,
        direction=[0, 0, 1], placement="center",
        tolerance=TOPOLOGY_TOLERANCE,
    )
    assert Topology.IsInstance(face, "Face")
    _assert_scalar(
        Face.Area(face, mantissa=MANTISSA),
        _regular_polygon_area(radius, sides),
        label=f"{sides}-gon area",
    )


def test_faceted_hollow_face_area_and_boundary_count():
    radius, thickness, sides = 3.0, 0.5, 12
    face = Face.CHS(
        origin=_vertex(0, 0, 0), radius=radius, thickness=thickness,
        sides=sides, direction=[0, 0, 1], placement="center",
        tolerance=TOPOLOGY_TOLERANCE, silent=True,
    )
    assert Topology.IsInstance(face, "Face")
    expected = _regular_polygon_area(radius, sides) - _regular_polygon_area(radius - thickness, sides)
    _assert_scalar(Face.Area(face, mantissa=MANTISSA), expected, label="CHS face area")
    internal = Face.InternalBoundaries(face)
    assert isinstance(internal, list) and len(internal) == 1


# Volume and surface area -----------------------------------------------------

def test_tetrahedron_volume_and_surface_area():
    cell = _geometry()["tetrahedron"]
    _assert_scalar(Cell.Volume(cell, mantissa=MANTISSA), 12.0, label="tetrahedron volume")
    _assert_scalar(Cell.Area(cell, mantissa=MANTISSA), 27.0 + math.sqrt(261.0), label="tetrahedron area")


@pytest.mark.parametrize(
    "radius, height, sides",
    ((2.0, 5.0, 7), (3.5, 2.25, 12), (1.25, 6.0, 32)),
)
def test_faceted_cylinder_volume(radius, height, sides):
    cell = Cell.Cylinder(
        origin=_vertex(0, 0, 0), radius=radius, height=height,
        uSides=sides, vSides=1, direction=[0, 0, 1],
        placement="center", mantissa=MANTISSA,
        tolerance=TOPOLOGY_TOLERANCE,
    )
    assert Topology.IsInstance(cell, "Cell")
    expected = _regular_polygon_area(radius, sides) * height
    _assert_scalar(Cell.Volume(cell, mantissa=MANTISSA), expected, label="faceted cylinder volume")


def test_faceted_hollow_section_volume():
    geometry = _geometry()
    radius = geometry["chs_radius"]
    inner_radius = radius - geometry["chs_thickness"]
    expected = (
        _regular_polygon_area(radius, geometry["chs_sides"])
        - _regular_polygon_area(inner_radius, geometry["chs_sides"])
    ) * geometry["chs_height"]
    measured = Cell.Volume(geometry["chs"], mantissa=MANTISSA)
    _assert_scalar(measured, expected, label="CHS volume")
    assert measured < _regular_polygon_area(radius, geometry["chs_sides"]) * geometry["chs_height"]


def test_prism_volume_and_surface_area():
    cell = _geometry()["prism"]
    _assert_scalar(Cell.Volume(cell, mantissa=MANTISSA), 4 * 6 * 8, label="prism volume")
    _assert_scalar(Cell.Area(cell, mantissa=MANTISSA), 2 * (4 * 6 + 4 * 8 + 6 * 8), label="prism area")
