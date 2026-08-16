# Copyright (C) 2026
# Wassim Jabi
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.

"""
Regression tests for TopologicPy transformations.

The test matrix covers these topology types:

    Vertex, Edge, Wire, Face, Shell, Cell, CellComplex

and these transformation operations:

    Translate, Scale, Rotate, Transform

The transformation ground truth is calculated independently with affine
matrix mathematics. The tests verify, where applicable:

    * topology type and subtopology counts;
    * transformed vertex-centroid location;
    * every edge length;
    * every face area;
    * every cell volume;
    * transformed face-normal vectors for Rotate and Transform.

The source topologies are also checked against explicit geometric ground truth
before they are used as transformation operands.

Legacy-backend comparison
-------------------------
Run the normal suite against the trusted legacy backend:

    pytest -q test_Transformations.py

To print a complete expected-versus-observed report, use:

PowerShell:

    $env:TOPOLOGICPY_TRANSFORMATION_REPORT = "1"
    pytest -q -s test_Transformations.py::test_legacy_backend_report
    Remove-Item Env:TOPOLOGICPY_TRANSFORMATION_REPORT

Command Prompt:

    set TOPOLOGICPY_TRANSFORMATION_REPORT=1
    pytest -q -s test_Transformations.py::test_legacy_backend_report
    set TOPOLOGICPY_TRANSFORMATION_REPORT=

The same file can then be run unchanged against the PythonOCC backend.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from typing import Optional, Sequence, Tuple

import pytest

from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


TOPOLOGY_TOLERANCE = 0.0001
ABSOLUTE_TOLERANCE = 0.0001
RELATIVE_TOLERANCE = 1.0e-7
NORMAL_TOLERANCE = 1.0e-5
MEASUREMENT_MANTISSA = 12
REPORT_ENVIRONMENT_VARIABLE = "TOPOLOGICPY_TRANSFORMATION_REPORT"

TOPOLOGY_TYPES = (
    "vertex",
    "edge",
    "wire",
    "face",
    "shell",
    "cell",
    "cellcomplex",
)

OPERATION_NAMES = (
    "translate",
    "scale",
    "rotate",
    "transform",
)

NORMAL_OPERATION_NAMES = (
    "rotate",
    "transform",
)

BASE_ORIGIN = (1.25, -0.75, 2.5)
PRISM_WIDTH = 2.0
PRISM_LENGTH = 3.0
PRISM_HEIGHT = 4.0
CELL_COMPLEX_RADIUS = 1.0

EDGE_START = (0.25, -1.5, 1.0)
EDGE_END = (2.25, 0.0, 4.0)

TRANSLATION = (3.25, -2.5, 4.75)

SCALE_ORIGIN = (0.5, -1.25, 2.0)
SCALE_FACTORS = (1.5, 0.75, 2.25)

ROTATION_ORIGIN = (-0.5, 1.25, 0.75)
ROTATION_AXIS = (1.0, 2.0, 3.0)
ROTATION_ANGLE = 37.0

TRANSFORM_AXIS = (0.5, 1.0, 2.0)
TRANSFORM_ANGLE = 29.0
TRANSFORM_SCALE_FACTORS = (1.2, 0.8, 1.5)
TRANSFORM_TRANSLATION = (2.0, -1.5, 3.25)

# Explicit source-topology ground truth:
#     (number_of_cells, number_of_faces, number_of_edges, number_of_vertices)
BASE_COUNTS = {
    "vertex": (0, 0, 0, 1),
    "edge": (0, 0, 1, 2),
    "wire": (0, 0, 4, 4),
    "face": (0, 1, 4, 4),
    "shell": (0, 6, 12, 8),
    "cell": (1, 6, 12, 8),
    "cellcomplex": (2, 9, 12, 6),
}

BASE_EDGE_LENGTHS = {
    "vertex": (),
    "edge": (math.sqrt(15.25),),
    "wire": (2.0, 2.0, 3.0, 3.0),
    "face": (2.0, 2.0, 3.0, 3.0),
    "shell": (2.0,) * 4 + (3.0,) * 4 + (4.0,) * 4,
    "cell": (2.0,) * 4 + (3.0,) * 4 + (4.0,) * 4,
    "cellcomplex": (math.sqrt(2.0),) * 12,
}

BASE_FACE_AREAS = {
    "vertex": (),
    "edge": (),
    "wire": (),
    "face": (6.0,),
    "shell": (6.0, 6.0, 8.0, 8.0, 12.0, 12.0),
    "cell": (6.0, 6.0, 8.0, 8.0, 12.0, 12.0),
    "cellcomplex": (math.sqrt(3.0) * 0.5,) * 8 + (2.0,),
}

BASE_CELL_VOLUMES = {
    "vertex": (),
    "edge": (),
    "wire": (),
    "face": (),
    "shell": (),
    "cell": (24.0,),
    "cellcomplex": (2.0 / 3.0, 2.0 / 3.0),
}


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class AffineSpecification:
    """A column-vector affine transformation: p' = linear * p + translation."""

    name: str
    linear: Matrix3
    translation: Vector3


@dataclass(frozen=True)
class GeometryMetrics:
    """The geometric measurements compared by each transformation test."""

    result_type: str
    counts: Tuple[int, int, int, int]
    location: Vector3
    edge_lengths: Tuple[float, ...]
    face_areas: Tuple[float, ...]
    cell_volumes: Tuple[float, ...]
    face_normals: Tuple[Vector3, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    """The expected and observed values for one transformation matrix entry."""

    topology_type: str
    operation_name: str
    expected: Optional[GeometryMetrics]
    observed: Optional[GeometryMetrics]
    mismatches: Tuple[str, ...]
    error: Optional[str]


def _vertex(coordinates: Sequence[float]):
    """Create a vertex from a three-component coordinate sequence."""
    vertex = Vertex.ByCoordinates(
        float(coordinates[0]),
        float(coordinates[1]),
        float(coordinates[2]),
    )
    assert Topology.IsInstance(vertex, "Vertex")
    return vertex


def _coordinates(vertex) -> Vector3:
    """Return unrounded vertex coordinates as a float tuple."""
    coordinates = Vertex.Coordinates(
        vertex,
        outputType="xyz",
        mantissa=MEASUREMENT_MANTISSA,
    )
    assert isinstance(coordinates, list)
    assert len(coordinates) == 3
    return (
        float(coordinates[0]),
        float(coordinates[1]),
        float(coordinates[2]),
    )


def _identity_matrix() -> Matrix3:
    """Return the 3 x 3 identity matrix."""
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _diagonal_matrix(values: Sequence[float]) -> Matrix3:
    """Return a diagonal 3 x 3 matrix."""
    return (
        (float(values[0]), 0.0, 0.0),
        (0.0, float(values[1]), 0.0),
        (0.0, 0.0, float(values[2])),
    )


def _matrix_vector_multiply(matrix: Matrix3, vector: Vector3) -> Vector3:
    """Multiply a 3 x 3 matrix by a three-component column vector."""
    return (
        matrix[0][0] * vector[0]
        + matrix[0][1] * vector[1]
        + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0]
        + matrix[1][1] * vector[1]
        + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0]
        + matrix[2][1] * vector[1]
        + matrix[2][2] * vector[2],
    )


def _matrix_multiply(matrix_a: Matrix3, matrix_b: Matrix3) -> Matrix3:
    """Multiply two 3 x 3 matrices."""
    rows = []
    for row in range(3):
        values = []
        for column in range(3):
            values.append(
                sum(
                    matrix_a[row][index] * matrix_b[index][column]
                    for index in range(3)
                )
            )
        rows.append(tuple(values))
    return tuple(rows)  # type: ignore[return-value]


def _transpose(matrix: Matrix3) -> Matrix3:
    """Return the transpose of a 3 x 3 matrix."""
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _determinant(matrix: Matrix3) -> float:
    """Return the determinant of a 3 x 3 matrix."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def _inverse(matrix: Matrix3) -> Matrix3:
    """Return the inverse of a non-singular 3 x 3 matrix."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    determinant = _determinant(matrix)
    if abs(determinant) <= 1.0e-12:
        raise ValueError("The transformation matrix is singular.")

    inverse_determinant = 1.0 / determinant
    return (
        (
            (e * i - f * h) * inverse_determinant,
            (c * h - b * i) * inverse_determinant,
            (b * f - c * e) * inverse_determinant,
        ),
        (
            (f * g - d * i) * inverse_determinant,
            (a * i - c * g) * inverse_determinant,
            (c * d - a * f) * inverse_determinant,
        ),
        (
            (d * h - e * g) * inverse_determinant,
            (b * g - a * h) * inverse_determinant,
            (a * e - b * d) * inverse_determinant,
        ),
    )


def _vector_add(vector_a: Vector3, vector_b: Vector3) -> Vector3:
    """Add two three-component vectors."""
    return (
        vector_a[0] + vector_b[0],
        vector_a[1] + vector_b[1],
        vector_a[2] + vector_b[2],
    )


def _vector_subtract(vector_a: Vector3, vector_b: Vector3) -> Vector3:
    """Subtract vector_b from vector_a."""
    return (
        vector_a[0] - vector_b[0],
        vector_a[1] - vector_b[1],
        vector_a[2] - vector_b[2],
    )


def _vector_multiply(vector: Vector3, scalar: float) -> Vector3:
    """Multiply a vector by a scalar."""
    return (
        vector[0] * scalar,
        vector[1] * scalar,
        vector[2] * scalar,
    )


def _vector_magnitude(vector: Vector3) -> float:
    """Return a vector's Euclidean magnitude."""
    return math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )


def _normalize(vector: Vector3) -> Vector3:
    """Return a unit vector."""
    magnitude = _vector_magnitude(vector)
    if magnitude <= 1.0e-12:
        raise ValueError("A zero-length vector cannot be normalized.")
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )


def _distance(point_a: Vector3, point_b: Vector3) -> float:
    """Return the Euclidean distance between two points."""
    return _vector_magnitude(_vector_subtract(point_a, point_b))


def _rotation_matrix(axis: Vector3, angle_degrees: float) -> Matrix3:
    """Return a right-handed axis-angle rotation matrix."""
    x, y, z = _normalize(axis)
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    one_minus_cosine = 1.0 - cosine

    return (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )


def _translation_about_origin(
    linear: Matrix3,
    origin: Vector3,
) -> Vector3:
    """Return t for p' = A*p+t when A is applied about a specified origin."""
    return _vector_subtract(
        origin,
        _matrix_vector_multiply(linear, origin),
    )


def _apply_affine(point: Vector3, specification: AffineSpecification) -> Vector3:
    """Apply an affine specification to a point."""
    return _vector_add(
        _matrix_vector_multiply(specification.linear, point),
        specification.translation,
    )


def _matrix4(specification: AffineSpecification):
    """Return a row-major 4 x 4 matrix with translation in the last column."""
    linear = specification.linear
    translation = specification.translation
    return [
        [linear[0][0], linear[0][1], linear[0][2], translation[0]],
        [linear[1][0], linear[1][1], linear[1][2], translation[1]],
        [linear[2][0], linear[2][1], linear[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


@lru_cache(maxsize=None)
def _affine_specification(operation_name: str) -> AffineSpecification:
    """Return the independently calculated affine specification of an operation."""
    if operation_name == "translate":
        return AffineSpecification(
            name=operation_name,
            linear=_identity_matrix(),
            translation=TRANSLATION,
        )

    if operation_name == "scale":
        linear = _diagonal_matrix(SCALE_FACTORS)
        return AffineSpecification(
            name=operation_name,
            linear=linear,
            translation=_translation_about_origin(linear, SCALE_ORIGIN),
        )

    if operation_name == "rotate":
        linear = _rotation_matrix(ROTATION_AXIS, ROTATION_ANGLE)
        return AffineSpecification(
            name=operation_name,
            linear=linear,
            translation=_translation_about_origin(linear, ROTATION_ORIGIN),
        )

    if operation_name == "transform":
        rotation = _rotation_matrix(TRANSFORM_AXIS, TRANSFORM_ANGLE)
        scale = _diagonal_matrix(TRANSFORM_SCALE_FACTORS)

        # Topology.Transform's documented column-vector decomposition assumes
        # the linear block is rotation * axis-aligned scale.
        linear = _matrix_multiply(rotation, scale)

        return AffineSpecification(
            name=operation_name,
            linear=linear,
            translation=TRANSFORM_TRANSLATION,
        )

    raise ValueError(f"Unsupported transformation operation: {operation_name!r}")


def _cell():
    """Create the source prismatic cell."""
    cell = Cell.Prism(
        origin=_vertex(BASE_ORIGIN),
        width=PRISM_WIDTH,
        length=PRISM_LENGTH,
        height=PRISM_HEIGHT,
        uSides=1,
        vSides=1,
        wSides=1,
        direction=[0.0, 0.0, 1.0],
        placement="center",
        tolerance=TOPOLOGY_TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _source_topology(topology_type: str):
    """Create one deterministic source topology."""
    if topology_type == "vertex":
        topology = _vertex(BASE_ORIGIN)

    elif topology_type == "edge":
        topology = Edge.ByVertices(
            _vertex(EDGE_START),
            _vertex(EDGE_END),
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    elif topology_type == "wire":
        topology = Wire.Rectangle(
            origin=_vertex(BASE_ORIGIN),
            width=PRISM_WIDTH,
            length=PRISM_LENGTH,
            direction=[0.0, 0.0, 1.0],
            placement="center",
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    elif topology_type == "face":
        topology = Face.Rectangle(
            origin=_vertex(BASE_ORIGIN),
            width=PRISM_WIDTH,
            length=PRISM_LENGTH,
            direction=[0.0, 0.0, 1.0],
            placement="center",
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    elif topology_type == "shell":
        cell = _cell()
        faces = Topology.Faces(cell, silent=True)
        assert isinstance(faces, list)
        assert len(faces) == 6
        topology = Shell.ByFaces(
            faces,
            transferDictionaries=False,
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    elif topology_type == "cell":
        topology = _cell()

    elif topology_type == "cellcomplex":
        cx, cy, cz = BASE_ORIGIN
        radius = CELL_COMPLEX_RADIUS

        west = _vertex((cx - radius, cy, cz))
        south = _vertex((cx, cy - radius, cz))
        east = _vertex((cx + radius, cy, cz))
        north = _vertex((cx, cy + radius, cz))
        top = _vertex((cx, cy, cz + radius))
        bottom = _vertex((cx, cy, cz - radius))

        face_vertices = (
            (top, west, south),
            (top, south, east),
            (top, east, north),
            (top, north, west),
            (bottom, west, south),
            (bottom, south, east),
            (bottom, east, north),
            (bottom, north, west),
            (west, south, east, north),
        )

        faces = [
            Face.ByVertices(
                list(vertices),
                tolerance=TOPOLOGY_TOLERANCE,
                silent=True,
            )
            for vertices in face_vertices
        ]
        assert all(Topology.IsInstance(face, "Face") for face in faces)

        topology = CellComplex.ByFaces(
            faces,
            transferDictionaries=False,
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    else:
        raise ValueError(f"Unsupported topology type: {topology_type!r}")

    assert Topology.IsInstance(topology, topology_type)
    return topology


def _apply_operation(
    topology,
    operation_name: str,
    specification: AffineSpecification,
):
    """Apply one TopologicPy transformation operation."""
    if operation_name == "translate":
        return Topology.Translate(
            topology,
            x=TRANSLATION[0],
            y=TRANSLATION[1],
            z=TRANSLATION[2],
            transferDictionaries=False,
            silent=True,
        )

    if operation_name == "scale":
        return Topology.Scale(
            topology,
            origin=_vertex(SCALE_ORIGIN),
            x=SCALE_FACTORS[0],
            y=SCALE_FACTORS[1],
            z=SCALE_FACTORS[2],
            transferDictionaries=False,
            silent=True,
        )

    if operation_name == "rotate":
        return Topology.Rotate(
            topology,
            origin=_vertex(ROTATION_ORIGIN),
            axis=list(ROTATION_AXIS),
            angle=ROTATION_ANGLE,
            transferDictionaries=False,
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    if operation_name == "transform":
        return Topology.Transform(
            topology,
            matrix=_matrix4(specification),
            transferDictionaries=False,
            tolerance=TOPOLOGY_TOLERANCE,
            silent=True,
        )

    raise ValueError(f"Unsupported transformation operation: {operation_name!r}")


def _vertices(topology):
    """Return all vertices of a valid topology."""
    vertices = Topology.Vertices(topology, silent=True)
    assert isinstance(vertices, list)
    return vertices


def _edges(topology):
    """Return all edges of a valid topology."""
    edges = Topology.Edges(topology, silent=True)
    assert isinstance(edges, list)
    return edges


def _faces(topology):
    """Return all faces of a valid topology."""
    faces = Topology.Faces(topology, silent=True)
    assert isinstance(faces, list)
    return faces


def _cells(topology):
    """Return all cells of a valid topology."""
    cells = Topology.Cells(topology, silent=True)
    assert isinstance(cells, list)
    return cells


def _count_signature(topology) -> Tuple[int, int, int, int]:
    """Return counts in the fixed order: cells, faces, edges, vertices."""
    return (
        len(_cells(topology)),
        len(_faces(topology)),
        len(_edges(topology)),
        len(_vertices(topology)),
    )


def _vertex_centroid(topology) -> Vector3:
    """Return the arithmetic centroid of all topology vertices."""
    coordinates = [_coordinates(vertex) for vertex in _vertices(topology)]
    assert len(coordinates) > 0

    count = float(len(coordinates))
    return (
        sum(point[0] for point in coordinates) / count,
        sum(point[1] for point in coordinates) / count,
        sum(point[2] for point in coordinates) / count,
    )


def _edge_endpoint_coordinates(edge) -> Tuple[Vector3, Vector3]:
    """Return an edge's start and end coordinates."""
    return (
        _coordinates(Edge.StartVertex(edge)),
        _coordinates(Edge.EndVertex(edge)),
    )


def _measured_edge_lengths(topology) -> Tuple[float, ...]:
    """Calculate all edge lengths directly from endpoint coordinates."""
    lengths = []
    for edge in _edges(topology):
        start, end = _edge_endpoint_coordinates(edge)
        lengths.append(_distance(start, end))
    return tuple(sorted(lengths))


def _measured_face_areas(topology) -> Tuple[float, ...]:
    """Return the sorted areas of all faces."""
    areas = []
    for face in _faces(topology):
        area = Face.Area(face, mantissa=MEASUREMENT_MANTISSA)
        assert isinstance(area, (int, float))
        areas.append(float(area))
    return tuple(sorted(areas))


def _measured_cell_volumes(topology) -> Tuple[float, ...]:
    """Return the sorted volumes of all cells."""
    volumes = []
    for cell in _cells(topology):
        volume = Cell.Volume(cell, mantissa=MEASUREMENT_MANTISSA)
        assert isinstance(volume, (int, float))
        volumes.append(float(volume))
    return tuple(sorted(volumes))


def _face_normal(face) -> Vector3:
    """Return one normalized face-normal vector."""
    normal = Face.Normal(
        face,
        outputType="xyz",
        mantissa=MEASUREMENT_MANTISSA,
    )
    assert isinstance(normal, list)
    assert len(normal) == 3
    return _normalize(
        (
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
        )
    )


def _measured_face_normals(topology) -> Tuple[Vector3, ...]:
    """Return all normalized face-normal vectors without assuming face order."""
    return tuple(_face_normal(face) for face in _faces(topology))


def _expected_edge_lengths(
    topology,
    specification: AffineSpecification,
) -> Tuple[float, ...]:
    """Calculate transformed edge lengths from source edge endpoints."""
    lengths = []
    for edge in _edges(topology):
        source_start, source_end = _edge_endpoint_coordinates(edge)
        expected_start = _apply_affine(source_start, specification)
        expected_end = _apply_affine(source_end, specification)
        lengths.append(_distance(expected_start, expected_end))
    return tuple(sorted(lengths))


def _normal_area_vector(
    source_normal: Vector3,
    linear: Matrix3,
) -> Vector3:
    """
    Transform an oriented unit-area vector by det(A) * inverse(A).T.

    Its direction gives the expected oriented normal. Its magnitude is the
    affine area scale for a face with the supplied unit normal.
    """
    determinant = _determinant(linear)
    inverse_transpose = _transpose(_inverse(linear))
    return _vector_multiply(
        _matrix_vector_multiply(inverse_transpose, source_normal),
        determinant,
    )


def _expected_face_areas(
    topology,
    specification: AffineSpecification,
) -> Tuple[float, ...]:
    """Calculate transformed face areas with the affine area-vector formula."""
    areas = []
    for face in _faces(topology):
        source_area = Face.Area(face, mantissa=MEASUREMENT_MANTISSA)
        assert isinstance(source_area, (int, float))
        source_normal = _face_normal(face)
        area_scale = _vector_magnitude(
            _normal_area_vector(source_normal, specification.linear)
        )
        areas.append(float(source_area) * area_scale)
    return tuple(sorted(areas))


def _expected_cell_volumes(
    topology,
    specification: AffineSpecification,
) -> Tuple[float, ...]:
    """Calculate transformed cell volumes with the affine determinant."""
    volume_scale = abs(_determinant(specification.linear))
    volumes = []
    for cell in _cells(topology):
        source_volume = Cell.Volume(cell, mantissa=MEASUREMENT_MANTISSA)
        assert isinstance(source_volume, (int, float))
        volumes.append(float(source_volume) * volume_scale)
    return tuple(sorted(volumes))


def _expected_face_normals(
    topology,
    specification: AffineSpecification,
) -> Tuple[Vector3, ...]:
    """Calculate transformed oriented face normals with inverse-transpose."""
    normals = []
    for face in _faces(topology):
        source_normal = _face_normal(face)
        normals.append(
            _normalize(
                _normal_area_vector(source_normal, specification.linear)
            )
        )
    return tuple(normals)


def _expected_metrics(
    source,
    topology_type: str,
    operation_name: str,
    specification: AffineSpecification,
) -> GeometryMetrics:
    """Return the independent affine ground truth for one test case."""
    include_normals = operation_name in NORMAL_OPERATION_NAMES
    return GeometryMetrics(
        result_type=topology_type,
        counts=_count_signature(source),
        location=_apply_affine(_vertex_centroid(source), specification),
        edge_lengths=_expected_edge_lengths(source, specification),
        face_areas=_expected_face_areas(source, specification),
        cell_volumes=_expected_cell_volumes(source, specification),
        face_normals=(
            _expected_face_normals(source, specification)
            if include_normals
            else ()
        ),
    )


def _observed_metrics(
    result,
    operation_name: str,
) -> GeometryMetrics:
    """Measure one returned transformation result."""
    include_normals = operation_name in NORMAL_OPERATION_NAMES
    return GeometryMetrics(
        result_type=Topology.TypeAsString(result, silent=True).lower(),
        counts=_count_signature(result),
        location=_vertex_centroid(result),
        edge_lengths=_measured_edge_lengths(result),
        face_areas=_measured_face_areas(result),
        cell_volumes=_measured_cell_volumes(result),
        face_normals=(
            _measured_face_normals(result)
            if include_normals
            else ()
        ),
    )


def _scalar_close(
    actual: float,
    expected: float,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    """Return True when two scalar measurements are sufficiently close."""
    return math.isclose(
        float(actual),
        float(expected),
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    )


def _vector_close(
    actual: Vector3,
    expected: Vector3,
    tolerance: float,
) -> bool:
    """Return True when all corresponding vector components are close."""
    return all(
        math.isclose(
            float(actual[index]),
            float(expected[index]),
            abs_tol=tolerance,
            rel_tol=RELATIVE_TOLERANCE,
        )
        for index in range(3)
    )


def _scalar_sequence_mismatches(
    label: str,
    actual: Sequence[float],
    expected: Sequence[float],
) -> Tuple[str, ...]:
    """Return detailed scalar-sequence comparison failures."""
    mismatches = []

    if len(actual) != len(expected):
        mismatches.append(
            f"{label} count mismatch: expected {len(expected)}, "
            f"observed {len(actual)}."
        )
        return tuple(mismatches)

    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected)
    ):
        if not _scalar_close(actual_value, expected_value):
            mismatches.append(
                f"{label}[{index}] mismatch: expected {expected_value:.12g}, "
                f"observed {actual_value:.12g}."
            )

    return tuple(mismatches)


def _normal_multiset_mismatches(
    actual: Sequence[Vector3],
    expected: Sequence[Vector3],
) -> Tuple[str, ...]:
    """Compare face normals as an unordered, orientation-sensitive multiset."""
    if len(actual) != len(expected):
        return (
            "face normal count mismatch: "
            f"expected {len(expected)}, observed {len(actual)}.",
        )

    unmatched_actual = list(actual)
    mismatches = []

    for expected_normal in expected:
        if len(unmatched_actual) == 0:
            mismatches.append(
                f"No observed normal remained for expected {expected_normal}."
            )
            continue

        nearest_index = min(
            range(len(unmatched_actual)),
            key=lambda index: _distance(
                unmatched_actual[index],
                expected_normal,
            ),
        )
        nearest = unmatched_actual[nearest_index]

        if not _vector_close(
            nearest,
            expected_normal,
            tolerance=NORMAL_TOLERANCE,
        ):
            mismatches.append(
                "face normal mismatch: "
                f"expected {expected_normal}, nearest observed {nearest}, "
                f"vector distance {_distance(nearest, expected_normal):.12g}."
            )
        else:
            unmatched_actual.pop(nearest_index)

    return tuple(mismatches)


def _metric_mismatches(
    expected: GeometryMetrics,
    observed: GeometryMetrics,
) -> Tuple[str, ...]:
    """Return all mismatches between expected and observed case metrics."""
    mismatches = []

    if observed.result_type != expected.result_type:
        mismatches.append(
            "result type mismatch: "
            f"expected {expected.result_type!r}, "
            f"observed {observed.result_type!r}."
        )

    if observed.counts != expected.counts:
        mismatches.append(
            "subtopology count mismatch: "
            f"expected (cells, faces, edges, vertices) = {expected.counts}, "
            f"observed = {observed.counts}."
        )

    if not _vector_close(
        observed.location,
        expected.location,
        tolerance=ABSOLUTE_TOLERANCE,
    ):
        mismatches.append(
            "location mismatch: "
            f"expected {expected.location}, "
            f"observed {observed.location}."
        )

    mismatches.extend(
        _scalar_sequence_mismatches(
            "edge length",
            observed.edge_lengths,
            expected.edge_lengths,
        )
    )
    mismatches.extend(
        _scalar_sequence_mismatches(
            "face area",
            observed.face_areas,
            expected.face_areas,
        )
    )
    mismatches.extend(
        _scalar_sequence_mismatches(
            "cell volume",
            observed.cell_volumes,
            expected.cell_volumes,
        )
    )
    mismatches.extend(
        _normal_multiset_mismatches(
            observed.face_normals,
            expected.face_normals,
        )
    )

    return tuple(mismatches)


@lru_cache(maxsize=None)
def _evaluate_case(
    topology_type: str,
    operation_name: str,
) -> CaseEvaluation:
    """Construct, transform, measure, and compare one matrix entry."""
    try:
        source = _source_topology(topology_type)
        specification = _affine_specification(operation_name)
        expected = _expected_metrics(
            source,
            topology_type,
            operation_name,
            specification,
        )

        result = _apply_operation(
            source,
            operation_name,
            specification,
        )
        if not Topology.IsInstance(result, "Topology"):
            return CaseEvaluation(
                topology_type=topology_type,
                operation_name=operation_name,
                expected=expected,
                observed=None,
                mismatches=(),
                error="The transformation returned no valid topology.",
            )

        observed = _observed_metrics(result, operation_name)
        return CaseEvaluation(
            topology_type=topology_type,
            operation_name=operation_name,
            expected=expected,
            observed=observed,
            mismatches=_metric_mismatches(expected, observed),
            error=None,
        )

    except Exception as error:
        return CaseEvaluation(
            topology_type=topology_type,
            operation_name=operation_name,
            expected=None,
            observed=None,
            mismatches=(),
            error=f"{type(error).__name__}: {error}",
        )


def _source_ground_truth_mismatches(topology_type: str) -> Tuple[str, ...]:
    """Compare a source topology with its explicit construction ground truth."""
    topology = _source_topology(topology_type)
    mismatches = []

    actual_counts = _count_signature(topology)
    expected_counts = BASE_COUNTS[topology_type]
    if actual_counts != expected_counts:
        mismatches.append(
            "source count mismatch: "
            f"expected {expected_counts}, observed {actual_counts}."
        )

    actual_location = _vertex_centroid(topology)
    if not _vector_close(
        actual_location,
        BASE_ORIGIN,
        tolerance=ABSOLUTE_TOLERANCE,
    ):
        mismatches.append(
            "source location mismatch: "
            f"expected {BASE_ORIGIN}, observed {actual_location}."
        )

    mismatches.extend(
        _scalar_sequence_mismatches(
            "source edge length",
            _measured_edge_lengths(topology),
            tuple(sorted(BASE_EDGE_LENGTHS[topology_type])),
        )
    )
    mismatches.extend(
        _scalar_sequence_mismatches(
            "source face area",
            _measured_face_areas(topology),
            tuple(sorted(BASE_FACE_AREAS[topology_type])),
        )
    )
    mismatches.extend(
        _scalar_sequence_mismatches(
            "source cell volume",
            _measured_cell_volumes(topology),
            tuple(sorted(BASE_CELL_VOLUMES[topology_type])),
        )
    )

    return tuple(mismatches)


def _rounded_value(value):
    """Round nested floating-point report values without changing their shape."""
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, tuple):
        return tuple(_rounded_value(item) for item in value)
    if isinstance(value, list):
        return [_rounded_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _rounded_value(item)
            for key, item in value.items()
        }
    return value


def _formatted_backend_report() -> str:
    """Return a JSON report of analytic and active-backend measurements."""
    report = {}

    for topology_type in TOPOLOGY_TYPES:
        for operation_name in OPERATION_NAMES:
            evaluation = _evaluate_case(topology_type, operation_name)
            key = f"{topology_type}/{operation_name}"
            report[key] = {
                "expected": (
                    _rounded_value(asdict(evaluation.expected))
                    if evaluation.expected is not None
                    else None
                ),
                "observed": (
                    _rounded_value(asdict(evaluation.observed))
                    if evaluation.observed is not None
                    else None
                ),
                "mismatches": list(evaluation.mismatches),
                "error": evaluation.error,
            }

    return json.dumps(report, indent=2, sort_keys=True)


@pytest.mark.parametrize("topology_type", TOPOLOGY_TYPES)
def test_source_topology_ground_truth(topology_type):
    """Verify every deterministic source topology before transforming it."""
    mismatches = _source_ground_truth_mismatches(topology_type)
    assert not mismatches, (
        f"Invalid source topology ground truth for {topology_type}:\n"
        + "\n".join(f"  - {message}" for message in mismatches)
    )


@pytest.mark.parametrize("topology_type", TOPOLOGY_TYPES)
@pytest.mark.parametrize("operation_name", OPERATION_NAMES)
def test_transformation_geometry(topology_type, operation_name):
    """Verify one topology/operation transformation matrix entry."""
    evaluation = _evaluate_case(topology_type, operation_name)

    assert evaluation.error is None, (
        f"{topology_type}/{operation_name} failed: {evaluation.error}"
    )
    assert not evaluation.mismatches, (
        f"Transformation mismatch for {topology_type}/{operation_name}:\n"
        + "\n".join(
            f"  - {message}"
            for message in evaluation.mismatches
        )
    )

def test_legacy_backend_report():
    """
    Verify all analytic and active-backend transformation measurements.

    The detailed comparison report is printed only when
    TOPOLOGICPY_TRANSFORMATION_REPORT=1, but the validation itself is always
    executed so this test is never skipped.
    """
    report = _formatted_backend_report()

    if os.getenv(REPORT_ENVIRONMENT_VARIABLE) == "1":
        print("\nTRANSFORMATION BACKEND COMPARISON\n")
        print(report)

    failures = []

    for topology_type in TOPOLOGY_TYPES:
        for operation_name in OPERATION_NAMES:
            evaluation = _evaluate_case(topology_type, operation_name)

            if evaluation.error is not None:
                failures.append(
                    f"{topology_type}/{operation_name}: {evaluation.error}"
                )

            for mismatch in evaluation.mismatches:
                failures.append(
                    f"{topology_type}/{operation_name}: {mismatch}"
                )

    assert not failures, (
        "The active backend differs from the analytic ground truth:\n"
        + "\n".join(f"  - {failure}" for failure in failures)
    )
