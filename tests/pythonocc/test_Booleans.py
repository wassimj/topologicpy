# Auto-generated PythonOCC backend parity test
# Copied from test_Booleans.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Copyright (C) 2026
# Wassim Jabi
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.

"""
Regression tests for TopologicPy boolean operations.

The test matrix covers these topology types:

    Vertex, Edge, Wire, Face, Shell, Cell, CellComplex

and these boolean operations:

    Union, Difference, Intersection, Symmetric Difference (XOR),
    Merge, Slice, Impose, Imprint

Each matrix row uses two deterministic, overlapping operands of the same
topology type. Every result is reduced to this count signature:

    (number_of_cells, number_of_faces, number_of_edges, number_of_vertices)

Ground-truth workflow
---------------------
1. Run only the collector with the trusted legacy backend:

       pytest -q test_Booleans.py::test_ground_truth_table_is_complete

2. Copy the EXPECTED_COUNTS table printed in the failure message and replace
   the placeholder table below.

3. Run this file against both backends:

       pytest -q test_Booleans.py

The collector intentionally fails while any expected entry is None. Once the
table is complete, it becomes a normal passing completeness check.
"""

from functools import lru_cache

import pytest

from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


TOLERANCE = 0.0001

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
    "union",
    "difference",
    "intersection",
    "xor",
    "merge",
    "slice",
    "impose",
    "imprint",
)

# Count tuple order:
#     (cells, faces, edges, vertices)
#
# Replace every None with the tuple reported by the trusted legacy backend.
EXPECTED_COUNTS = {
    ("vertex", "union"): (0, 0, 0, 1),  # result: Vertex
    ("vertex", "difference"): (0, 0, 0, 0),  # result: None
    ("vertex", "intersection"): (0, 0, 0, 1),  # result: Vertex
    ("vertex", "xor"): (0, 0, 0, 0),  # result: None
    ("vertex", "merge"): (0, 0, 0, 1),  # result: Vertex
    ("vertex", "slice"): (0, 0, 0, 1),  # result: Vertex
    ("vertex", "impose"): (0, 0, 0, 1),  # result: Vertex
    ("vertex", "imprint"): (0, 0, 0, 1),  # result: Vertex

    ("edge", "union"): (0, 0, 1, 2),  # result: Edge
    ("edge", "difference"): (0, 0, 1, 2),  # result: Edge
    ("edge", "intersection"): (0, 0, 1, 2),  # result: Edge
    ("edge", "xor"): (0, 0, 2, 4),  # result: Cluster
    ("edge", "merge"): (0, 0, 3, 4),  # result: Wire
    ("edge", "slice"): (0, 0, 2, 3),  # result: Wire
    ("edge", "impose"): (0, 0, 2, 4),  # result: Cluster
    ("edge", "imprint"): (0, 0, 2, 3),  # result: Wire

    ("wire", "union"): (0, 0, 10, 8),  # result: Wire
    ("wire", "difference"): (0, 0, 4, 6),  # result: Cluster
    ("wire", "intersection"): (0, 0, 2, 4),  # result: Cluster
    ("wire", "xor"): (0, 0, 8, 8),  # result: Cluster
    ("wire", "merge"): (0, 0, 10, 8),  # result: Wire
    ("wire", "slice"): (0, 0, 6, 6),  # result: Wire
    ("wire", "impose"): (0, 0, 8, 12),  # result: Cluster
    ("wire", "imprint"): (0, 0, 6, 6),  # result: Wire

    ("face", "union"): (0, 1, 8, 8),  # result: Face
    ("face", "difference"): (0, 1, 4, 4),  # result: Face
    ("face", "intersection"): (0, 1, 4, 4),  # result: Cluster
    ("face", "xor"): (0, 2, 8, 8),  # result: Cluster
    ("face", "merge"): (0, 3, 10, 8),  # result: Shell
    ("face", "slice"): (0, 2, 7, 6),  # result: Shell
    ("face", "impose"): (0, 2, 9, 8),  # result: Shell
    ("face", "imprint"): (0, 2, 7, 6),  # result: Shell

    ("shell", "union"): (0, 16, 28, 16),  # result: Cluster
    ("shell", "difference"): (0, 6, 16, 12),  # result: Cluster
    ("shell", "intersection"): (0, 4, 12, 8),  # result: Cluster
    ("shell", "xor"): (0, 12, 24, 16),  # result: Cluster
    ("shell", "merge"): (0, 16, 28, 16),  # result: Shell
    ("shell", "slice"): (0, 10, 20, 12),  # result: Shell
    ("shell", "impose"): (0, 12, 28, 16),  # result: Cluster
    ("shell", "imprint"): (0, 10, 20, 12),  # result: Shell

    ("cell", "union"): (1, 14, 28, 16),  # result: Cell
    ("cell", "difference"): (1, 6, 12, 8),  # result: Cell
    ("cell", "intersection"): (1, 6, 12, 8),  # result: Cluster
    ("cell", "xor"): (2, 12, 24, 16),  # result: Cluster
    ("cell", "merge"): (3, 16, 28, 16),  # result: CellComplex
    ("cell", "slice"): (2, 11, 20, 12),  # result: CellComplex
    ("cell", "impose"): (2, 15, 28, 16),  # result: CellComplex
    ("cell", "imprint"): (2, 11, 20, 12),  # result: CellComplex

    ("cellcomplex", "union"): (1, 22, 44, 24),  # result: Cell
    ("cellcomplex", "difference"): (1, 6, 12, 8),  # result: Cell
    ("cellcomplex", "intersection"): (6, 16, 28, 16),  # result: Cluster
    ("cellcomplex", "xor"): (2, 12, 24, 16),  # result: Cluster
    ("cellcomplex", "merge"): (5, 26, 44, 24),  # result: CellComplex
    ("cellcomplex", "slice"): (4, 21, 36, 20),  # result: CellComplex
    ("cellcomplex", "impose"): (3, 24, 44, 24),  # result: CellComplex
    ("cellcomplex", "imprint"): (4, 21, 36, 20),  # result: CellComplex

}


def _vertex(x, y=0.0, z=0.0):
    """Create a vertex and fail immediately if construction is unsuccessful."""
    vertex = Vertex.ByCoordinates(x, y, z)
    assert Topology.IsInstance(vertex, "Vertex")
    return vertex


def _cell(origin_x):
    """Create a deterministic 2 x 2 x 2 prismatic cell."""
    cell = Cell.Prism(
        origin=_vertex(origin_x, 0.0, 0.0),
        width=2.0,
        length=2.0,
        height=2.0,
        uSides=1,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(cell, "Cell")
    return cell


def _shell(origin_x):
    """Create a closed shell from the faces of a deterministic prismatic cell."""
    cell = _cell(origin_x)
    faces = Topology.Faces(cell, silent=True)
    assert isinstance(faces, list)
    assert len(faces) > 0

    shell = Shell.ByFaces(
        faces,
        transferDictionaries=False,
        tolerance=TOLERANCE,
        silent=True,
    )
    assert Topology.IsInstance(shell, "Shell")
    return shell


def _cellcomplex(origin_x):
    """Create a deterministic two-cell prismatic cell complex."""
    cellcomplex = CellComplex.Prism(
        origin=_vertex(origin_x, 0.0, 0.0),
        width=2.0,
        length=2.0,
        height=2.0,
        uSides=2,
        vSides=1,
        wSides=1,
        placement="center",
        tolerance=TOLERANCE,
    )
    assert Topology.IsInstance(cellcomplex, "CellComplex")
    return cellcomplex


def _operands(topology_type):
    """
    Return two deterministic operands for the requested topology type.

    Except for vertices, operand B is translated 0.75 units along X. This
    creates a non-trivial partial overlap without introducing near-tolerance
    coordinates. The two vertices are coincident because a zero-dimensional
    topology cannot partially overlap another vertex.
    """
    if topology_type == "vertex":
        return _vertex(0.0, 0.0, 0.0), _vertex(0.0, 0.0, 0.0)

    if topology_type == "edge":
        topology_a = Edge.ByVertices(
            [_vertex(-1.0, 0.0, 0.0), _vertex(1.0, 0.0, 0.0)],
            tolerance=TOLERANCE,
            silent=True,
        )
        topology_b = Edge.ByVertices(
            [_vertex(-0.25, 0.0, 0.0), _vertex(1.75, 0.0, 0.0)],
            tolerance=TOLERANCE,
            silent=True,
        )

    elif topology_type == "wire":
        topology_a = Wire.Rectangle(
            origin=_vertex(0.0, 0.0, 0.0),
            width=2.0,
            length=2.0,
            placement="center",
            tolerance=TOLERANCE,
            silent=True,
        )
        topology_b = Wire.Rectangle(
            origin=_vertex(0.75, 0.0, 0.0),
            width=2.0,
            length=2.0,
            placement="center",
            tolerance=TOLERANCE,
            silent=True,
        )

    elif topology_type == "face":
        topology_a = Face.Rectangle(
            origin=_vertex(0.0, 0.0, 0.0),
            width=2.0,
            length=2.0,
            placement="center",
            tolerance=TOLERANCE,
            silent=True,
        )
        topology_b = Face.Rectangle(
            origin=_vertex(0.75, 0.0, 0.0),
            width=2.0,
            length=2.0,
            placement="center",
            tolerance=TOLERANCE,
            silent=True,
        )

    elif topology_type == "shell":
        topology_a = _shell(0.0)
        topology_b = _shell(0.75)

    elif topology_type == "cell":
        topology_a = _cell(0.0)
        topology_b = _cell(0.75)

    elif topology_type == "cellcomplex":
        topology_a = _cellcomplex(0.0)
        topology_b = _cellcomplex(0.75)

    else:
        raise ValueError(f"Unsupported topology type: {topology_type!r}")

    assert Topology.IsInstance(topology_a, topology_type)
    assert Topology.IsInstance(topology_b, topology_type)
    return topology_a, topology_b


def _apply_operation(operation_name, topology_a, topology_b):
    """Apply one supported TopologicPy boolean operation."""
    operations = {
        "union": Topology.Union,
        "difference": Topology.Difference,
        "intersection": Topology.Intersect,
        "xor": Topology.SymmetricDifference,
        "merge": Topology.Merge,
        "slice": Topology.Slice,
        "impose": Topology.Impose,
        "imprint": Topology.Imprint,
    }
    operation = operations[operation_name]
    return operation(
        topology_a,
        topology_b,
        tranDict=False,
        tolerance=TOLERANCE,
        silent=True,
    )


def _subtopology_count(topology, extractor):
    """Return zero for an absent result; otherwise count extracted subtopologies."""
    if not Topology.IsInstance(topology, "Topology"):
        return 0

    subtopologies = extractor(topology, silent=True)
    if subtopologies is None:
        return 0
    assert isinstance(subtopologies, list)
    return len(subtopologies)


def _count_signature(topology):
    """Return counts in the fixed order: cells, faces, edges, vertices."""
    return (
        _subtopology_count(topology, Topology.Cells),
        _subtopology_count(topology, Topology.Faces),
        _subtopology_count(topology, Topology.Edges),
        _subtopology_count(topology, Topology.Vertices),
    )


@lru_cache(maxsize=None)
def _evaluate_case(topology_type, operation_name):
    """
    Build one case, run its boolean operation, and return diagnostics.

    Returns
    -------
    tuple
        A tuple of (count_signature, result_type, error_text).
    """
    try:
        topology_a, topology_b = _operands(topology_type)
        result = _apply_operation(operation_name, topology_a, topology_b)
        counts = _count_signature(result)

        if Topology.IsInstance(result, "Topology"):
            result_type = Topology.TypeAsString(result)
        else:
            result_type = "None"

        return counts, result_type, None

    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _formatted_ground_truth_table():
    """Return a complete EXPECTED_COUNTS Python literal from the active backend."""
    lines = [
        "EXPECTED_COUNTS = {",
    ]
    errors = []

    for topology_type in TOPOLOGY_TYPES:
        for operation_name in OPERATION_NAMES:
            counts, result_type, error = _evaluate_case(
                topology_type,
                operation_name,
            )
            key = f'("{topology_type}", "{operation_name}")'
            if error is None:
                lines.append(
                    f"    {key}: {counts},  # result: {result_type}"
                )
            else:
                lines.append(
                    f"    {key}: None,  # ERROR: {error}"
                )
                errors.append(
                    f"{topology_type}/{operation_name}: {error}"
                )
        lines.append("")

    lines.append("}")

    if errors:
        lines.extend(
            [
                "",
                "Errors encountered while collecting ground truth:",
                *[f"  - {error}" for error in errors],
            ]
        )

    return "\n".join(lines)


def test_ground_truth_table_is_complete():
    """
    Fail once with a complete backend-generated table while placeholders remain.
    """
    expected_keys = {
        (topology_type, operation_name)
        for topology_type in TOPOLOGY_TYPES
        for operation_name in OPERATION_NAMES
    }
    actual_keys = set(EXPECTED_COUNTS)

    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    placeholder_keys = sorted(
        key for key in expected_keys if EXPECTED_COUNTS.get(key) is None
    )

    if missing_keys or unexpected_keys or placeholder_keys:
        details = []

        if missing_keys:
            details.append(f"Missing keys: {missing_keys}")

        if unexpected_keys:
            details.append(f"Unexpected keys: {unexpected_keys}")

        if placeholder_keys:
            details.append(
                "Replace the placeholder EXPECTED_COUNTS table with this "
                "table generated by the active backend:\n\n"
                + _formatted_ground_truth_table()
            )

        pytest.fail("\n\n".join(details), pytrace=False)


@pytest.mark.parametrize("topology_type", TOPOLOGY_TYPES)
@pytest.mark.parametrize("operation_name", OPERATION_NAMES)
def test_boolean_subtopology_counts(topology_type, operation_name):
    """
    Assert cell, face, edge, and vertex counts for one boolean matrix entry.
    """
    expected = EXPECTED_COUNTS[(topology_type, operation_name)]
    if expected is None:
        pytest.skip(
            "Ground truth has not yet been recorded for "
            f"{topology_type}/{operation_name}."
        )

    actual, result_type, error = _evaluate_case(topology_type, operation_name)

    assert error is None, (
        f"{topology_type}/{operation_name} raised an exception: {error}"
    )
    assert actual == expected, (
        f"Boolean count mismatch for {topology_type}/{operation_name}. "
        f"Result type: {result_type}. "
        f"Expected (cells, faces, edges, vertices) = {expected}; "
        f"actual = {actual}."
    )
