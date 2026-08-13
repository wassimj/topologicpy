# Auto-generated PythonOCC backend parity test
# Copied from test_TQueries.py

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
Regression tests for TopologicPy topological queries.

The suite tests the number and type of objects returned by:

    * Topology.SubTopologies
    * Topology.SuperTopologies
    * Topology.AdjacentTopologies
    * Topology.SharedTopologies
    * Topology.SharedVertices
    * Topology.SharedEdges
    * Topology.SharedWires
    * Topology.SharedFaces

The deterministic host is a two-cell octahedral CellComplex. It consists of
an upper pyramidal cell and a lower pyramidal cell that share one quadrilateral
equatorial face.

Its exact incidence structure is:

    CellComplex:
        6 vertices
        12 edges
        9 wires
        9 faces
        2 shells
        2 cells
        1 cellcomplex

    Each pyramidal cell:
        5 vertices
        8 edges
        5 wires
        5 faces
        1 shell
        1 cell

    Shared equatorial face:
        4 vertices
        4 edges
        1 wire
        1 face

No Boolean slicing is used to construct the test host.
"""

from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest

from topologicpy.CellComplex import CellComplex
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


TOLERANCE = 0.0001
COORDINATE_MANTISSA = 9

# Coordinates produced by CellComplex.Octahedron with radius=1.
WEST = (-1.0, 0.0, 0.0)
SOUTH = (0.0, -1.0, 0.0)
EAST = (1.0, 0.0, 0.0)
NORTH = (0.0, 1.0, 0.0)
TOP = (0.0, 0.0, 1.0)
BOTTOM = (0.0, 0.0, -1.0)

SHARED_KEYS = ("vertices", "edges", "wires", "faces")
SHARED_TYPES = {
    "vertices": "vertex",
    "edges": "edge",
    "wires": "wire",
    "faces": "face",
}


def _coordinates(vertex) -> Tuple[float, float, float]:
    """Return a vertex's rounded XYZ coordinates."""
    coordinates = Vertex.Coordinates(
        vertex,
        outputType="xyz",
        mantissa=COORDINATE_MANTISSA,
    )
    assert isinstance(coordinates, list)
    assert len(coordinates) == 3
    return tuple(float(value) for value in coordinates)


def _coordinate_key(
    coordinates: Sequence[float],
) -> Tuple[float, float, float]:
    """Return a stable rounded coordinate key."""
    return tuple(
        round(float(value), COORDINATE_MANTISSA)
        for value in coordinates
    )


def _vertex_signature(topology) -> Tuple[Tuple[float, float, float], ...]:
    """Return a topology signature formed from its sorted vertex coordinates."""
    vertices = Topology.Vertices(topology, silent=True)
    assert isinstance(vertices, list)
    return tuple(
        sorted(_coordinate_key(_coordinates(vertex)) for vertex in vertices)
    )


def _target_signature(
    coordinates: Iterable[Sequence[float]],
) -> Tuple[Tuple[float, float, float], ...]:
    """Return a sorted coordinate signature for target coordinates."""
    return tuple(sorted(_coordinate_key(point) for point in coordinates))


def _centroid_z(topology) -> float:
    """Return the Z coordinate of a topology centroid."""
    centroid = Topology.Centroid(topology)
    assert Topology.IsInstance(centroid, "Vertex")
    return float(Vertex.Z(centroid, mantissa=COORDINATE_MANTISSA))


def _type_name(topology) -> str:
    """Return a normalized Topologic type name."""
    try:
        type_name = Topology.TypeAsString(topology, silent=True)
    except TypeError:
        type_name = Topology.TypeAsString(topology)

    assert isinstance(type_name, str)
    return type_name.lower()


def _actual_type_names(topologies) -> List[str]:
    """Return normalized type names for diagnostics."""
    if not isinstance(topologies, list):
        return [type(topologies).__name__]
    return [_type_name(topology) for topology in topologies]


def _assert_topology_list(
    result,
    expected_count: int,
    expected_type: str,
    context: str,
):
    """Assert that a result is a list of the expected size and topology type."""
    assert isinstance(result, list), (
        f"{context} did not return a list. "
        f"Returned Python type: {type(result).__name__}."
    )
    assert len(result) == expected_count, (
        f"{context} returned {len(result)} objects; "
        f"expected {expected_count}. "
        f"Returned Topologic types: {_actual_type_names(result)}."
    )

    invalid_indices = [
        index
        for index, topology in enumerate(result)
        if not Topology.IsInstance(topology, expected_type)
    ]
    assert not invalid_indices, (
        f"{context} returned objects of the wrong type at indices "
        f"{invalid_indices}. Expected {expected_type!r}; "
        f"returned types: {_actual_type_names(result)}."
    )

    actual_types = [_type_name(topology) for topology in result]
    assert all(type_name == expected_type.lower() for type_name in actual_types), (
        f"{context} returned unexpected type names. "
        f"Expected only {expected_type!r}; returned {actual_types}."
    )


def _find_by_signature(
    topologies: Sequence,
    target_coordinates: Iterable[Sequence[float]],
    topology_type: str,
):
    """Find exactly one topology with the requested vertex-coordinate signature."""
    target = _target_signature(target_coordinates)
    matches = [
        topology
        for topology in topologies
        if Topology.IsInstance(topology, topology_type)
        and _vertex_signature(topology) == target
    ]
    assert len(matches) == 1, (
        f"Could not uniquely locate a {topology_type} with signature {target}. "
        f"Found {len(matches)} matches."
    )
    return matches[0]


@lru_cache(maxsize=1)
def _query_model() -> Dict[str, object]:
    """Create and index the deterministic query test model."""
    origin = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    assert Topology.IsInstance(origin, "Vertex")

    host = CellComplex.Octahedron(
        origin=origin,
        radius=1.0,
        direction=[0.0, 0.0, 1.0],
        placement="center",
        tolerance=TOLERANCE,
    )
    assert Topology.IsInstance(host, "CellComplex")

    vertices = Topology.Vertices(host, silent=True)
    edges = Topology.Edges(host, silent=True)
    faces = Topology.Faces(host, silent=True)
    shells = Topology.Shells(host, silent=True)
    cells = Topology.Cells(host, silent=True)

    _assert_topology_list(vertices, 6, "vertex", "Fixture vertices")
    _assert_topology_list(edges, 12, "edge", "Fixture edges")
    _assert_topology_list(faces, 9, "face", "Fixture faces")
    _assert_topology_list(shells, 2, "shell", "Fixture shells")
    _assert_topology_list(cells, 2, "cell", "Fixture cells")

    west = _find_by_signature(vertices, [WEST], "vertex")
    south = _find_by_signature(vertices, [SOUTH], "vertex")
    east = _find_by_signature(vertices, [EAST], "vertex")
    north = _find_by_signature(vertices, [NORTH], "vertex")
    top = _find_by_signature(vertices, [TOP], "vertex")
    bottom = _find_by_signature(vertices, [BOTTOM], "vertex")

    edge_west_south = _find_by_signature(
        edges,
        [WEST, SOUTH],
        "edge",
    )
    edge_west_north = _find_by_signature(
        edges,
        [WEST, NORTH],
        "edge",
    )

    shared_face = _find_by_signature(
        faces,
        [WEST, SOUTH, EAST, NORTH],
        "face",
    )
    top_side_face = _find_by_signature(
        faces,
        [TOP, WEST, SOUTH],
        "face",
    )

    top_cells = [cell for cell in cells if _centroid_z(cell) > TOLERANCE]
    bottom_cells = [cell for cell in cells if _centroid_z(cell) < -TOLERANCE]
    assert len(top_cells) == 1
    assert len(bottom_cells) == 1
    top_cell = top_cells[0]
    bottom_cell = bottom_cells[0]

    top_shells = Topology.Shells(top_cell, silent=True)
    bottom_shells = Topology.Shells(bottom_cell, silent=True)
    _assert_topology_list(top_shells, 1, "shell", "Top-cell shells")
    _assert_topology_list(bottom_shells, 1, "shell", "Bottom-cell shells")
    top_shell = top_shells[0]
    bottom_shell = bottom_shells[0]

    shared_wires = Topology.Wires(shared_face, silent=True)
    top_side_wires = Topology.Wires(top_side_face, silent=True)
    _assert_topology_list(shared_wires, 1, "wire", "Shared-face wires")
    _assert_topology_list(top_side_wires, 1, "wire", "Top-side-face wires")
    shared_wire = shared_wires[0]
    top_side_wire = top_side_wires[0]

    return {
        "cellcomplex": host,
        "vertex": west,
        "edge": edge_west_south,
        "wire": shared_wire,
        "face": shared_face,
        "shell": top_shell,
        "cell": top_cell,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "top": top,
        "bottom": bottom,
        "edge_west_south": edge_west_south,
        "edge_west_north": edge_west_north,
        "shared_wire": shared_wire,
        "top_side_wire": top_side_wire,
        "shared_face": shared_face,
        "top_side_face": top_side_face,
        "top_shell": top_shell,
        "bottom_shell": bottom_shell,
        "top_cell": top_cell,
        "bottom_cell": bottom_cell,
        "host": host,
    }


# ---------------------------------------------------------------------------
# Subtopology queries
# ---------------------------------------------------------------------------

SUBTOPOLOGY_CASES = (
    # Vertex
    ("vertex", "vertex", 1),

    # Edge
    ("edge", "vertex", 2),
    ("edge", "edge", 1),

    # Wire
    ("wire", "vertex", 4),
    ("wire", "edge", 4),
    ("wire", "wire", 1),

    # Face
    ("face", "vertex", 4),
    ("face", "edge", 4),
    ("face", "wire", 1),
    ("face", "face", 1),

    # Shell
    ("shell", "vertex", 5),
    ("shell", "edge", 8),
    ("shell", "wire", 5),
    ("shell", "face", 5),
    ("shell", "shell", 1),

    # Cell
    ("cell", "vertex", 5),
    ("cell", "edge", 8),
    ("cell", "wire", 5),
    ("cell", "face", 5),
    ("cell", "shell", 1),
    ("cell", "cell", 1),

    # CellComplex
    ("cellcomplex", "vertex", 6),
    ("cellcomplex", "edge", 12),
    ("cellcomplex", "wire", 9),
    ("cellcomplex", "face", 9),
    ("cellcomplex", "shell", 2),
    ("cellcomplex", "cell", 2),
    ("cellcomplex", "cellcomplex", 1),
)


@pytest.mark.parametrize(
    "source_name, requested_type, expected_count",
    SUBTOPOLOGY_CASES,
)
def test_subtopologies(source_name, requested_type, expected_count):
    """Test the number and type returned by Topology.SubTopologies."""
    model = _query_model()
    source = model[source_name]

    result = Topology.SubTopologies(
        source,
        subTopologyType=requested_type,
        silent=True,
    )

    _assert_topology_list(
        result,
        expected_count,
        requested_type,
        (
            "Topology.SubTopologies"
            f"({source_name}, {requested_type})"
        ),
    )


# ---------------------------------------------------------------------------
# Supertopology queries
# ---------------------------------------------------------------------------

SUPERTOPOLOGY_CASES = (
    # Explicit supertopology types from an equatorial vertex.
    ("vertex", "edge", 4, "edge"),
    ("vertex", "wire", 5, "wire"),
    ("vertex", "face", 5, "face"),
    ("vertex", "shell", 2, "shell"),
    ("vertex", "cell", 2, "cell"),
    ("vertex", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from an equatorial edge.
    ("edge", "wire", 3, "wire"),
    ("edge", "face", 3, "face"),
    ("edge", "shell", 2, "shell"),
    ("edge", "cell", 2, "cell"),
    ("edge", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from the shared equatorial wire.
    ("wire", "face", 1, "face"),
    ("wire", "shell", 2, "shell"),
    ("wire", "cell", 2, "cell"),
    ("wire", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from the shared equatorial face.
    ("face", "shell", 2, "shell"),
    ("face", "cell", 2, "cell"),
    ("face", "cellcomplex", 1, "cellcomplex"),

    # Explicit supertopology types from one constituent shell and cell.
    ("shell", "cell", 1, "cell"),
    ("shell", "cellcomplex", 1, "cellcomplex"),
    ("cell", "cellcomplex", 1, "cellcomplex"),

    # Inferred immediate supertopology type.
    ("vertex", None, 4, "edge"),
    ("edge", None, 3, "wire"),
    ("wire", None, 1, "face"),
    ("face", None, 2, "shell"),
    ("shell", None, 1, "cell"),
    ("cell", None, 1, "cellcomplex"),
)


@pytest.mark.parametrize(
    "source_name, requested_type, expected_count, expected_type",
    SUPERTOPOLOGY_CASES,
)
def test_supertopologies(
    source_name,
    requested_type,
    expected_count,
    expected_type,
):
    """Test the number and type returned by Topology.SuperTopologies."""
    model = _query_model()
    source = model[source_name]
    host = model["host"]

    result = Topology.SuperTopologies(
        source,
        hostTopology=host,
        topologyType=requested_type,
        silent=True,
    )

    requested_label = requested_type if requested_type is not None else "inferred"
    _assert_topology_list(
        result,
        expected_count,
        expected_type,
        (
            "Topology.SuperTopologies"
            f"({source_name}, {requested_label})"
        ),
    )


# ---------------------------------------------------------------------------
# Adjacency queries
# ---------------------------------------------------------------------------

ADJACENCY_CASES = (
    # Same-dimensional adjacency follows the expected boundary relation.
    ("vertex", "vertex", 4),
    ("edge", "edge", 6),
    ("wire", "wire", 1),
    ("face", "face", 8),
    ("shell", "shell", 1),
    ("cell", "cell", 1),
)


@pytest.mark.parametrize(
    "source_name, requested_type, expected_count",
    ADJACENCY_CASES,
)
def test_adjacent_topologies(
    source_name,
    requested_type,
    expected_count,
):
    """Test the number and type returned by Topology.AdjacentTopologies."""
    model = _query_model()
    source = model[source_name]
    host = model["host"]

    result = Topology.AdjacentTopologies(
        source,
        hostTopology=host,
        topologyType=requested_type,
        silent=True,
    )

    _assert_topology_list(
        result,
        expected_count,
        requested_type,
        (
            "Topology.AdjacentTopologies"
            f"({source_name}, {requested_type})"
        ),
    )


# ---------------------------------------------------------------------------
# Shared-topology queries
# ---------------------------------------------------------------------------

SHARED_CASES = (
    # Two equatorial edges meeting at the west vertex.
    (
        "edge_west_south",
        "edge_west_north",
        {"vertices": 1, "edges": 0, "wires": 0, "faces": 0},
    ),

    # The shared equatorial face and one triangular side face.
    (
        "shared_face",
        "top_side_face",
        {"vertices": 2, "edges": 1, "wires": 0, "faces": 0},
    ),

    # Their external wires share the same equatorial edge.
    (
        "shared_wire",
        "top_side_wire",
        {"vertices": 2, "edges": 1, "wires": 0, "faces": 0},
    ),

    # The two cell shells share the complete equatorial face.
    (
        "top_shell",
        "bottom_shell",
        {"vertices": 4, "edges": 4, "wires": 1, "faces": 1},
    ),

    # The two cells share the complete equatorial face.
    (
        "top_cell",
        "bottom_cell",
        {"vertices": 4, "edges": 4, "wires": 1, "faces": 1},
    ),

    # A constituent cell shares all of its boundary topology with its host.
    (
        "host",
        "top_cell",
        {"vertices": 5, "edges": 8, "wires": 5, "faces": 5},
    ),
)


@pytest.mark.parametrize(
    "topology_a_name, topology_b_name, expected_counts",
    SHARED_CASES,
)
def test_shared_topologies(
    topology_a_name,
    topology_b_name,
    expected_counts,
):
    """Test Topology.SharedTopologies and its four typed convenience methods."""
    model = _query_model()
    topology_a = model[topology_a_name]
    topology_b = model[topology_b_name]

    result = Topology.SharedTopologies(
        topology_a,
        topology_b,
        silent=True,
    )

    assert isinstance(result, dict), (
        "Topology.SharedTopologies did not return a dictionary. "
        f"Returned Python type: {type(result).__name__}."
    )
    assert set(result.keys()) == set(SHARED_KEYS), (
        "Topology.SharedTopologies returned unexpected dictionary keys. "
        f"Expected {SHARED_KEYS}; returned {tuple(result.keys())}."
    )

    for key in SHARED_KEYS:
        _assert_topology_list(
            result[key],
            expected_counts[key],
            SHARED_TYPES[key],
            (
                "Topology.SharedTopologies"
                f"({topology_a_name}, {topology_b_name})[{key!r}]"
            ),
        )

    typed_methods = {
        "vertices": Topology.SharedVertices,
        "edges": Topology.SharedEdges,
        "wires": Topology.SharedWires,
        "faces": Topology.SharedFaces,
    }

    for key, method in typed_methods.items():
        typed_result = method(
            topology_a,
            topology_b,
            silent=True,
        )
        _assert_topology_list(
            typed_result,
            expected_counts[key],
            SHARED_TYPES[key],
            (
                f"{method.__qualname__}"
                f"({topology_a_name}, {topology_b_name})"
            ),
        )
