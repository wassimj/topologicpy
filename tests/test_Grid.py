# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

import pytest

from topologicpy.Dictionary import Dictionary
from topologicpy.Face import Face
from topologicpy.Grid import Grid
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire


TOLERANCE = 0.0001


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _edges(topology):
    return Topology.Edges(topology) or []


def _vertices(topology):
    return Topology.Vertices(topology) or []


def _value(topology, key, default=None):
    dictionary = Topology.Dictionary(topology, silent=True)
    if dictionary is None:
        return default
    try:
        return Dictionary.ValueAtKey(dictionary, key, default)
    except TypeError:
        value = Dictionary.ValueAtKey(dictionary, key)
        return default if value is None else value


def _axis_edges(grid, axis):
    result = [edge for edge in _edges(grid) if _value(edge, "grid_axis") == axis]
    return sorted(
        result,
        key=lambda edge: (
            float(_value(edge, "grid_coordinate", 0.0)),
            int(_value(edge, "grid_segment", 0)),
        ),
    )


def _axis_coordinates(grid, axis, unique=True):
    values = [float(_value(edge, "grid_coordinate")) for edge in _axis_edges(grid, axis)]
    if not unique:
        return values
    result = []
    for value in values:
        if not result or abs(value - result[-1]) > TOLERANCE:
            result.append(value)
    return result


def _xyz(vertex):
    return [
        float(Vertex.X(vertex, mantissa=9)),
        float(Vertex.Y(vertex, mantissa=9)),
        float(Vertex.Z(vertex, mantissa=9)),
    ]


def _edge_xyz(edge):
    vertices = _vertices(edge)
    assert len(vertices) == 2
    return _xyz(vertices[0]), _xyz(vertices[1])


def _edge_at_coordinate(grid, axis, coordinate, segment=None):
    candidates = [
        edge
        for edge in _axis_edges(grid, axis)
        if float(_value(edge, "grid_coordinate")) == pytest.approx(coordinate, abs=TOLERANCE)
    ]
    if segment is None:
        assert len(candidates) >= 1
        return candidates[0]
    for edge in candidates:
        if int(_value(edge, "grid_segment", 0)) == segment:
            return edge
    raise AssertionError(
        f"Could not find grid edge axis={axis!r}, coordinate={coordinate}, segment={segment}."
    )


def _assert_common_edge_semantics(edge, axis, role="axis"):
    assert _value(edge, "grid_type") == "orthogonal"
    assert _value(edge, "grid_axis") == axis
    assert _value(edge, "grid_role") == role
    assert isinstance(_value(edge, "grid_index"), int)
    assert _value(edge, "grid_coordinate") is not None
    assert isinstance(_value(edge, "grid_segment"), int)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def test_public_api_contains_only_the_new_grid_concepts():
    expected = [
        "Square",
        "Rectangular",
        "ByDivisions",
        "Structural",
        "OnFace",
        "TileLayout",
        "Vertices",
    ]
    for name in expected:
        assert hasattr(Grid, name)
        assert callable(getattr(Grid, name))

    legacy = [
        "EdgesByDistances",
        "EdgesByParameters",
        "VerticesByDistances",
        "VerticesByDistances_old",
        "VerticesByParameters",
    ]
    for name in legacy:
        assert not hasattr(Grid, name)


# -----------------------------------------------------------------------------
# Pure setting-out rules
# -----------------------------------------------------------------------------


def test_regular_coordinates_centered_without_forced_boundaries():
    coordinates, datum = Grid._RegularCoordinates(
        [-2.5, 2.5],
        spacing=2.0,
        alignment="center",
        includeBoundary=False,
        tolerance=TOLERANCE,
    )

    assert coordinates == pytest.approx([-2.0, 0.0, 2.0], abs=TOLERANCE)
    assert datum == pytest.approx(0.0, abs=TOLERANCE)


def test_regular_coordinates_can_force_outer_boundaries():
    coordinates, datum = Grid._RegularCoordinates(
        [-2.5, 2.5],
        spacing=2.0,
        alignment="center",
        includeBoundary=True,
        tolerance=TOLERANCE,
    )

    assert coordinates == pytest.approx([-2.5, -2.0, 0.0, 2.0, 2.5], abs=TOLERANCE)
    assert datum == pytest.approx(0.0, abs=TOLERANCE)


def test_label_generation_supports_architectural_conventions():
    assert Grid._Labels(5, "letters") == ["A", "B", "C", "D", "E"]
    assert Grid._Labels(4, "numbers") == ["1", "2", "3", "4"]
    assert Grid._Labels(3, "G") == ["G1", "G2", "G3"]
    assert Grid._Labels(4, ["X", "Y"]) == ["X", "Y", None, None]
    assert Grid._AlphaLabel(25) == "Z"
    assert Grid._AlphaLabel(26) == "AA"
    assert Grid._AlphaLabel(27) == "AB"


def test_tile_best_mode_can_choose_tile_centred_layout():
    coordinates, mode, pitch = Grid._TileJointCoordinates(
        [-0.9, 0.9],
        tileSize=0.6,
        grout=0.0,
        centerMode="best",
        tolerance=TOLERANCE,
    )

    assert mode == "tile"
    assert pitch == pytest.approx(0.6, abs=TOLERANCE)
    assert coordinates == pytest.approx([-0.3, 0.3], abs=TOLERANCE)


def test_tile_best_mode_can_choose_joint_centred_layout():
    coordinates, mode, pitch = Grid._TileJointCoordinates(
        [-1.0, 1.0],
        tileSize=0.6,
        grout=0.0,
        centerMode="best",
        tolerance=TOLERANCE,
    )

    assert mode == "joint"
    assert pitch == pytest.approx(0.6, abs=TOLERANCE)
    assert coordinates == pytest.approx([-0.6, 0.0, 0.6], abs=TOLERANCE)


# -----------------------------------------------------------------------------
# Square
# -----------------------------------------------------------------------------


def test_square_creates_expected_regular_grid():
    grid = Grid.Square(
        size=4.0,
        spacing=1.0,
        uLabels="letters",
        vLabels="numbers",
        silent=True,
    )

    assert Topology.IsInstance(grid, "Cluster")
    assert len(_edges(grid)) == 10
    assert len(_axis_edges(grid, "u")) == 5
    assert len(_axis_edges(grid, "v")) == 5
    assert _axis_coordinates(grid, "u") == pytest.approx([-2, -1, 0, 1, 2], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-2, -1, 0, 1, 2], abs=TOLERANCE)

    u_edges = _axis_edges(grid, "u")
    v_edges = _axis_edges(grid, "v")
    assert [_value(edge, "grid_label") for edge in u_edges] == ["A", "B", "C", "D", "E"]
    assert [_value(edge, "grid_label") for edge in v_edges] == ["1", "2", "3", "4", "5"]

    for edge in u_edges:
        _assert_common_edge_semantics(edge, "u")
        assert float(_value(edge, "grid_spacing")) == pytest.approx(1.0, abs=TOLERANCE)
        assert _value(edge, "grid_alignment") == "center"

    for edge in v_edges:
        _assert_common_edge_semantics(edge, "v")
        assert float(_value(edge, "grid_spacing")) == pytest.approx(1.0, abs=TOLERANCE)
        assert _value(edge, "grid_alignment") == "center"

    assert bool(_value(u_edges[0], "grid_is_boundary")) is True
    assert bool(_value(u_edges[-1], "grid_is_boundary")) is True
    assert bool(_value(_edge_at_coordinate(grid, "u", 0.0), "grid_is_datum")) is True
    assert bool(_value(_edge_at_coordinate(grid, "v", 0.0), "grid_is_datum")) is True


def test_square_include_boundary_controls_non_modular_edge_lines():
    without_boundaries = Grid.Square(
        size=5.0,
        spacing=2.0,
        includeBoundary=False,
        silent=True,
    )
    with_boundaries = Grid.Square(
        size=5.0,
        spacing=2.0,
        includeBoundary=True,
        silent=True,
    )

    assert _axis_coordinates(without_boundaries, "u") == pytest.approx([-2, 0, 2], abs=TOLERANCE)
    assert _axis_coordinates(without_boundaries, "v") == pytest.approx([-2, 0, 2], abs=TOLERANCE)
    assert _axis_coordinates(with_boundaries, "u") == pytest.approx([-2.5, -2, 0, 2, 2.5], abs=TOLERANCE)
    assert _axis_coordinates(with_boundaries, "v") == pytest.approx([-2.5, -2, 0, 2, 2.5], abs=TOLERANCE)


# -----------------------------------------------------------------------------
# Rectangular
# -----------------------------------------------------------------------------


def test_rectangular_supports_independent_spacing_and_lowerleft_placement():
    origin = Vertex.ByCoordinates(10, 20, 3)
    grid = Grid.Rectangular(
        origin=origin,
        width=6.0,
        length=4.0,
        uSpacing=2.0,
        vSpacing=1.0,
        placement="lowerleft",
        uAlignment="start",
        vAlignment="start",
        includeBoundary=True,
        silent=True,
    )

    assert Topology.IsInstance(grid, "Cluster")
    assert _axis_coordinates(grid, "u") == pytest.approx([0, 2, 4, 6], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([0, 1, 2, 3, 4], abs=TOLERANCE)
    assert len(_edges(grid)) == 9

    points = [_xyz(vertex) for vertex in _vertices(grid)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]

    assert min(xs) == pytest.approx(10.0, abs=TOLERANCE)
    assert max(xs) == pytest.approx(16.0, abs=TOLERANCE)
    assert min(ys) == pytest.approx(20.0, abs=TOLERANCE)
    assert max(ys) == pytest.approx(24.0, abs=TOLERANCE)
    assert min(zs) == pytest.approx(3.0, abs=TOLERANCE)
    assert max(zs) == pytest.approx(3.0, abs=TOLERANCE)

    for edge in _axis_edges(grid, "u"):
        assert _value(edge, "grid_source") == "Rectangular"
        assert float(_value(edge, "grid_spacing")) == pytest.approx(2.0, abs=TOLERANCE)
        assert _value(edge, "grid_alignment") == "start"

    for edge in _axis_edges(grid, "v"):
        assert _value(edge, "grid_source") == "Rectangular"
        assert float(_value(edge, "grid_spacing")) == pytest.approx(1.0, abs=TOLERANCE)
        assert _value(edge, "grid_alignment") == "start"


# -----------------------------------------------------------------------------
# ByDivisions
# -----------------------------------------------------------------------------


def test_by_divisions_treats_divisions_as_bays_not_lines():
    grid = Grid.ByDivisions(
        width=12.0,
        length=8.0,
        uDivisions=6,
        vDivisions=4,
        uLabels="letters",
        vLabels="numbers",
        silent=True,
    )

    u_edges = _axis_edges(grid, "u")
    v_edges = _axis_edges(grid, "v")

    assert len(u_edges) == 7
    assert len(v_edges) == 5
    assert len(_edges(grid)) == 12
    assert _axis_coordinates(grid, "u") == pytest.approx([-6, -4, -2, 0, 2, 4, 6], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-4, -2, 0, 2, 4], abs=TOLERANCE)

    assert [_value(edge, "grid_label") for edge in u_edges] == ["A", "B", "C", "D", "E", "F", "G"]
    assert [_value(edge, "grid_label") for edge in v_edges] == ["1", "2", "3", "4", "5"]

    for edge in u_edges + v_edges:
        assert _value(edge, "grid_source") == "ByDivisions"
        assert int(_value(edge, "grid_u_divisions")) == 6
        assert int(_value(edge, "grid_v_divisions")) == 4
        assert _value(edge, "grid_alignment") == "fit"

    assert float(_value(u_edges[0], "grid_spacing")) == pytest.approx(2.0, abs=TOLERANCE)
    assert float(_value(v_edges[0], "grid_spacing")) == pytest.approx(2.0, abs=TOLERANCE)


# -----------------------------------------------------------------------------
# Structural
# -----------------------------------------------------------------------------


def test_structural_grid_uses_bay_widths_and_architectural_labels():
    u_bays = [6.0, 6.0, 7.5, 6.0]
    v_bays = [8.0, 8.0, 6.0]

    grid = Grid.Structural(
        uBays=u_bays,
        vBays=v_bays,
        extension=1.5,
        silent=True,
    )

    u_edges = _axis_edges(grid, "u")
    v_edges = _axis_edges(grid, "v")

    assert len(u_edges) == 5
    assert len(v_edges) == 4
    assert len(_edges(grid)) == 9

    assert _axis_coordinates(grid, "u") == pytest.approx(
        [-12.75, -6.75, -0.75, 6.75, 12.75], abs=TOLERANCE
    )
    assert _axis_coordinates(grid, "v") == pytest.approx(
        [-11.0, -3.0, 5.0, 11.0], abs=TOLERANCE
    )

    assert [_value(edge, "grid_label") for edge in u_edges] == ["A", "B", "C", "D", "E"]
    assert [_value(edge, "grid_label") for edge in v_edges] == ["1", "2", "3", "4"]

    for edge in u_edges:
        _assert_common_edge_semantics(edge, "u", role="structural_axis")
        assert _value(edge, "grid_source") == "Structural"
        assert float(_value(edge, "grid_width")) == pytest.approx(25.5, abs=TOLERANCE)
        assert float(_value(edge, "grid_length")) == pytest.approx(22.0, abs=TOLERANCE)

    for edge in v_edges:
        _assert_common_edge_semantics(edge, "v", role="structural_axis")
        assert _value(edge, "grid_source") == "Structural"

    assert _value(u_edges[0], "grid_bay_before") is None
    assert float(_value(u_edges[0], "grid_bay_after")) == pytest.approx(6.0, abs=TOLERANCE)
    assert float(_value(u_edges[1], "grid_bay_before")) == pytest.approx(6.0, abs=TOLERANCE)
    assert float(_value(u_edges[1], "grid_bay_after")) == pytest.approx(6.0, abs=TOLERANCE)
    assert float(_value(u_edges[2], "grid_bay_before")) == pytest.approx(6.0, abs=TOLERANCE)
    assert float(_value(u_edges[2], "grid_bay_after")) == pytest.approx(7.5, abs=TOLERANCE)
    assert float(_value(u_edges[-1], "grid_bay_before")) == pytest.approx(6.0, abs=TOLERANCE)
    assert _value(u_edges[-1], "grid_bay_after") is None

    assert _value(v_edges[0], "grid_bay_before") is None
    assert float(_value(v_edges[0], "grid_bay_after")) == pytest.approx(8.0, abs=TOLERANCE)
    assert float(_value(v_edges[2], "grid_bay_before")) == pytest.approx(8.0, abs=TOLERANCE)
    assert float(_value(v_edges[2], "grid_bay_after")) == pytest.approx(6.0, abs=TOLERANCE)
    assert float(_value(v_edges[-1], "grid_bay_before")) == pytest.approx(6.0, abs=TOLERANCE)
    assert _value(v_edges[-1], "grid_bay_after") is None


def test_structural_explicit_labels_override_defaults():
    grid = Grid.Structural(
        uBays=[5, 7],
        vBays=[4, 4],
        uLabels=["X1", "X2", "X3"],
        vLabels=["Y1", "Y2", "Y3"],
        silent=True,
    )

    assert [_value(edge, "grid_label") for edge in _axis_edges(grid, "u")] == ["X1", "X2", "X3"]
    assert [_value(edge, "grid_label") for edge in _axis_edges(grid, "v")] == ["Y1", "Y2", "Y3"]


# -----------------------------------------------------------------------------
# OnFace
# -----------------------------------------------------------------------------


def test_on_face_creates_physical_grid_independent_of_uv_parameters():
    face = Face.Rectangle(
        width=4.5,
        length=3.5,
        direction=[0, 0, 1],
        placement="center",
        silent=True,
    )
    grid = Grid.OnFace(
        face,
        spacing=1.0,
        includeBoundary=False,
        silent=True,
    )

    assert Topology.IsInstance(grid, "Cluster")
    assert _axis_coordinates(grid, "u") == pytest.approx([-2, -1, 0, 1, 2], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-1, 0, 1], abs=TOLERANCE)
    assert len(_edges(grid)) == 8

    for edge in _edges(grid):
        assert _value(edge, "grid_source") == "OnFace"
        assert bool(_value(edge, "grid_clipped")) is True
        _assert_common_edge_semantics(edge, _value(edge, "grid_axis"))

        for vertex in _vertices(edge):
            x, y, z = _xyz(vertex)
            assert -2.25 - TOLERANCE <= x <= 2.25 + TOLERANCE
            assert -1.75 - TOLERANCE <= y <= 1.75 + TOLERANCE
            assert z == pytest.approx(0.0, abs=TOLERANCE)


def test_on_face_vertical_wall_defaults_to_horizontal_and_vertical_architectural_axes():
    wall = Face.Rectangle(
        width=4.5,
        length=3.5,
        direction=[0, -1, 0],
        placement="center",
        silent=True,
    )
    grid = Grid.OnFace(
        wall,
        spacing=1.0,
        includeBoundary=False,
        silent=True,
    )

    u_edge = _edge_at_coordinate(grid, "u", 0.0)
    v_edge = _edge_at_coordinate(grid, "v", 0.0)

    u0, u1 = _edge_xyz(u_edge)
    v0, v1 = _edge_xyz(v_edge)

    # Constant-u lines on a wall should be vertical: constant X/Y, changing Z.
    assert u0[0] == pytest.approx(u1[0], abs=TOLERANCE)
    assert u0[1] == pytest.approx(u1[1], abs=TOLERANCE)
    assert abs(u0[2] - u1[2]) > 3.0

    # Constant-v lines should be horizontal: changing X, constant Y/Z.
    assert abs(v0[0] - v1[0]) > 4.0
    assert v0[1] == pytest.approx(v1[1], abs=TOLERANCE)
    assert v0[2] == pytest.approx(v1[2], abs=TOLERANCE)


def test_on_face_with_opening_preserves_logical_axis_semantics_after_clipping():
    outer = Wire.Rectangle(width=6.0, length=6.0, placement="center", silent=True)
    inner = Wire.Rectangle(width=2.0, length=2.0, placement="center", silent=True)
    face = Face.ByWires(outer, [inner], silent=True)

    assert Topology.IsInstance(face, "Face")

    grid = Grid.OnFace(
        face,
        spacing=2.0,
        includeBoundary=False,
        silent=True,
    )

    assert Topology.IsInstance(grid, "Cluster")

    # Three logical axes exist in each family: -2, 0, +2. The two centre axes
    # cross the opening and are therefore split into two physical edge segments.
    assert _axis_coordinates(grid, "u") == pytest.approx([-2, 0, 2], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-2, 0, 2], abs=TOLERANCE)

    u_zero = [
        edge for edge in _axis_edges(grid, "u")
        if float(_value(edge, "grid_coordinate")) == pytest.approx(0.0, abs=TOLERANCE)
    ]
    v_zero = [
        edge for edge in _axis_edges(grid, "v")
        if float(_value(edge, "grid_coordinate")) == pytest.approx(0.0, abs=TOLERANCE)
    ]

    assert len(u_zero) == 2
    assert len(v_zero) == 2
    assert {_value(edge, "grid_segment") for edge in u_zero} == {0, 1}
    assert {_value(edge, "grid_segment") for edge in v_zero} == {0, 1}

    assert len({_value(edge, "grid_index") for edge in u_zero}) == 1
    assert len({_value(edge, "grid_index") for edge in v_zero}) == 1
    assert all(_value(edge, "grid_source") == "OnFace" for edge in u_zero + v_zero)
    assert all(bool(_value(edge, "grid_clipped")) is True for edge in u_zero + v_zero)


# -----------------------------------------------------------------------------
# TileLayout
# -----------------------------------------------------------------------------


def test_tile_layout_best_selects_tile_centred_solution_when_it_avoids_smaller_edge_cuts():
    face = Face.Rectangle(width=1.8, length=1.8, placement="center", silent=True)
    grid = Grid.TileLayout(
        face,
        tileWidth=0.6,
        tileHeight=0.6,
        groutWidth=0.0,
        groutHeight=0.0,
        uCenterMode="best",
        vCenterMode="best",
        silent=True,
    )

    assert Topology.IsInstance(grid, "Cluster")
    assert _axis_coordinates(grid, "u") == pytest.approx([-0.3, 0.3], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-0.3, 0.3], abs=TOLERANCE)
    assert len(_edges(grid)) == 4

    for edge in _edges(grid):
        _assert_common_edge_semantics(edge, _value(edge, "grid_axis"), role="tile_joint")
        assert _value(edge, "grid_source") == "TileLayout"
        assert _value(edge, "grid_u_center_mode") == "tile"
        assert _value(edge, "grid_v_center_mode") == "tile"
        assert float(_value(edge, "grid_tile_width")) == pytest.approx(0.6, abs=TOLERANCE)
        assert float(_value(edge, "grid_tile_height")) == pytest.approx(0.6, abs=TOLERANCE)
        assert float(_value(edge, "grid_spacing")) == pytest.approx(0.6, abs=TOLERANCE)
        assert bool(_value(edge, "grid_clipped")) is True


def test_tile_layout_best_selects_joint_centred_solution_when_it_avoids_smaller_edge_cuts():
    face = Face.Rectangle(width=2.0, length=2.0, placement="center", silent=True)
    grid = Grid.TileLayout(
        face,
        tileWidth=0.6,
        tileHeight=0.6,
        uCenterMode="best",
        vCenterMode="best",
        silent=True,
    )

    assert _axis_coordinates(grid, "u") == pytest.approx([-0.6, 0.0, 0.6], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-0.6, 0.0, 0.6], abs=TOLERANCE)
    assert len(_edges(grid)) == 6

    for edge in _edges(grid):
        assert _value(edge, "grid_u_center_mode") == "joint"
        assert _value(edge, "grid_v_center_mode") == "joint"

    assert bool(_value(_edge_at_coordinate(grid, "u", 0.0), "grid_is_datum")) is True
    assert bool(_value(_edge_at_coordinate(grid, "v", 0.0), "grid_is_datum")) is True


def test_tile_layout_respects_forced_tile_and_joint_modes():
    face = Face.Rectangle(width=2.0, length=2.0, placement="center", silent=True)
    grid = Grid.TileLayout(
        face,
        tileWidth=0.6,
        tileHeight=0.6,
        uCenterMode="tile",
        vCenterMode="joint",
        silent=True,
    )

    assert _axis_coordinates(grid, "u") == pytest.approx([-0.9, -0.3, 0.3, 0.9], abs=TOLERANCE)
    assert _axis_coordinates(grid, "v") == pytest.approx([-0.6, 0.0, 0.6], abs=TOLERANCE)

    assert all(_value(edge, "grid_u_center_mode") == "tile" for edge in _edges(grid))
    assert all(_value(edge, "grid_v_center_mode") == "joint" for edge in _edges(grid))


def test_tile_layout_spacing_is_tile_plus_grout_pitch():
    face = Face.Rectangle(width=2.5, length=2.5, placement="center", silent=True)
    grid = Grid.TileLayout(
        face,
        tileWidth=0.6,
        tileHeight=0.4,
        groutWidth=0.02,
        groutHeight=0.01,
        uCenterMode="joint",
        vCenterMode="joint",
        silent=True,
    )

    u_edges = _axis_edges(grid, "u")
    v_edges = _axis_edges(grid, "v")
    assert len(u_edges) > 1
    assert len(v_edges) > 1

    assert float(_value(u_edges[0], "grid_spacing")) == pytest.approx(0.62, abs=TOLERANCE)
    assert float(_value(v_edges[0], "grid_spacing")) == pytest.approx(0.41, abs=TOLERANCE)
    assert float(_value(u_edges[0], "grid_grout_width")) == pytest.approx(0.02, abs=TOLERANCE)
    assert float(_value(v_edges[0], "grid_grout_height")) == pytest.approx(0.01, abs=TOLERANCE)


# -----------------------------------------------------------------------------
# Semantic intersection vertices
# -----------------------------------------------------------------------------


def test_vertices_returns_all_square_grid_intersections():
    grid = Grid.Square(
        size=2.0,
        spacing=1.0,
        uLabels="letters",
        vLabels="numbers",
        silent=True,
    )
    intersections = Grid.Vertices(grid, silent=True)

    assert Topology.IsInstance(intersections, "Cluster")
    vertices = _vertices(intersections)
    assert len(vertices) == 9

    uv_pairs = sorted(
        (
            float(_value(vertex, "grid_u_coordinate")),
            float(_value(vertex, "grid_v_coordinate")),
        )
        for vertex in vertices
    )
    expected = sorted((u, v) for u in [-1.0, 0.0, 1.0] for v in [-1.0, 0.0, 1.0])
    assert uv_pairs == expected

    for vertex in vertices:
        assert _value(vertex, "grid_type") == "orthogonal"
        assert _value(vertex, "grid_role") == "intersection"
        assert isinstance(_value(vertex, "grid_u_index"), int)
        assert isinstance(_value(vertex, "grid_v_index"), int)
        assert _value(vertex, "grid_u_coordinate") is not None
        assert _value(vertex, "grid_v_coordinate") is not None


def test_vertices_transfers_axis_labels_to_intersections():
    grid = Grid.ByDivisions(
        width=2.0,
        length=2.0,
        uDivisions=2,
        vDivisions=2,
        uLabels="letters",
        vLabels="numbers",
        silent=True,
    )
    intersections = Grid.Vertices(grid, silent=True)
    vertices = _vertices(intersections)

    centre = [
        vertex
        for vertex in vertices
        if float(_value(vertex, "grid_u_coordinate")) == pytest.approx(0.0, abs=TOLERANCE)
        and float(_value(vertex, "grid_v_coordinate")) == pytest.approx(0.0, abs=TOLERANCE)
    ]

    assert len(centre) == 1
    centre = centre[0]
    assert _value(centre, "grid_u_index") == 1
    assert _value(centre, "grid_v_index") == 1
    assert _value(centre, "grid_u_label") == "B"
    assert _value(centre, "grid_v_label") == "2"


def test_vertices_deduplicates_intersections_of_split_grid_segments():
    outer = Wire.Rectangle(width=6.0, length=6.0, placement="center", silent=True)
    inner = Wire.Rectangle(width=2.0, length=2.0, placement="center", silent=True)
    face = Face.ByWires(outer, [inner], silent=True)
    grid = Grid.OnFace(face, spacing=2.0, includeBoundary=False, silent=True)

    intersections = Grid.Vertices(grid, silent=True)
    assert Topology.IsInstance(intersections, "Cluster")

    vertices = _vertices(intersections)
    xyz = {tuple(round(value, 6) for value in _xyz(vertex)) for vertex in vertices}
    assert len(vertices) == len(xyz)

    # The central point lies in the opening and therefore must not appear.
    assert (0.0, 0.0, 0.0) not in xyz


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        lambda: Grid.Square(size=0.0, silent=True),
        lambda: Grid.Square(spacing=0.0, silent=True),
        lambda: Grid.Rectangular(width=0.0, silent=True),
        lambda: Grid.Rectangular(length=0.0, silent=True),
        lambda: Grid.Rectangular(placement="unsupported", silent=True),
        lambda: Grid.ByDivisions(uDivisions=0, silent=True),
        lambda: Grid.ByDivisions(vDivisions=0, silent=True),
        lambda: Grid.Structural(uBays=[6.0, 0.0, 6.0], silent=True),
        lambda: Grid.Structural(vBays=[], silent=True),
        lambda: Grid.OnFace(None, silent=True),
        lambda: Grid.TileLayout(None, silent=True),
        lambda: Grid.TileLayout(Face.Rectangle(), tileWidth=0.0, silent=True),
        lambda: Grid.TileLayout(Face.Rectangle(), uCenterMode="unsupported", silent=True),
        lambda: Grid.Vertices(Vertex.ByCoordinates(0, 0, 0), silent=True),
    ],
)
def test_invalid_inputs_return_none(result):
    assert result() is None
