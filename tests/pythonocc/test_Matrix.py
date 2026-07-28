# Auto-generated PythonOCC backend parity test
# Copied from test_Matrix.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Matrix."""

from __future__ import annotations

import math

import pytest

from topologicpy.Matrix import Matrix


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _assert_matrix_close(actual, expected, tol=1e-7):
    assert actual is not None
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, abs=tol)


def _apply_4x4(matrix, vector):
    x, y, z = vector
    v = [x, y, z, 1.0]
    return [sum(matrix[i][j] * v[j] for j in range(4)) for i in range(3)]


def _apply_direction(matrix, vector):
    x, y, z = vector
    v = [x, y, z, 0.0]
    return [sum(matrix[i][j] * v[j] for j in range(4)) for i in range(3)]


def _det3_from_4x4(matrix):
    a = [row[:3] for row in matrix[:3]]
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def test_add_and_subtract_support_rectangular_matrices_and_validate_shapes():
    mat_a = [[1, 2, 3], [4, 5, 6]]
    mat_b = [[10, 20, 30], [40, 50, 60]]

    assert Matrix.Add(mat_a, mat_b) == [[11, 22, 33], [44, 55, 66]]
    assert Matrix.Subtract(mat_b, mat_a) == [[9, 18, 27], [36, 45, 54]]

    assert Matrix.Add(mat_a, [[1, 2], [3, 4]]) is None
    assert Matrix.Subtract(mat_a, [[1, 2], [3, 4]]) is None
    assert Matrix.Add("not a matrix", mat_b) is None
    assert Matrix.Subtract(mat_a, [[1, 2, 3], [4]]) is None


def test_identity_scaling_and_translation_matrices():
    assert Matrix.Identity() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    assert Matrix.ByScaling(2, 3, 4) == [[2.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 4.0, 0], [0, 0, 0, 1]]
    assert Matrix.ByScaling("bad", 1, 1) is None

    translation = Matrix.ByTranslation(5, -2, 7)
    assert translation == [[1, 0, 0, 5.0], [0, 1, 0, -2.0], [0, 0, 1, 7.0], [0, 0, 0, 1]]
    assert _apply_4x4(translation, [1, 2, 3]) == pytest.approx([6, 0, 10])
    assert Matrix.ByTranslation(0, object(), 0) is None


def test_multiply_rectangular_matrices_and_rejects_incompatible_inputs():
    assert Matrix.Multiply([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]]

    with pytest.raises(ValueError):
        Matrix.Multiply([[1, 2]], [[1, 2]])

    with pytest.raises(ValueError):
        Matrix.Multiply([[1, 2], [3]], [[1], [2]])


def test_transpose_handles_rectangular_and_invalid_matrices():
    assert Matrix.Transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert Matrix.Transpose([[1, 2], [3]]) is None
    assert Matrix.Transpose("bad") is None


def test_by_rotation_orders_and_invalid_inputs():
    rot_z = Matrix.ByRotation(angleZ=90, order="xyz")
    assert _apply_direction(rot_z, [1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-7)
    assert _apply_direction(rot_z, [0, 1, 0]) == pytest.approx([-1, 0, 0], abs=1e-7)

    for order in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]:
        m = Matrix.ByRotation(10, 20, 30, order=order)
        assert Matrix._MatrixShape(m) == (4, 4)
        assert _det3_from_4x4(m) == pytest.approx(1.0, abs=1e-7)

    assert Matrix.ByRotation(0, 0, 0, order="bad") is None
    assert Matrix.ByRotation("bad", 0, 0) is None


def test_by_coordinate_systems_maps_source_to_target_and_validates_input():
    source = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    target = [[10, 20, 30], [0, 1, 0], [-1, 0, 0], [0, 0, 1]]

    matrix = Matrix.ByCoordinateSystems(source, target, mantissa=6, silent=True)
    assert _apply_4x4(matrix, [0, 0, 0]) == pytest.approx([10, 20, 30], abs=1e-7)
    assert _apply_4x4(matrix, [1, 0, 0]) == pytest.approx([10, 21, 30], abs=1e-7)
    assert _apply_4x4(matrix, [0, 1, 0]) == pytest.approx([9, 20, 30], abs=1e-7)

    assert Matrix.ByCoordinateSystems("bad", target, silent=True) is None
    assert Matrix.ByCoordinateSystems(source[:3], target, silent=True) is None
    singular = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1]]
    assert Matrix.ByCoordinateSystems(singular, target, silent=True) is None


def test_by_vectors_aligns_direction_and_orientation_with_right_handed_rotation():
    matrix = Matrix.ByVectors([1, 0, 0], [0, 1, 0], orientationA=[0, 1, 0], orientationB=[-1, 0, 0])
    assert _apply_direction(matrix, [1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-7)
    assert _apply_direction(matrix, [0, 1, 0]) == pytest.approx([-1, 0, 0], abs=1e-7)
    assert _det3_from_4x4(matrix) == pytest.approx(1.0, abs=1e-7)


def test_by_vectors_handles_antiparallel_vectors_without_reflection():
    matrix = Matrix.ByVectors([1, 0, 0], [-1, 0, 0], orientationA=[0, 1, 0], orientationB=[0, 1, 0])
    assert matrix is not None
    assert _apply_direction(matrix, [1, 0, 0]) == pytest.approx([-1, 0, 0], abs=1e-7)
    assert _apply_direction(matrix, [0, 1, 0]) == pytest.approx([0, 1, 0], abs=1e-7)
    assert _det3_from_4x4(matrix) == pytest.approx(1.0, abs=1e-7)

    assert Matrix.ByVectors_old([1, 0, 0], [-1, 0, 0], orientationA=[0, 1, 0], orientationB=[0, 1, 0]) == matrix
    assert Matrix.ByVectors([0, 0, 0], [1, 0, 0]) is None
    assert Matrix.ByVectors([1, 0, 0], [0, 0, 0]) is None


def test_eigenvalues_and_vectors_are_sorted_and_paired_for_symmetric_matrices():
    values, vectors = Matrix.EigenvaluesAndVectors([[3, 0], [0, 2]], mantissa=6, silent=True)
    assert values == [2.0, 3.0]

    # The first eigenvector corresponds to eigenvalue 2, i.e. the Y basis vector,
    # and the second corresponds to eigenvalue 3, i.e. the X basis vector. Signs are arbitrary.
    assert [abs(x) for x in vectors[0]] == pytest.approx([0, 1], abs=1e-7)
    assert [abs(x) for x in vectors[1]] == pytest.approx([1, 0], abs=1e-7)

    assert Matrix.EigenvaluesAndVectors([[1, 2, 3], [4, 5, 6]], silent=True) is None
    assert Matrix.EigenvaluesAndVectors([[1, 2], [3, 4]], silent=True) is None
    assert Matrix.EigenvaluesAndVectors([["bad"]], silent=True) is None


def test_invert_returns_inverse_for_4x4_and_rejects_invalid_or_singular_matrices():
    transform = Matrix.Multiply(Matrix.ByTranslation(3, 4, 5), Matrix.ByScaling(2, 2, 2))
    inverse = Matrix.Invert(transform, silent=True)
    product = Matrix.Multiply(transform, inverse)
    _assert_matrix_close(product, Matrix.Identity(), tol=1e-7)

    assert Matrix.Invert([[1, 2], [3, 4]], silent=True) is None
    assert Matrix.Invert([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], silent=True) is None
    assert Matrix.Invert([["bad", 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], silent=True) is None


def test_private_shape_and_rounding_helpers():
    assert Matrix._MatrixShape([[1, 2], [3, 4]]) == (2, 2)
    assert Matrix._MatrixShape([[1, 2], [3]]) is None
    assert Matrix._MatrixShape([]) is None
    assert Matrix._IsNumericMatrix([[1, "2"], [3.0, 4]]) is True
    assert Matrix._IsNumericMatrix([[1, "two"]]) is False
    assert Matrix._RoundValue(1e-12, mantissa=6) == 0.0
    assert Matrix._RoundMatrix([[1.23456789]], mantissa=3) == [[1.235]]
