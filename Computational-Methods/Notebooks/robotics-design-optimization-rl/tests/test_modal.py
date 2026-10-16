
import numpy as np
import pytest

from src.dynamics import (
    build_inertia_matrix,
    build_torsional_stiffness_matrix,
)
from src.modal import angular_to_hz, solve_modes


@pytest.fixture
def system_matrices():
    inertia = build_inertia_matrix([1.0, 2.0, 3.0])
    stiffness = build_torsional_stiffness_matrix(
        [10.0, 20.0, 30.0]
    )
    return inertia, stiffness


def test_natural_frequencies_have_expected_shape(system_matrices):
    inertia, stiffness = system_matrices

    frequencies, _ = solve_modes(inertia, stiffness)

    assert frequencies.shape == (3,)


def test_mode_shapes_have_expected_shape(system_matrices):
    inertia, stiffness = system_matrices

    _, mode_shapes = solve_modes(inertia, stiffness)

    assert mode_shapes.shape == (3, 3)


def test_natural_frequencies_are_positive_and_sorted(system_matrices):
    inertia, stiffness = system_matrices

    frequencies, _ = solve_modes(inertia, stiffness)

    assert np.all(frequencies > 0.0)
    assert np.all(np.diff(frequencies) > 0.0)


def test_generalized_eigenvalue_equation(system_matrices):
    inertia, stiffness = system_matrices

    frequencies, mode_shapes = solve_modes(inertia, stiffness)
    eigenvalues = frequencies**2

    lhs = stiffness @ mode_shapes
    rhs = inertia @ mode_shapes @ np.diag(eigenvalues)

    assert np.allclose(lhs, rhs, rtol=1e-9, atol=1e-10)


def test_mode_shapes_are_inertia_normalized(system_matrices):
    inertia, stiffness = system_matrices

    _, mode_shapes = solve_modes(inertia, stiffness)

    normalized = mode_shapes.T @ inertia @ mode_shapes

    assert np.allclose(normalized, np.eye(3), atol=1e-10)


def test_modes_are_stiffness_orthogonal(system_matrices):
    inertia, stiffness = system_matrices

    _, mode_shapes = solve_modes(inertia, stiffness)

    transformed_stiffness = mode_shapes.T @ stiffness @ mode_shapes

    assert np.allclose(
        transformed_stiffness,
        np.diag(np.diag(transformed_stiffness)),
        atol=1e-9,
    )


def test_angular_to_hz_conversion():
    angular_frequencies = np.array(
        [2.0 * np.pi, 4.0 * np.pi, 6.0 * np.pi]
    )

    frequencies_hz = angular_to_hz(angular_frequencies)

    assert np.allclose(frequencies_hz, [1.0, 2.0, 3.0])


def test_angular_to_hz_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        angular_to_hz([[1.0, 2.0], [3.0, 4.0]])


def test_angular_to_hz_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        angular_to_hz([1.0, np.nan])


def test_angular_to_hz_rejects_negative_frequencies():
    with pytest.raises(ValueError, match="nonnegative"):
        angular_to_hz([1.0, -2.0])


def test_invalid_inertia_matrix_shape(system_matrices):
    _, stiffness = system_matrices

    with pytest.raises(ValueError, match="inertia_matrix"):
        solve_modes(np.eye(2), stiffness)


def test_invalid_stiffness_matrix_shape(system_matrices):
    inertia, _ = system_matrices

    with pytest.raises(ValueError, match="torsional_stiffness_matrix"):
        solve_modes(inertia, np.eye(2))


def test_nonfinite_inertia_matrix_is_rejected(system_matrices):
    inertia, stiffness = system_matrices
    invalid_inertia = inertia.copy()
    invalid_inertia[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        solve_modes(invalid_inertia, stiffness)


def test_nonfinite_stiffness_matrix_is_rejected(system_matrices):
    inertia, stiffness = system_matrices
    invalid_stiffness = stiffness.copy()
    invalid_stiffness[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        solve_modes(inertia, invalid_stiffness)


def test_nonsymmetric_inertia_matrix_is_rejected(system_matrices):
    inertia, stiffness = system_matrices
    invalid_inertia = inertia.copy()
    invalid_inertia[0, 1] = 0.5

    with pytest.raises(ValueError, match="symmetric"):
        solve_modes(invalid_inertia, stiffness)


def test_nonsymmetric_stiffness_matrix_is_rejected(system_matrices):
    inertia, stiffness = system_matrices
    invalid_stiffness = stiffness.copy()
    invalid_stiffness[0, 1] += 1.0

    with pytest.raises(ValueError, match="symmetric"):
        solve_modes(inertia, invalid_stiffness)


def test_non_positive_definite_inertia_is_rejected(system_matrices):
    _, stiffness = system_matrices
    invalid_inertia = np.diag([1.0, 2.0, 0.0])

    with pytest.raises(ValueError, match="positive definite"):
        solve_modes(invalid_inertia, stiffness)


def test_non_positive_definite_stiffness_is_rejected(system_matrices):
    inertia, _ = system_matrices
    invalid_stiffness = np.diag([10.0, 20.0, 0.0])

    with pytest.raises(ValueError, match="positive definite"):
        solve_modes(inertia, invalid_stiffness)
