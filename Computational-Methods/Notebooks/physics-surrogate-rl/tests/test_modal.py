import numpy as np
import pytest

from src.dynamics import (
    build_mass_matrix,
    build_stiffness_matrix,
)
from src.modal import (
    angular_to_hz,
    solve_modes,
)


def test_solve_modes_returns_correct_shapes():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    natural_frequencies, mode_shapes = solve_modes(
        mass,
        stiffness,
    )

    assert natural_frequencies.shape == (3,)
    assert mode_shapes.shape == (3, 3)


def test_natural_frequencies_are_sorted_and_positive():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    natural_frequencies, _ = solve_modes(
        mass,
        stiffness,
    )

    assert np.all(natural_frequencies > 0.0)
    assert np.all(np.diff(natural_frequencies) > 0.0)


def test_generalized_eigenvalue_equation():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    natural_frequencies, mode_shapes = solve_modes(
        mass,
        stiffness,
    )

    for i in range(3):
        mode = mode_shapes[:, i]
        omega_squared = natural_frequencies[i] ** 2

        np.testing.assert_allclose(
            stiffness @ mode,
            omega_squared * (mass @ mode),
            rtol=1e-10,
            atol=1e-10,
        )


def test_modes_are_mass_normalized():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    _, mode_shapes = solve_modes(
        mass,
        stiffness,
    )

    modal_mass = mode_shapes.T @ mass @ mode_shapes

    np.testing.assert_allclose(
        modal_mass,
        np.eye(3),
        rtol=1e-10,
        atol=1e-10,
    )


def test_modes_are_orthogonal_with_respect_to_stiffness():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    natural_frequencies, mode_shapes = solve_modes(
        mass,
        stiffness,
    )

    modal_stiffness = mode_shapes.T @ stiffness @ mode_shapes
    expected = np.diag(natural_frequencies ** 2)

    np.testing.assert_allclose(
        modal_stiffness,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_angular_to_hz():
    natural_frequencies = np.array(
        [2.0 * np.pi, 4.0 * np.pi, 6.0 * np.pi]
    )

    frequencies_hz = angular_to_hz(natural_frequencies)

    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        frequencies_hz,
        expected,
    )


def test_invalid_mass_matrix_shape():
    mass = np.eye(2)
    stiffness = np.eye(3)

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_invalid_stiffness_matrix_shape():
    mass = np.eye(3)
    stiffness = np.eye(2)

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_non_symmetric_mass_matrix():
    mass = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_non_positive_definite_mass_matrix():
    mass = np.diag([1.0, -1.0, 3.0])
    stiffness = build_stiffness_matrix([10.0, 20.0, 30.0])

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_non_symmetric_stiffness_matrix():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = np.array(
        [
            [10.0, -20.0, 0.0],
            [-10.0, 50.0, -30.0],
            [0.0, -30.0, 30.0],
        ]
    )

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_nonpositive_eigenvalues_rejected():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    stiffness = np.zeros((3, 3))

    with pytest.raises(ValueError):
        solve_modes(mass, stiffness)


def test_invalid_frequency_input():
    with pytest.raises(ValueError):
        angular_to_hz(np.array([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError):
        angular_to_hz(np.array([1.0, np.nan, 3.0]))

    with pytest.raises(ValueError):
        angular_to_hz(np.array([1.0, -2.0, 3.0]))