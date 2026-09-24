import numpy as np
import pytest

from src.dynamics import (
    build_damping_matrix,
    build_mass_matrix,
    build_stiffness_matrix,
    state_derivative,
)


def test_mass_matrix():
    masses = [1.0, 2.0, 3.0]

    expected = np.diag([1.0, 2.0, 3.0])

    actual = build_mass_matrix(masses)

    np.testing.assert_allclose(actual, expected)


def test_stiffness_matrix():
    stiffnesses = [1.0, 2.0, 3.0]

    expected = np.array(
        [
            [3.0, -2.0, 0.0],
            [-2.0, 5.0, -3.0],
            [0.0, -3.0, 3.0],
        ]
    )

    actual = build_stiffness_matrix(stiffnesses)

    np.testing.assert_allclose(actual, expected)


def test_damping_matrix():
    dampings = [1.0, 2.0, 3.0]

    expected = np.array(
        [
            [3.0, -2.0, 0.0],
            [-2.0, 5.0, -3.0],
            [0.0, -3.0, 3.0],
        ]
    )

    actual = build_damping_matrix(dampings)

    np.testing.assert_allclose(actual, expected)


def test_matrices_are_symmetric():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    stiffness = build_stiffness_matrix([1.0, 2.0, 3.0])
    damping = build_damping_matrix([1.0, 2.0, 3.0])

    assert np.allclose(mass, mass.T)
    assert np.allclose(stiffness, stiffness.T)
    assert np.allclose(damping, damping.T)


def test_mass_matrix_is_positive_definite():
    mass = build_mass_matrix([1.0, 2.0, 3.0])

    eigenvalues = np.linalg.eigvalsh(mass)

    assert np.all(eigenvalues > 0.0)


def test_state_derivative_zero_state():
    mass = build_mass_matrix([1.0, 2.0, 3.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([1.0, 1.0, 1.0])

    state = np.zeros(6)

    derivative = state_derivative(
        state,
        mass,
        damping,
        stiffness,
    )

    np.testing.assert_allclose(derivative, np.zeros(6))


def test_state_derivative_known_displacement():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([1.0, 1.0, 1.0])

    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    derivative = state_derivative(
        state,
        mass,
        damping,
        stiffness,
    )

    expected = np.array(
        [0.0, 0.0, 0.0, -2.0, 1.0, 0.0]
    )

    np.testing.assert_allclose(derivative, expected)


@pytest.mark.parametrize(
    "builder, values",
    [
        (build_mass_matrix, [1.0, -1.0, 1.0]),
        (build_stiffness_matrix, [1.0, 0.0, 1.0]),
        (build_damping_matrix, [1.0, -1.0, 1.0]),
    ],
)
def test_invalid_parameters(builder, values):
    with pytest.raises(ValueError):
        builder(values)


@pytest.mark.parametrize(
    "builder, values",
    [
        (build_mass_matrix, [1.0, np.nan, 1.0]),
        (build_mass_matrix, [1.0, np.inf, 1.0]),
        (build_stiffness_matrix, [1.0, np.nan, 1.0]),
        (build_damping_matrix, [1.0, np.inf, 1.0]),
    ],
)
def test_nonfinite_parameters(builder, values):
    with pytest.raises(ValueError):
        builder(values)


def test_invalid_state_shape():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([1.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        state_derivative(
            np.zeros(5),
            mass,
            damping,
            stiffness,
        )


def test_nonzero_force_requires_input_matrix():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([1.0, 1.0, 1.0])

    state = np.zeros(6)

    with pytest.raises(ValueError):
        state_derivative(
            state,
            mass,
            damping,
            stiffness,
            force_input=[1.0, 0.0, 0.0],
        )


def test_control_input():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([1.0, 1.0, 1.0])

    state = np.zeros(6)

    input_matrix = np.eye(3)
    force = np.array([1.0, 0.0, 0.0])

    derivative = state_derivative(
        state,
        mass,
        damping,
        stiffness,
        force_input=force,
        input_matrix=input_matrix,
    )

    expected = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(derivative, expected)