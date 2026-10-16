import numpy as np
import pytest

from src.dynamics import (
    build_inertia_matrix,
    build_rotational_damping_matrix,
    build_torsional_stiffness_matrix,
    state_derivative,
)


@pytest.fixture
def system_matrices():
    """Return valid matrices for a 3-DOF rotational system."""
    inertia = build_inertia_matrix([1.0, 1.0, 1.0])
    damping = build_rotational_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_torsional_stiffness_matrix([1.0, 1.0, 1.0])

    return inertia, damping, stiffness


# ---------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------

def test_inertia_matrix():
    actual = build_inertia_matrix([1.0, 2.0, 3.0])
    expected = np.diag([1.0, 2.0, 3.0])

    np.testing.assert_allclose(actual, expected)


def test_torsional_stiffness_matrix():
    actual = build_torsional_stiffness_matrix([1.0, 2.0, 3.0])
    expected = np.array([
        [3.0, -2.0, 0.0],
        [-2.0, 5.0, -3.0],
        [0.0, -3.0, 3.0],
    ])

    np.testing.assert_allclose(actual, expected)


def test_rotational_damping_matrix():
    actual = build_rotational_damping_matrix([1.0, 2.0, 3.0])
    expected = np.array([
        [3.0, -2.0, 0.0],
        [-2.0, 5.0, -3.0],
        [0.0, -3.0, 3.0],
    ])

    np.testing.assert_allclose(actual, expected)


def test_matrices_are_symmetric(system_matrices):
    inertia, damping, stiffness = system_matrices

    assert np.allclose(inertia, inertia.T)
    assert np.allclose(damping, damping.T)
    assert np.allclose(stiffness, stiffness.T)

    for matrix in (inertia, damping, stiffness):
        np.testing.assert_allclose(matrix, matrix.T)


def test_inertia_matrix_is_positive_definite():
    inertia = build_inertia_matrix([1.0, 2.0, 3.0])

    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)


def test_torsional_stiffness_matrix_is_positive_definite():
    stiffness = build_torsional_stiffness_matrix([1.0, 2.0, 3.0])

    assert np.all(np.linalg.eigvalsh(stiffness) > 0.0)


# ---------------------------------------------------------------------
# State derivative and governing equations
# ---------------------------------------------------------------------

def test_state_derivative_zero_state(system_matrices):
    inertia, damping, stiffness = system_matrices

    derivative = state_derivative(
        np.zeros(6),
        inertia,
        damping,
        stiffness,
    )

    np.testing.assert_allclose(derivative, np.zeros(6))


def test_state_derivative_known_angle(system_matrices):
    inertia, damping, stiffness = system_matrices
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    derivative = state_derivative(
        state, inertia, damping, stiffness
    )

    # J = I, q = [1, 0, 0], and q_dot = 0.
    # Therefore q_ddot = -K_theta @ q = [-2, 1, 0].
    expected = np.array([0.0, 0.0, 0.0, -2.0, 1.0, 0.0])

    np.testing.assert_allclose(derivative, expected)


def test_state_derivative_known_angular_velocity(system_matrices):
    inertia, damping, stiffness = system_matrices
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    derivative = state_derivative(
        state, inertia, damping, stiffness
    )

    np.testing.assert_allclose(derivative[:3], [1.0, 0.0, 0.0])

    # With zero angles, q_ddot = -J^{-1} @ D @ q_dot.
    np.testing.assert_allclose(
        derivative[3:],
        [-2.0, 1.0, 0.0],
    )


def test_torque_input(system_matrices):
    inertia, damping, stiffness = system_matrices

    derivative = state_derivative(
        np.zeros(6),
        inertia,
        damping,
        stiffness,
        torque_input=[1.0, 0.0, 0.0],
        input_matrix=np.eye(3),
    )

    expected = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    np.testing.assert_allclose(derivative, expected)


def test_zero_torque_without_input_matrix_is_valid(system_matrices):
    inertia, damping, stiffness = system_matrices

    derivative = state_derivative(
        np.zeros(6),
        inertia,
        damping,
        stiffness,
        torque_input=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(derivative, np.zeros(6))


# ---------------------------------------------------------------------
# Matrix-builder validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "builder, values",
    [
        (build_inertia_matrix, [1.0, 0.0, 1.0]),
        (build_inertia_matrix, [1.0, -1.0, 1.0]),
        (build_torsional_stiffness_matrix, [1.0, 0.0, 1.0]),
        (build_torsional_stiffness_matrix, [1.0, -1.0, 1.0]),
        (build_rotational_damping_matrix, [1.0, -1.0, 1.0]),
    ],
)
def test_invalid_matrix_parameters(builder, values):
    with pytest.raises(ValueError):
        builder(values)


@pytest.mark.parametrize(
    "builder, values",
    [
        (build_inertia_matrix, [1.0, 2.0]),
        (build_inertia_matrix, [1.0, 2.0, 3.0, 4.0]),
        (build_torsional_stiffness_matrix, [1.0, 2.0]),
        (build_rotational_damping_matrix, [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_invalid_parameter_dimensions(builder, values):
    with pytest.raises(ValueError):
        builder(values)


@pytest.mark.parametrize(
    "builder, values",
    [
        (build_inertia_matrix, [1.0, np.nan, 1.0]),
        (build_inertia_matrix, [1.0, np.inf, 1.0]),
        (build_torsional_stiffness_matrix, [1.0, np.nan, 1.0]),
        (build_torsional_stiffness_matrix, [1.0, -np.inf, 1.0]),
        (build_rotational_damping_matrix, [1.0, np.inf, 1.0]),
        (build_rotational_damping_matrix, [1.0, np.nan, 1.0]),
    ],
)
def test_nonfinite_matrix_parameters(builder, values):
    with pytest.raises(ValueError):
        builder(values)


@pytest.mark.parametrize(
    "builder",
    [
        build_inertia_matrix,
        build_torsional_stiffness_matrix,
        build_rotational_damping_matrix,
    ],
)
def test_matrix_builders_accept_numpy_arrays(builder):
    matrix = builder(np.array([1.0, 2.0, 3.0]))

    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))


# ---------------------------------------------------------------------
# State and torque validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "state",
    [
        np.zeros(5),
        np.zeros(7),
        np.zeros((6, 1)),
    ],
)
def test_invalid_state_shape(system_matrices, state):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError):
        state_derivative(state, inertia, damping, stiffness)


@pytest.mark.parametrize(
    "state",
    [
        [0.0, 0.0, 0.0, 0.0, np.nan, 0.0],
        [0.0, 0.0, np.inf, 0.0, 0.0, 0.0],
    ],
)
def test_nonfinite_state(system_matrices, state):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="finite"):
        state_derivative(state, inertia, damping, stiffness)


def test_nonzero_torque_requires_input_matrix(system_matrices):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="input_matrix is required"):
        state_derivative(
            np.zeros(6),
            inertia,
            damping,
            stiffness,
            torque_input=[1.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "torque",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, np.nan, 0.0],
        [1.0, np.inf, 0.0],
    ],
)
def test_invalid_torque_input(system_matrices, torque):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError):
        state_derivative(
            np.zeros(6),
            inertia,
            damping,
            stiffness,
            torque_input=torque,
            input_matrix=np.eye(3),
        )


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.eye(2),
        np.ones((3, 2)),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
    ],
)
def test_invalid_input_matrix(system_matrices, input_matrix):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError):
        state_derivative(
            np.zeros(6),
            inertia,
            damping,
            stiffness,
            torque_input=[1.0, 0.0, 0.0],
            input_matrix=input_matrix,
        )