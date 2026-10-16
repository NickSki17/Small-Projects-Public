
import numpy as np
import pytest

from src.dynamics import (
    build_inertia_matrix,
    build_rotational_damping_matrix,
    build_torsional_stiffness_matrix,
)
from src.simulation import simulate_system


@pytest.fixture
def system_matrices():
    inertia = build_inertia_matrix([0.5, 0.8, 1.2])
    damping = build_rotational_damping_matrix([0.2, 0.3, 0.4])
    stiffness = build_torsional_stiffness_matrix([20.0, 35.0, 50.0])
    return inertia, damping, stiffness


def test_output_shapes(system_matrices):
    inertia, damping, stiffness = system_matrices

    time, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_span=(0.0, 1.0),
        time_step=0.01,
    )

    assert time.ndim == 1
    assert state_history.shape == (6, time.size)
    assert time.size > 1


def test_simulation_starts_at_initial_state(system_matrices):
    inertia, damping, stiffness = system_matrices
    initial_state = np.array([0.1, -0.02, 0.03, 0.2, 0.0, -0.1])

    time, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state,
        (0.0, 0.5),
        0.01,
    )

    assert time[0] == pytest.approx(0.0)
    np.testing.assert_allclose(
        state_history[:, 0],
        initial_state,
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_state_remains_zero(system_matrices):
    inertia, damping, stiffness = system_matrices

    _, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state=np.zeros(6),
        time_span=(0.0, 1.0),
        time_step=0.01,
    )

    np.testing.assert_allclose(
        state_history,
        np.zeros_like(state_history),
        atol=1e-12,
    )


def test_free_response_decays_with_rotational_damping(system_matrices):
    inertia, damping, stiffness = system_matrices

    initial_state = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    _, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state,
        (0.0, 8.0),
        0.01,
    )

    initial_angle = np.max(np.abs(state_history[:3, 0]))
    final_angle = np.max(np.abs(state_history[:3, -1]))

    assert final_angle < initial_angle


def test_constant_torque_produces_rotational_response(system_matrices):
    inertia, damping, stiffness = system_matrices

    _, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state=np.zeros(6),
        time_span=(0.0, 1.0),
        time_step=0.01,
        torque_input=[1.0, 0.0, 0.0],
        input_matrix=np.eye(3),
    )

    assert np.max(np.abs(state_history[:3, :])) > 0.0


def test_nonzero_torque_requires_input_matrix(system_matrices):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="input_matrix is required"):
        simulate_system(
            inertia,
            damping,
            stiffness,
            initial_state=np.zeros(6),
            time_span=(0.0, 1.0),
            time_step=0.01,
            torque_input=[1.0, 0.0, 0.0],
        )


def test_zero_torque_without_input_matrix_is_valid(system_matrices):
    inertia, damping, stiffness = system_matrices

    time, state_history = simulate_system(
        inertia,
        damping,
        stiffness,
        initial_state=np.zeros(6),
        time_span=(0.0, 0.2),
        time_step=0.01,
        torque_input=[0.0, 0.0, 0.0],
    )

    assert time.size > 1
    np.testing.assert_allclose(state_history, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "time_span",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, np.inf),
    ],
)
def test_invalid_time_span_is_rejected(system_matrices, time_span):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError):
        simulate_system(
            inertia,
            damping,
            stiffness,
            np.zeros(6),
            time_span,
            0.01,
        )


@pytest.mark.parametrize("time_step", [0.0, -0.01, np.inf, np.nan])
def test_invalid_time_step_is_rejected(system_matrices, time_step):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="time_step"):
        simulate_system(
            inertia,
            damping,
            stiffness,
            np.zeros(6),
            (0.0, 1.0),
            time_step,
        )


@pytest.mark.parametrize(
    "initial_state",
    [
        np.zeros(5),
        np.zeros(7),
        np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]),
    ],
)
def test_invalid_initial_state_is_rejected(system_matrices, initial_state):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="initial_state"):
        simulate_system(
            inertia,
            damping,
            stiffness,
            initial_state,
            (0.0, 1.0),
            0.01,
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
def test_invalid_torque_input_is_rejected(system_matrices, torque):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="torque_input"):
        simulate_system(
            inertia,
            damping,
            stiffness,
            np.zeros(6),
            (0.0, 1.0),
            0.01,
            torque_input=torque,
            input_matrix=np.eye(3),
        )


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.eye(2),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
    ],
)
def test_invalid_input_matrix_is_rejected(system_matrices, input_matrix):
    inertia, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="input_matrix"):
        simulate_system(
            inertia,
            damping,
            stiffness,
            np.zeros(6),
            (0.0, 1.0),
            0.01,
            torque_input=[1.0, 0.0, 0.0],
            input_matrix=input_matrix,
        )


def test_output_time_is_strictly_increasing(system_matrices):
    inertia, damping, stiffness = system_matrices

    time, _ = simulate_system(
        inertia,
        damping,
        stiffness,
        np.zeros(6),
        (0.0, 1.0),
        0.03,
    )

    assert np.all(np.diff(time) > 0.0)
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(1.0)


def test_nondivisible_time_step_includes_final_time(system_matrices):
    inertia, damping, stiffness = system_matrices

    time, _ = simulate_system(
        inertia,
        damping,
        stiffness,
        np.zeros(6),
        (0.0, 1.0),
        0.3,
    )

    assert time[-1] == pytest.approx(1.0)
    assert time.size == 5


def test_invalid_matrix_shape_is_rejected(system_matrices):
    _, damping, stiffness = system_matrices

    with pytest.raises(ValueError, match="inertia_matrix"):
        simulate_system(
            np.eye(2),
            damping,
            stiffness,
            np.zeros(6),
            (0.0, 1.0),
            0.01,
        )
