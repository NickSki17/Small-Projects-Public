
import numpy as np
import pytest

from src.dynamics import (
    build_inertia_matrix,
    build_rotational_damping_matrix,
    build_torsional_stiffness_matrix,
)
from src.metrics import (
    compute_acceleration,
    compute_response_metrics,
    dominant_frequency,
)


@pytest.fixture
def system_matrices():
    inertia = build_inertia_matrix([1.0, 2.0, 3.0])
    damping = build_rotational_damping_matrix([1.0, 2.0, 3.0])
    stiffness = build_torsional_stiffness_matrix([10.0, 20.0, 30.0])
    return inertia, damping, stiffness


@pytest.fixture
def time_vector():
    return np.linspace(0.0, 1.0, 101)


@pytest.fixture
def zero_state_history(time_vector):
    return np.zeros((6, time_vector.size), dtype=float)


def test_acceleration_shape(system_matrices, zero_state_history):
    inertia, damping, stiffness = system_matrices

    acceleration = compute_acceleration(
        zero_state_history, inertia, damping, stiffness
    )

    assert acceleration.shape == (3, zero_state_history.shape[1])


def test_zero_state_has_zero_acceleration(system_matrices, zero_state_history):
    acceleration = compute_acceleration(
        zero_state_history, *system_matrices
    )

    np.testing.assert_allclose(acceleration, 0.0)


def test_known_torque_produces_known_angular_acceleration():
    inertia = np.eye(3)
    damping = np.zeros((3, 3))
    stiffness = np.zeros((3, 3))
    input_matrix = np.eye(3)
    torque = np.array([2.0, -3.0, 4.0])
    state_history = np.zeros((6, 5))

    acceleration = compute_acceleration(
        state_history,
        inertia,
        damping,
        stiffness,
        torque_input=torque,
        input_matrix=input_matrix,
    )

    expected = np.repeat(torque[:, np.newaxis], 5, axis=1)
    np.testing.assert_allclose(acceleration, expected)


def test_stiffness_restoring_acceleration():
    inertia = np.eye(3)
    damping = np.zeros((3, 3))
    stiffness = np.diag([2.0, 3.0, 4.0])
    state_history = np.zeros((6, 4))
    state_history[0, :] = 0.5
    state_history[1, :] = -1.0
    state_history[2, :] = 0.25

    acceleration = compute_acceleration(
        state_history, inertia, damping, stiffness
    )

    expected = np.array([-1.0, 3.0, -1.0])[:, np.newaxis]
    np.testing.assert_allclose(
        acceleration,
        np.repeat(expected, 4, axis=1),
    )


def test_nonzero_torque_requires_input_matrix():
    with pytest.raises(ValueError, match="input_matrix is required"):
        compute_acceleration(
            np.zeros((6, 3)),
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            torque_input=[1.0, 0.0, 0.0],
        )


def test_zero_torque_without_input_matrix_is_valid():
    acceleration = compute_acceleration(
        np.zeros((6, 3)),
        np.eye(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        torque_input=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(acceleration, 0.0)


def test_invalid_state_history_shape():
    with pytest.raises(ValueError, match="shape"):
        compute_acceleration(
            np.zeros((5, 10)),
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )


def test_nonfinite_state_history_is_rejected():
    state_history = np.zeros((6, 4))
    state_history[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_acceleration(
            state_history,
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )


@pytest.mark.parametrize(
    "matrix_index",
    [0, 1, 2],
)
def test_invalid_matrix_shape_is_rejected(matrix_index):
    matrices = [
        np.eye(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    ]
    matrices[matrix_index] = np.eye(2)

    with pytest.raises(ValueError, match="shape"):
        compute_acceleration(
            np.zeros((6, 4)),
            *matrices,
        )


def test_invalid_torque_shape_is_rejected():
    with pytest.raises(ValueError, match="three values"):
        compute_acceleration(
            np.zeros((6, 3)),
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            torque_input=[1.0, 2.0],
        )


def test_invalid_input_matrix_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        compute_acceleration(
            np.zeros((6, 3)),
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            torque_input=[1.0, 0.0, 0.0],
            input_matrix=np.eye(2),
        )


def test_zero_response_metrics(system_matrices, time_vector, zero_state_history):
    metrics = compute_response_metrics(
        time_vector,
        zero_state_history,
        *system_matrices,
    )

    assert metrics == {
        "theta_max": 0.0,
        "theta1_max": 0.0,
        "theta2_max": 0.0,
        "theta3_max": 0.0,
        "alpha_max": 0.0,
        "theta_rms": 0.0,
    }


def test_response_metrics_calculate_maximum_angle_and_rms(
    system_matrices, time_vector
):
    state_history = np.zeros((6, time_vector.size))
    state_history[0, :] = 2.0
    state_history[1, :] = -1.0
    state_history[2, :] = 0.5

    metrics = compute_response_metrics(
        time_vector,
        state_history,
        *system_matrices,
    )

    assert metrics["theta_max"] == pytest.approx(2.0)
    assert metrics["theta1_max"] == pytest.approx(2.0)
    assert metrics["theta2_max"] == pytest.approx(1.0)
    assert metrics["theta3_max"] == pytest.approx(0.5)

    expected_rms = np.sqrt((2.0**2 + (-1.0)**2 + 0.5**2) / 3.0)
    assert metrics["theta_rms"] == pytest.approx(expected_rms)


def test_metrics_reject_mismatched_time_and_state_lengths(system_matrices):
    with pytest.raises(ValueError, match="same number"):
        compute_response_metrics(
            [0.0, 0.1, 0.2],
            np.zeros((6, 4)),
            *system_matrices,
        )


def test_metrics_reject_nonincreasing_time(system_matrices):
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_response_metrics(
            [0.0, 0.2, 0.1],
            np.zeros((6, 3)),
            *system_matrices,
        )


def test_dominant_frequency_of_sinusoid():
    sampling_rate = 100.0
    time = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    signal = np.sin(2.0 * np.pi * 5.0 * time)

    frequency = dominant_frequency(time, signal)

    assert frequency == pytest.approx(5.0)


def test_dominant_frequency_rejects_nonuniform_sampling():
    time = np.array([0.0, 0.1, 0.2, 0.35, 0.4])
    signal = np.sin(2.0 * np.pi * time)

    with pytest.raises(ValueError, match="uniformly sampled"):
        dominant_frequency(time, signal)


def test_dominant_frequency_rejects_constant_signal():
    time = np.linspace(0.0, 1.0, 20)

    with pytest.raises(ValueError, match="constant signal"):
        dominant_frequency(time, np.ones_like(time))


def test_dominant_frequency_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number"):
        dominant_frequency(
            np.linspace(0.0, 1.0, 10),
            np.zeros(9),
        )
