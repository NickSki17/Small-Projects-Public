import numpy as np
import pytest

from src.dynamics import (
    build_damping_matrix,
    build_mass_matrix,
    build_stiffness_matrix,
)
from src.metrics import (
    compute_acceleration,
    compute_response_metrics,
    dominant_frequency,
)


def make_test_system():
    masses = np.array([1.0, 2.0, 1.5])
    stiffnesses = np.array([100.0, 150.0, 200.0])
    dampings = np.array([1.0, 2.0, 1.5])

    mass_matrix = build_mass_matrix(masses)
    stiffness_matrix = build_stiffness_matrix(stiffnesses)
    damping_matrix = build_damping_matrix(dampings)

    return mass_matrix, damping_matrix, stiffness_matrix


def test_compute_acceleration_shape():
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()

    state_history = np.zeros((6, 101))

    acceleration = compute_acceleration(
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    assert acceleration.shape == (3, 101)


def test_zero_state_has_zero_acceleration():
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()

    state_history = np.zeros((6, 101))

    acceleration = compute_acceleration(
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    np.testing.assert_allclose(acceleration, 0.0)


def test_acceleration_matches_governing_equation():
    mass_matrix = np.eye(3)
    damping_matrix = np.zeros((3, 3))
    stiffness_matrix = np.diag([10.0, 20.0, 30.0])

    state_history = np.zeros((6, 2))
    state_history[:3] = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )

    acceleration = compute_acceleration(
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    expected = np.array(
        [
            [-10.0, -10.0],
            [-40.0, -40.0],
            [-90.0, -90.0],
        ]
    )

    np.testing.assert_allclose(
        acceleration,
        expected,
    )


def test_control_input_contributes_to_acceleration():
    mass_matrix = np.eye(3)
    damping_matrix = np.zeros((3, 3))
    stiffness_matrix = np.zeros((3, 3))

    input_matrix = np.eye(3)
    force_input = np.array([2.0, -3.0, 4.0])

    state_history = np.zeros((6, 5))

    acceleration = compute_acceleration(
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
        force_input=force_input,
        input_matrix=input_matrix,
    )

    expected = np.tile(
        force_input[:, np.newaxis],
        (1, 5),
    )

    np.testing.assert_allclose(
        acceleration,
        expected,
    )


def test_nonzero_force_requires_input_matrix():
    mass_matrix = np.eye(3)
    damping_matrix = np.zeros((3, 3))
    stiffness_matrix = np.zeros((3, 3))

    state_history = np.zeros((6, 5))

    with pytest.raises(ValueError):
        compute_acceleration(
            state_history,
            mass_matrix,
            damping_matrix,
            stiffness_matrix,
            force_input=np.array([1.0, 0.0, 0.0]),
        )


def test_response_metrics_zero_response():
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()

    time = np.linspace(0.0, 1.0, 101)
    state_history = np.zeros((6, time.size))

    metrics = compute_response_metrics(
        time,
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    assert metrics["xmax"] == pytest.approx(0.0)
    assert metrics["amax"] == pytest.approx(0.0)
    assert metrics["x_rms"] == pytest.approx(0.0)


def test_response_metrics_constant_displacement():
    mass_matrix = np.eye(3)
    damping_matrix = np.zeros((3, 3))
    stiffness_matrix = np.zeros((3, 3))

    time = np.linspace(0.0, 2.0, 201)

    state_history = np.zeros((6, time.size))
    state_history[0] = 2.0
    state_history[1] = -1.0
    state_history[2] = 0.5

    metrics = compute_response_metrics(
        time,
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    assert metrics["xmax"] == pytest.approx(2.0)
    assert metrics["amax"] == pytest.approx(0.0)

    expected_rms = np.sqrt(
        (2.0**2 + (-1.0)**2 + 0.5**2) / 3.0
    )

    assert metrics["x_rms"] == pytest.approx(
        expected_rms
    )


def test_response_metrics_keys():
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()

    time = np.linspace(0.0, 1.0, 101)
    state_history = np.zeros((6, time.size))

    metrics = compute_response_metrics(
        time,
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    assert set(metrics) == {
        "xmax",
        "amax",
        "x_rms",
    }


def test_dominant_frequency():
    frequency = 5.0
    time = np.linspace(0.0, 2.0, 1001)
    signal = np.sin(2.0 * np.pi * frequency * time)

    result = dominant_frequency(
        time,
        signal,
    )

    assert result == pytest.approx(
        frequency,
        abs=0.01,
    )


def test_dominant_frequency_removes_dc_offset():
    frequency = 8.0
    time = np.linspace(0.0, 2.0, 1001)

    signal = (
        4.0
        + np.sin(2.0 * np.pi * frequency * time)
    )

    result = dominant_frequency(
        time,
        signal,
    )

    assert result == pytest.approx(
        frequency,
        abs=0.01,
    )


def test_dominant_frequency_requires_uniform_sampling():
    time = np.array([
        0.0,
        0.1,
        0.2,
        0.35,
        0.4,
    ])

    signal = np.ones(time.size)

    with pytest.raises(ValueError):
        dominant_frequency(
            time,
            signal,
        )


@pytest.mark.parametrize(
    "time",
    [
        np.array([0.0, 0.1, 0.1, 0.2]),
        np.array([0.0, 0.2, 0.1, 0.3]),
    ],
)
def test_response_metrics_requires_increasing_time(time):
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()
    state_history = np.zeros((6, time.size))

    with pytest.raises(ValueError):
        compute_response_metrics(
            time,
            state_history,
            mass_matrix,
            damping_matrix,
            stiffness_matrix,
        )


def test_response_metrics_rejects_mismatched_shapes():
    mass_matrix, damping_matrix, stiffness_matrix = make_test_system()

    time = np.linspace(0.0, 1.0, 101)
    state_history = np.zeros((6, 100))

    with pytest.raises(ValueError):
        compute_response_metrics(
            time,
            state_history,
            mass_matrix,
            damping_matrix,
            stiffness_matrix,
        )