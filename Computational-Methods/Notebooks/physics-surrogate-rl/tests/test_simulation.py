import numpy as np
import pytest

from src.dynamics import (
    build_damping_matrix,
    build_mass_matrix,
    build_stiffness_matrix,
)
from src.simulation import simulate_system


def create_test_system():
    mass = build_mass_matrix([1.0, 1.0, 1.0])
    damping = build_damping_matrix([1.0, 1.0, 1.0])
    stiffness = build_stiffness_matrix([10.0, 10.0, 10.0])

    return mass, damping, stiffness


def test_simulation_output_shapes():
    mass, damping, stiffness = create_test_system()

    initial_state = np.array(
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    time, state_history = simulate_system(
        mass,
        damping,
        stiffness,
        initial_state,
        time_span=(0.0, 2.0),
        time_step=0.01,
    )

    assert time.ndim == 1
    assert state_history.ndim == 2

    assert state_history.shape[0] == 6
    assert state_history.shape[1] == len(time)


def test_simulation_starts_at_initial_state():
    mass, damping, stiffness = create_test_system()

    initial_state = np.array(
        [0.1, -0.05, 0.02, 0.0, 0.01, -0.02]
    )

    time, state_history = simulate_system(
        mass,
        damping,
        stiffness,
        initial_state,
        time_span=(0.0, 1.0),
        time_step=0.01,
    )

    assert time[0] == pytest.approx(0.0)

    np.testing.assert_allclose(
        state_history[:, 0],
        initial_state,
    )


def test_zero_initial_state_remains_zero():
    mass, damping, stiffness = create_test_system()

    initial_state = np.zeros(6)

    _, state_history = simulate_system(
        mass,
        damping,
        stiffness,
        initial_state,
        time_span=(0.0, 2.0),
        time_step=0.01,
    )

    np.testing.assert_allclose(
        state_history,
        np.zeros_like(state_history),
        atol=1e-10,
    )


def test_free_response_decays_with_damping():
    mass, damping, stiffness = create_test_system()

    initial_state = np.array(
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    _, state_history = simulate_system(
        mass,
        damping,
        stiffness,
        initial_state,
        time_span=(0.0, 5.0),
        time_step=0.01,
    )

    initial_amplitude = np.max(
        np.abs(state_history[:3, 0])
    )

    final_amplitude = np.max(
        np.abs(state_history[:3, -1])
    )

    assert final_amplitude < initial_amplitude


def test_constant_force_produces_response():
    mass, damping, stiffness = create_test_system()

    initial_state = np.zeros(6)

    input_matrix = np.eye(3)
    force = np.array([1.0, 0.0, 0.0])

    _, state_history = simulate_system(
        mass,
        damping,
        stiffness,
        initial_state,
        time_span=(0.0, 2.0),
        time_step=0.01,
        force_input=force,
        input_matrix=input_matrix,
    )

    assert np.max(np.abs(state_history[0])) > 0.0


def test_nonzero_force_requires_input_matrix():
    mass, damping, stiffness = create_test_system()

    initial_state = np.zeros(6)

    with pytest.raises(ValueError):
        simulate_system(
            mass,
            damping,
            stiffness,
            initial_state,
            time_span=(0.0, 1.0),
            time_step=0.01,
            force_input=[1.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "time_span",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
    ],
)
def test_invalid_time_span(time_span):
    mass, damping, stiffness = create_test_system()

    with pytest.raises(ValueError):
        simulate_system(
            mass,
            damping,
            stiffness,
            np.zeros(6),
            time_span=time_span,
            time_step=0.01,
        )


def test_invalid_time_step():
    mass, damping, stiffness = create_test_system()

    with pytest.raises(ValueError):
        simulate_system(
            mass,
            damping,
            stiffness,
            np.zeros(6),
            time_span=(0.0, 1.0),
            time_step=0.0,
        )


def test_invalid_initial_state():
    mass, damping, stiffness = create_test_system()

    with pytest.raises(ValueError):
        simulate_system(
            mass,
            damping,
            stiffness,
            np.zeros(5),
            time_span=(0.0, 1.0),
            time_step=0.01,
        )


def test_time_output_is_monotonic():
    mass, damping, stiffness = create_test_system()

    time, _ = simulate_system(
        mass,
        damping,
        stiffness,
        np.zeros(6),
        time_span=(0.0, 1.0),
        time_step=0.01,
    )

    assert np.all(np.diff(time) > 0.0)