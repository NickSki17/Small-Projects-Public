
"""
Engineering response metrics for the 3-DOF rotational torsional system.

The governing equation is

    J theta_ddot + C_theta theta_dot + K_theta theta = B_tau tau

This module calculates angular acceleration from the physical equation,
extracts system-level angular response metrics, and provides an FFT
utility for uniformly sampled response signals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_simulation_output(
    time: ArrayLike,
    state_history: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate time and state-history arrays."""
    time = np.asarray(time, dtype=float)
    state_history = np.asarray(state_history, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")

    if state_history.ndim != 2 or state_history.shape[0] != 6:
        raise ValueError("state_history must have shape (6, n_time).")

    if state_history.shape[1] != time.size:
        raise ValueError(
            "state_history and time must contain the same number "
            "of time points."
        )

    if time.size < 2:
        raise ValueError("At least two time points are required.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(state_history)):
        raise ValueError("state_history must contain only finite values.")

    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    return time, state_history


def compute_acceleration(
    state_history: ArrayLike,
    inertia_matrix: ArrayLike,
    damping_matrix: ArrayLike,
    stiffness_matrix: ArrayLike,
    torque_input: ArrayLike | None = None,
    input_matrix: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """
    Compute angular acceleration from the governing torsional equation.

    Parameters
    ----------
    state_history:
        State history with shape (6, n_time), ordered as
        [theta1, theta2, theta3, omega1, omega2, omega3].

    inertia_matrix:
        3x3 rotational inertia matrix J.

    damping_matrix:
        3x3 rotational damping matrix C_theta.

    stiffness_matrix:
        3x3 torsional stiffness matrix K_theta.

    torque_input:
        Constant generalized torque vector with shape (3,).
        Defaults to zero torque.

    input_matrix:
        3x3 torque-input distribution matrix B_tau. Required when
        torque_input is nonzero.

    Returns
    -------
    np.ndarray
        Angular acceleration history with shape (3, n_time).
    """
    state_history = np.asarray(state_history, dtype=float)
    inertia_matrix = np.asarray(inertia_matrix, dtype=float)
    damping_matrix = np.asarray(damping_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)

    if state_history.ndim != 2 or state_history.shape[0] != 6:
        raise ValueError("state_history must have shape (6, n_time).")

    if state_history.shape[1] == 0:
        raise ValueError("state_history must contain at least one time point.")

    if not np.all(np.isfinite(state_history)):
        raise ValueError("state_history must contain only finite values.")

    for name, matrix in (
        ("inertia_matrix", inertia_matrix),
        ("damping_matrix", damping_matrix),
        ("stiffness_matrix", stiffness_matrix),
    ):
        if matrix.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values.")

    if torque_input is None:
        torque = np.zeros(3, dtype=float)
    else:
        torque = np.asarray(torque_input, dtype=float)
        if torque.shape != (3,):
            raise ValueError("torque_input must contain exactly three values.")
        if not np.all(np.isfinite(torque)):
            raise ValueError("torque_input must contain only finite values.")

    if input_matrix is None:
        if np.any(torque != 0.0):
            raise ValueError(
                "input_matrix is required when torque_input is nonzero."
            )
        applied_torque = np.zeros(
            (3, state_history.shape[1]), dtype=float
        )
    else:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.shape != (3, 3):
            raise ValueError("input_matrix must have shape (3, 3).")
        if not np.all(np.isfinite(input_matrix)):
            raise ValueError("input_matrix must contain only finite values.")

        applied_torque = (input_matrix @ torque)[:, np.newaxis]

    angles = state_history[:3]
    angular_velocities = state_history[3:]

    rhs = (
        applied_torque
        - damping_matrix @ angular_velocities
        - stiffness_matrix @ angles
    )

    # Solve J * alpha = rhs; do not explicitly invert J.
    return np.linalg.solve(inertia_matrix, rhs)


def compute_response_metrics(
    time: ArrayLike,
    state_history: ArrayLike,
    inertia_matrix: ArrayLike,
    damping_matrix: ArrayLike,
    stiffness_matrix: ArrayLike,
    torque_input: ArrayLike | None = None,
    input_matrix: ArrayLike | None = None,
) -> dict[str, float]:
    """
    Compute system-level angular response metrics.

    Returns
    -------
    dict[str, float]
        theta_max: Maximum absolute angular displacement across all DOFs
                   and time, in radians.
        alpha_max: Maximum absolute angular acceleration across all DOFs
                   and time, in rad/s^2.
        theta_rms: Time-weighted RMS angular displacement, with each DOF
                   weighted equally, in radians.
    """
    time, state_history = _validate_simulation_output(time, state_history)

    acceleration = compute_acceleration(
        state_history,
        inertia_matrix,
        damping_matrix,
        stiffness_matrix,
        torque_input=torque_input,
        input_matrix=input_matrix,
    )

    angles = state_history[:3]

    # Peak absolute angular displacement for each joint.
    theta1_max = float(np.max(np.abs(angles[0])))
    theta2_max = float(np.max(np.abs(angles[1])))
    theta3_max = float(np.max(np.abs(angles[2])))

    # Global peak across all joints and time.
    theta_max = max(theta1_max, theta2_max, theta3_max)

    # Global peak absolute angular acceleration.
    alpha_max = float(np.max(np.abs(acceleration)))

    # Time-weighted RMS, with equal weighting for each joint.
    duration = time[-1] - time[0]
    mean_square_by_dof = (
        np.trapezoid(angles**2, x=time, axis=1) / duration
    )
    theta_rms = float(np.sqrt(np.mean(mean_square_by_dof)))

    return {
        "theta_max": theta_max,
        "theta1_max": theta1_max,
        "theta2_max": theta2_max,
        "theta3_max": theta3_max,
        "alpha_max": alpha_max,
        "theta_rms": theta_rms,
    }


def dominant_frequency(
    time: ArrayLike,
    signal: ArrayLike,
) -> float:
    """
    Find the dominant nonzero FFT frequency of a uniformly sampled signal.

    Parameters
    ----------
    time:
        Uniformly sampled time vector in seconds.
    signal:
        One-dimensional response signal.

    Returns
    -------
    float
        Dominant frequency in Hz.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array.")

    if time.size != signal.size:
        raise ValueError("time and signal must contain the same number of values.")

    if time.size < 3:
        raise ValueError("At least three samples are required for FFT analysis.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")

    time_differences = np.diff(time)

    if np.any(time_differences <= 0.0):
        raise ValueError("time must be strictly increasing.")

    if not np.allclose(
        time_differences,
        time_differences[0],
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError("time must be uniformly sampled for FFT analysis.")

    centered_signal = signal - np.mean(signal)

    if np.allclose(centered_signal, 0.0):
        raise ValueError("Cannot determine a dominant frequency for a constant signal.")

    spectrum = np.abs(np.fft.rfft(centered_signal))
    frequencies = np.fft.rfftfreq(signal.size, d=time_differences[0])

    # Exclude the DC component.
    dominant_index = int(np.argmax(spectrum[1:]) + 1)

    return float(frequencies[dominant_index])
