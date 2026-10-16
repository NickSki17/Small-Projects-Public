"""
Engineering response metrics for the 3-DOF linear mass-spring-damper system.

This module extracts engineering quantities from time-domain simulation
results produced by simulation.py.

The primary metrics are:

    xmax  : maximum absolute displacement across all DOFs and time
    amax  : maximum absolute acceleration across all DOFs and time
    x_rms : system-level RMS displacement

Acceleration is reconstructed from the governing equation

    M x_ddot + C x_dot + K x = B u

rather than by numerically differentiating velocity.

A separate FFT utility is provided for identifying the dominant frequency
of a uniformly sampled response signal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_simulation_output(
    time: ArrayLike,
    state_history: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Validate and convert simulation time and state-history arrays.

    Parameters
    ----------
    time : array-like
        Simulation time vector with shape (n_time,).

    state_history : array-like
        State history with shape (6, n_time).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated time vector and state history.
    """
    time = np.asarray(time, dtype=float)
    state_history = np.asarray(state_history, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")

    if state_history.ndim != 2:
        raise ValueError("state_history must be a two-dimensional array.")

    if state_history.shape[0] != 6:
        raise ValueError(
            "state_history must have shape (6, n_time)."
        )

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
        raise ValueError(
            "state_history must contain only finite values."
        )

    if np.any(np.diff(time) <= 0.0):
        raise ValueError(
            "time must be strictly increasing."
        )

    return time, state_history


def compute_acceleration(
    state_history: ArrayLike,
    mass_matrix: NDArray[np.float64],
    damping_matrix: NDArray[np.float64],
    stiffness_matrix: NDArray[np.float64],
    force_input: ArrayLike | None = None,
    input_matrix: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Compute acceleration from the governing mechanical equation.

    The equation

        M x_ddot = B u - C x_dot - K x

    is evaluated at every simulation time point.

    Parameters
    ----------
    state_history : np.ndarray
        State history with shape (6, n_time).

    mass_matrix : np.ndarray
        3x3 mass matrix M.

    damping_matrix : np.ndarray
        3x3 damping matrix C.

    stiffness_matrix : np.ndarray
        3x3 stiffness matrix K.

    force_input : array-like, optional
        Constant force input vector u. Defaults to zero.

    input_matrix : np.ndarray, optional
        3x3 input matrix B. Required when force_input is nonzero.

    Returns
    -------
    np.ndarray
        Acceleration history with shape (3, n_time).
    """
    state_history = np.asarray(state_history, dtype=float)
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    damping_matrix = np.asarray(damping_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)

    if state_history.ndim != 2 or state_history.shape[0] != 6:
        raise ValueError(
            "state_history must have shape (6, n_time)."
        )

    if mass_matrix.shape != (3, 3):
        raise ValueError("mass_matrix must have shape (3, 3).")

    if damping_matrix.shape != (3, 3):
        raise ValueError("damping_matrix must have shape (3, 3).")

    if stiffness_matrix.shape != (3, 3):
        raise ValueError("stiffness_matrix must have shape (3, 3).")

    if not np.all(np.isfinite(state_history)):
        raise ValueError(
            "state_history must contain only finite values."
        )

    if force_input is None:
        force = np.zeros(3, dtype=float)
    else:
        force = np.asarray(force_input, dtype=float)

        if force.shape != (3,):
            raise ValueError(
                "force_input must contain exactly three values."
            )

        if not np.all(np.isfinite(force)):
            raise ValueError(
                "force_input must contain only finite values."
            )

    if input_matrix is None:
        if np.any(force != 0.0):
            raise ValueError(
                "input_matrix is required when force_input is nonzero."
            )

        applied_force = np.zeros(
            (3, state_history.shape[1]),
            dtype=float,
        )

    else:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.shape != (3, 3):
            raise ValueError(
                "input_matrix must have shape (3, 3)."
            )

        if not np.all(np.isfinite(input_matrix)):
            raise ValueError(
                "input_matrix must contain only finite values."
            )

        applied_force = (
            input_matrix @ force
        )[:, np.newaxis]

    positions = state_history[:3]
    velocities = state_history[3:]

    rhs = (
        applied_force
        - damping_matrix @ velocities
        - stiffness_matrix @ positions
    )

    return np.linalg.solve(mass_matrix, rhs)


def compute_response_metrics(
    time: ArrayLike,
    state_history: ArrayLike,
    mass_matrix: NDArray[np.float64],
    damping_matrix: NDArray[np.float64],
    stiffness_matrix: NDArray[np.float64],
    force_input: ArrayLike | None = None,
    input_matrix: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """
    Compute the primary engineering response metrics.

    The system-level metrics combine all three DOFs.

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        "xmax"
            Maximum absolute displacement across all DOFs and time.

        "amax"
            Maximum absolute acceleration across all DOFs and time.

        "x_rms"
            RMS displacement, with each DOF weighted equally.
    """
    time, state_history = _validate_simulation_output(
        time,
        state_history,
    )

    acceleration = compute_acceleration(
        state_history,
        mass_matrix,
        damping_matrix,
        stiffness_matrix,
        force_input=force_input,
        input_matrix=input_matrix,
    )

    positions = state_history[:3]

    xmax = float(np.max(np.abs(positions)))
    amax = float(np.max(np.abs(acceleration)))

    duration = time[-1] - time[0]

    displacement_squared_integral = np.trapezoid(
        positions**2,
        time,
        axis=1,
    )

    per_dof_mean_square = (
        displacement_squared_integral / duration
    )

    x_rms = float(
        np.sqrt(np.mean(per_dof_mean_square))
    )

    return {
        "xmax": xmax,
        "amax": amax,
        "x_rms": x_rms,
    }


def dominant_frequency(
    time: ArrayLike,
    signal: ArrayLike,
) -> float:
    """
    Determine the dominant nonzero frequency of a uniformly sampled signal.

    The mean is removed before the FFT so that a DC offset does not
    dominate the frequency spectrum.

    Parameters
    ----------
    time : array-like
        Uniformly sampled time vector in seconds.

    signal : array-like
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
        raise ValueError(
            "time and signal must contain the same number of values."
        )

    if time.size < 3:
        raise ValueError(
            "At least three samples are required for FFT analysis."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "signal must contain only finite values."
        )

    time_differences = np.diff(time)

    if np.any(time_differences <= 0.0):
        raise ValueError(
            "time must be strictly increasing."
        )

    if not np.allclose(
        time_differences,
        time_differences[0],
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(
            "time must be uniformly sampled for FFT analysis."
        )

    signal_centered = signal - np.mean(signal)

    spectrum = np.abs(
        np.fft.rfft(signal_centered)
    )

    frequencies = np.fft.rfftfreq(
        signal.size,
        d=time_differences[0],
    )

    if frequencies.size < 2:
        raise ValueError(
            "Insufficient frequency resolution for FFT analysis."
        )

    dominant_index = np.argmax(
        spectrum[1:]
    ) + 1

    return float(frequencies[dominant_index])