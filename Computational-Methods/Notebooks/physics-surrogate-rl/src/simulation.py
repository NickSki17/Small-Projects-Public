"""
Time-domain simulation for the 3-DOF linear mass-spring-damper system.

This module integrates the first-order state-space representation
defined in dynamics.py using scipy.integrate.solve_ivp.

The physical model is

    M x_ddot + C x_dot + K x = B u

with state

    z = [x1, x2, x3, v1, v2, v3].

For passive simulations, the force input is zero.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from src.dynamics import state_derivative


def simulate_system(
    mass_matrix: NDArray[np.float64],
    damping_matrix: NDArray[np.float64],
    stiffness_matrix: NDArray[np.float64],
    initial_state: ArrayLike,
    time_span: tuple[float, float],
    time_step: float,
    force_input: ArrayLike | None = None,
    input_matrix: NDArray[np.float64] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Simulate the 3-DOF mass-spring-damper system.

    Parameters
    ----------
    mass_matrix : np.ndarray
        3x3 mass matrix M.

    damping_matrix : np.ndarray
        3x3 damping matrix C.

    stiffness_matrix : np.ndarray
        3x3 stiffness matrix K.

    initial_state : array-like
        Initial state [x1, x2, x3, v1, v2, v3].

    time_span : tuple[float, float]
        Start and end times (t_start, t_end) in seconds.

    time_step : float
        Time interval between returned output points.

    force_input : array-like, optional
        Constant force input vector u. Defaults to zero input.

    input_matrix : np.ndarray, optional
        Input matrix B. Required for nonzero force input.

    rtol : float, optional
        Relative tolerance passed to solve_ivp.

    atol : float, optional
        Absolute tolerance passed to solve_ivp.

    Returns
    -------
    time : np.ndarray
        Simulation time vector.

    state_history : np.ndarray
        State history with shape (6, n_time).

        Each column contains

            [x1, x2, x3, v1, v2, v3].

    Raises
    ------
    ValueError
        If an input has an invalid shape or value.

    RuntimeError
        If solve_ivp fails to integrate the system.
    """
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    damping_matrix = np.asarray(damping_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)

    if mass_matrix.shape != (3, 3):
        raise ValueError("mass_matrix must have shape (3, 3).")

    if damping_matrix.shape != (3, 3):
        raise ValueError("damping_matrix must have shape (3, 3).")

    if stiffness_matrix.shape != (3, 3):
        raise ValueError("stiffness_matrix must have shape (3, 3).")

    if initial_state.shape != (6,):
        raise ValueError("initial_state must contain exactly six values.")

    if not np.all(np.isfinite(mass_matrix)):
        raise ValueError("mass_matrix must contain only finite values.")

    if not np.all(np.isfinite(damping_matrix)):
        raise ValueError("damping_matrix must contain only finite values.")

    if not np.all(np.isfinite(stiffness_matrix)):
        raise ValueError("stiffness_matrix must contain only finite values.")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain only finite values.")

    if len(time_span) != 2:
        raise ValueError("time_span must contain exactly two values.")

    t_start, t_end = map(float, time_span)

    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("time_span must contain only finite values.")

    if t_end <= t_start:
        raise ValueError("time_span must have t_end greater than t_start.")

    time_step = float(time_step)

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be positive and finite.")

    rtol = float(rtol)
    atol = float(atol)

    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite.")

    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")

    if force_input is not None:
        force_input = np.asarray(force_input, dtype=float)

        if force_input.shape != (3,):
            raise ValueError("force_input must contain exactly three values.")

        if not np.all(np.isfinite(force_input)):
            raise ValueError(
                "force_input must contain only finite values."
            )

    if input_matrix is not None:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.shape != (3, 3):
            raise ValueError("input_matrix must have shape (3, 3).")

        if not np.all(np.isfinite(input_matrix)):
            raise ValueError(
                "input_matrix must contain only finite values."
            )

    output_times = np.arange(
        t_start,
        t_end + 0.5 * time_step,
        time_step,
        dtype=float,
    )

    if output_times[-1] > t_end:
        output_times[-1] = t_end

    def derivative(
        time: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del time

        return state_derivative(
            state,
            mass_matrix,
            damping_matrix,
            stiffness_matrix,
            force_input=force_input,
            input_matrix=input_matrix,
        )

    solution = solve_ivp(
        derivative,
        (t_start, t_end),
        initial_state,
        t_eval=output_times,
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            f"Time integration failed: {solution.message}"
        )

    return solution.t, solution.y