
"""
Time-domain simulation for a 3-DOF rotational torsional system.

The governing equation is

    J theta_ddot + D theta_dot + K_theta theta = B_tau tau

where:
    J       = rotational inertia matrix
    D       = rotational damping matrix
    K_theta = torsional stiffness matrix
    B_tau   = torque-input matrix
    tau     = applied torque vector

The state vector is

    z = [theta1, theta2, theta3, omega1, omega2, omega3]

where theta denotes angular displacement in rad and omega denotes
angular velocity in rad/s.

This module integrates the first-order state-space representation
using scipy.integrate.solve_ivp.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from src.dynamics import state_derivative


def simulate_system(
    inertia_matrix: ArrayLike,
    damping_matrix: ArrayLike,
    stiffness_matrix: ArrayLike,
    initial_state: ArrayLike,
    time_span: tuple[float, float],
    time_step: float,
    torque_input: ArrayLike | None = None,
    input_matrix: ArrayLike | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simulate the 3-DOF rotational torsional system.

    Parameters
    ----------
    inertia_matrix : array-like
        3x3 rotational inertia matrix J.

    damping_matrix : array-like
        3x3 rotational damping matrix D.

    stiffness_matrix : array-like
        3x3 torsional stiffness matrix K_theta.

    initial_state : array-like
        Initial state [theta1, theta2, theta3, omega1, omega2, omega3].
        Angular displacements are in rad; angular velocities are in rad/s.

    time_span : tuple[float, float]
        Start and end times in seconds.

    time_step : float
        Requested interval between output points, in seconds.
        The final output point is always the requested end time.

    torque_input : array-like, optional
        Constant torque vector with three components, in N*m.
        Defaults to zero torque.

    input_matrix : array-like, optional
        3x3 torque-input matrix B_tau. Required for nonzero torque.

    rtol : float, optional
        Relative tolerance passed to solve_ivp.

    atol : float, optional
        Absolute tolerance passed to solve_ivp.

    Returns
    -------
    time : np.ndarray
        Output time vector with shape (n_time,).

    state_history : np.ndarray
        State history with shape (6, n_time). Each column contains
        the three angular displacements followed by the three
        angular velocities.

    Raises
    ------
    ValueError
        If an input has an invalid shape or value.

    RuntimeError
        If time integration fails.
    """
    inertia_matrix = np.asarray(inertia_matrix, dtype=float)
    damping_matrix = np.asarray(damping_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)

    matrices = (
        ("inertia_matrix", inertia_matrix),
        ("damping_matrix", damping_matrix),
        ("stiffness_matrix", stiffness_matrix),
    )

    for name, matrix in matrices:
        if matrix.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values.")

    if initial_state.shape != (6,):
        raise ValueError("initial_state must contain exactly six values.")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain only finite values.")

    if len(time_span) != 2:
        raise ValueError("time_span must contain exactly two values.")

    try:
        t_start, t_end = map(float, time_span)
        time_step = float(time_step)
        rtol = float(rtol)
        atol = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_span, time_step, rtol, and atol must be numeric."
        ) from exc

    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("time_span must contain only finite values.")

    if t_end <= t_start:
        raise ValueError("time_span must have t_end greater than t_start.")

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be positive and finite.")

    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite.")

    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")

    if torque_input is None:
        torque = np.zeros(3, dtype=float)
    else:
        torque = np.asarray(torque_input, dtype=float)

        if torque.shape != (3,):
            raise ValueError("torque_input must contain exactly three values.")

        if not np.all(np.isfinite(torque)):
            raise ValueError(
                "torque_input must contain only finite values."
            )

    if input_matrix is None:
        if np.any(torque != 0.0):
            raise ValueError(
                "input_matrix is required when torque_input is nonzero."
            )
        torque_input_matrix = None
    else:
        torque_input_matrix = np.asarray(input_matrix, dtype=float)

        if torque_input_matrix.shape != (3, 3):
            raise ValueError("input_matrix must have shape (3, 3).")

        if not np.all(np.isfinite(torque_input_matrix)):
            raise ValueError(
                "input_matrix must contain only finite values."
            )

    # Construct output times without exceeding the integration interval.
    # Append the endpoint when time_step does not divide the interval evenly.
    duration = t_end - t_start
    number_of_steps = int(np.floor(duration / time_step))
    output_times = t_start + time_step * np.arange(
        number_of_steps + 1,
        dtype=float,
    )

    endpoint_tolerance = (
        10.0 * np.finfo(float).eps * max(1.0, abs(t_end), abs(t_start))
    )

    if abs(output_times[-1] - t_end) <= endpoint_tolerance:
        output_times[-1] = t_end
    elif output_times[-1] < t_end:
        output_times = np.append(output_times, t_end)

    def derivative(
        time: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del time

        return state_derivative(
            state,
            inertia_matrix,
            damping_matrix,
            stiffness_matrix,
            torque_input=torque,
            input_matrix=torque_input_matrix,
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

    if solution.t.size != output_times.size:
        raise RuntimeError(
            "Time integration returned fewer output points than requested."
        )

    return solution.t, solution.y
