"""
Physical model for a 3-DOF linear mass-spring-damper system.

The mechanical system is:

    wall -- k1,c1 -- m1 -- k2,c2 -- m2 -- k3,c3 -- m3

The governing equation is

    M x_ddot + C x_dot + K x = B u

where:
    M = mass matrix
    C = damping matrix
    K = stiffness matrix
    B = force-input matrix
    u = applied force input

For passive analysis, u = 0.

The state vector is

    z = [x1, x2, x3, v1, v2, v3]

where x_i are displacements and v_i are velocities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def build_mass_matrix(masses: ArrayLike) -> np.ndarray:
    """
    Build the diagonal mass matrix M.

    Parameters
    ----------
    masses : array-like
        Masses [m1, m2, m3]. All values must be positive.

    Returns
    -------
    np.ndarray
        3x3 diagonal mass matrix.
    """
    masses = np.asarray(masses, dtype=float)

    if masses.shape != (3,):
        raise ValueError("masses must contain exactly three values.")

    if not np.all(np.isfinite(masses)):
        raise ValueError("masses must contain only finite values.")

    if np.any(masses <= 0.0):
        raise ValueError("All masses must be positive.")

    return np.diag(masses)


def build_stiffness_matrix(stiffnesses: ArrayLike) -> np.ndarray:
    """
    Build the stiffness matrix K for the fixed-wall 3-DOF chain.

    Parameters
    ----------
    stiffnesses : array-like
        Spring stiffnesses [k1, k2, k3]. All values must be positive.

    Returns
    -------
    np.ndarray
        3x3 stiffness matrix.
    """
    stiffnesses = np.asarray(stiffnesses, dtype=float)

    if stiffnesses.shape != (3,):
        raise ValueError("stiffnesses must contain exactly three values.")

    if not np.all(np.isfinite(stiffnesses)):
        raise ValueError("stiffnesses must contain only finite values.")

    if np.any(stiffnesses <= 0.0):
        raise ValueError("All stiffnesses must be positive.")

    k1, k2, k3 = stiffnesses

    return np.array(
        [
            [k1 + k2, -k2, 0.0],
            [-k2, k2 + k3, -k3],
            [0.0, -k3, k3],
        ],
        dtype=float,
    )


def build_damping_matrix(dampings: ArrayLike) -> np.ndarray:
    """
    Build the damping matrix C for the fixed-wall 3-DOF chain.

    Parameters
    ----------
    dampings : array-like
        Damping coefficients [c1, c2, c3].
        Values must be nonnegative.

    Returns
    -------
    np.ndarray
        3x3 damping matrix.
    """
    dampings = np.asarray(dampings, dtype=float)

    if dampings.shape != (3,):
        raise ValueError("dampings must contain exactly three values.")

    if not np.all(np.isfinite(dampings)):
        raise ValueError("dampings must contain only finite values.")

    if np.any(dampings < 0.0):
        raise ValueError("Damping coefficients must be nonnegative.")

    c1, c2, c3 = dampings

    return np.array(
        [
            [c1 + c2, -c2, 0.0],
            [-c2, c2 + c3, -c3],
            [0.0, -c3, c3],
        ],
        dtype=float,
    )


def state_derivative(
    state: ArrayLike,
    mass_matrix: np.ndarray,
    damping_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
    force_input: ArrayLike | None = None,
    input_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the first-order state derivative.

    The second-order equation is

        M x_ddot + C x_dot + K x = B u

    which gives

        x_ddot = M^{-1}(B u - C x_dot - K x).

    Parameters
    ----------
    state : array-like
        State vector [x1, x2, x3, v1, v2, v3].
    mass_matrix : np.ndarray
        3x3 mass matrix M.
    damping_matrix : np.ndarray
        3x3 damping matrix C.
    stiffness_matrix : np.ndarray
        3x3 stiffness matrix K.
    force_input : array-like, optional
        Applied force vector u. Defaults to zero input.
    input_matrix : np.ndarray, optional
        Input matrix B. Required when a nonzero force input is supplied.

    Returns
    -------
    np.ndarray
        State derivative [v1, v2, v3, a1, a2, a3].
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError("state must contain exactly six values.")

    positions = state[:3]
    velocities = state[3:]

    if force_input is None:
        force = np.zeros(3, dtype=float)
    else:
        force = np.asarray(force_input, dtype=float)

        if force.shape != (3,):
            raise ValueError("force_input must contain exactly three values.")

        if not np.all(np.isfinite(force)):
            raise ValueError("force_input must contain only finite values.")

    if input_matrix is None:
        if np.any(force != 0.0):
            raise ValueError(
                "input_matrix is required when force_input is nonzero."
            )

        applied_force = np.zeros(3, dtype=float)

    else:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.shape != (3, 3):
            raise ValueError("input_matrix must have shape (3, 3).")

        if not np.all(np.isfinite(input_matrix)):
            raise ValueError("input_matrix must contain only finite values.")

        applied_force = input_matrix @ force

    acceleration = np.linalg.solve(
        mass_matrix,
        applied_force
        - damping_matrix @ velocities
        - stiffness_matrix @ positions,
    )

    return np.concatenate((velocities, acceleration))