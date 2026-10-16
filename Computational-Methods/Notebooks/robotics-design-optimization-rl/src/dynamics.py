
"""
Reduced-order rotational dynamics for a 3-DOF mechanical system.

The model is a chain of three rotational inertias coupled by
torsional springs and dampers:

fixed base -- k1,c1 -- J1 -- k2,c2 -- J2 -- k3,c3 -- J3

This is a linearized rotational model, not a full rigid-body
robot manipulator model.

The governing equation is

    J @ q_ddot + D @ q_dot + K_theta @ q = B @ torque

where:
    J       : rotational inertia matrix [kg m^2]
    D       : rotational damping matrix [N m s/rad]
    K_theta : torsional stiffness matrix [N m/rad]
    B       : actuator input matrix
    torque  : applied joint torques [N m]
    q       : joint-angle deviations [rad]

The state vector is

    z = [q1, q2, q3, qd1, qd2, qd3]

where q_i are angular deviations and qd_i are angular velocities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def build_inertia_matrix(rotational_inertias: ArrayLike) -> np.ndarray:
    """
    Build the diagonal rotational inertia matrix.

    Parameters
    ----------
    rotational_inertias : array-like
        [J1, J2, J3] in kg m^2. All values must be positive.

    Returns
    -------
    np.ndarray
        3x3 diagonal rotational inertia matrix.
    """
    inertias = np.asarray(rotational_inertias, dtype=float)

    if inertias.shape != (3,):
        raise ValueError(
            "rotational_inertias must contain exactly three values."
        )

    if not np.all(np.isfinite(inertias)):
        raise ValueError(
            "rotational_inertias must contain only finite values."
        )

    if np.any(inertias <= 0.0):
        raise ValueError("All rotational inertias must be positive.")

    return np.diag(inertias)


def build_torsional_stiffness_matrix(
    stiffnesses: ArrayLike,
) -> np.ndarray:
    """
    Build the torsional stiffness matrix for the 3-DOF chain.

    Parameters
    ----------
    stiffnesses : array-like
        [k1, k2, k3] in N m/rad. All values must be positive.

    Returns
    -------
    np.ndarray
        3x3 torsional stiffness matrix.
    """
    stiffnesses = np.asarray(stiffnesses, dtype=float)

    if stiffnesses.shape != (3,):
        raise ValueError("stiffnesses must contain exactly three values.")

    if not np.all(np.isfinite(stiffnesses)):
        raise ValueError("stiffnesses must contain only finite values.")

    if np.any(stiffnesses <= 0.0):
        raise ValueError("All torsional stiffnesses must be positive.")

    k1, k2, k3 = stiffnesses

    return np.array(
        [
            [k1 + k2, -k2, 0.0],
            [-k2, k2 + k3, -k3],
            [0.0, -k3, k3],
        ],
        dtype=float,
    )


def build_rotational_damping_matrix(
    dampings: ArrayLike,
) -> np.ndarray:
    """
    Build the rotational damping matrix for the 3-DOF chain.

    Parameters
    ----------
    dampings : array-like
        [c1, c2, c3] in N m s/rad.
        Values must be nonnegative.

    Returns
    -------
    np.ndarray
        3x3 rotational damping matrix.
    """
    dampings = np.asarray(dampings, dtype=float)

    if dampings.shape != (3,):
        raise ValueError("dampings must contain exactly three values.")

    if not np.all(np.isfinite(dampings)):
        raise ValueError("dampings must contain only finite values.")

    if np.any(dampings < 0.0):
        raise ValueError("Rotational damping coefficients must be nonnegative.")

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
    inertia_matrix: np.ndarray,
    damping_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
    torque_input: ArrayLike | None = None,
    input_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the first-order state derivative.

    The equation of motion is

        J @ q_ddot = B @ torque - D @ q_dot - K_theta @ q

    Parameters
    ----------
    state : array-like
        [q1, q2, q3, qd1, qd2, qd3].
    inertia_matrix : np.ndarray
        3x3 rotational inertia matrix.
    damping_matrix : np.ndarray
        3x3 rotational damping matrix.
    stiffness_matrix : np.ndarray
        3x3 torsional stiffness matrix.
    torque_input : array-like, optional
        Three applied joint torques in N m. Defaults to zero.
    input_matrix : np.ndarray, optional
        3x3 actuator input matrix. Required for nonzero torque.

    Returns
    -------
    np.ndarray
        [qd1, qd2, qd3, qdd1, qdd2, qdd3].
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError("state must contain exactly six values.")

    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values.")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values.")

    inertia_matrix = np.asarray(inertia_matrix, dtype=float)
    damping_matrix = np.asarray(damping_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)

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
            raise ValueError(
                "torque_input must contain only finite values."
            )

    if input_matrix is None:
        if np.any(torque != 0.0):
            raise ValueError(
                "input_matrix is required when torque_input is nonzero."
            )

        applied_torque = np.zeros(3, dtype=float)

    else:
        input_matrix = np.asarray(input_matrix, dtype=float)

        if input_matrix.shape != (3, 3):
            raise ValueError("input_matrix must have shape (3, 3).")

        if not np.all(np.isfinite(input_matrix)):
            raise ValueError(
                "input_matrix must contain only finite values."
            )

        applied_torque = input_matrix @ torque

    angles = state[:3]
    angular_velocities = state[3:]

    angular_accelerations = np.linalg.solve(
        inertia_matrix,
        applied_torque
        - damping_matrix @ angular_velocities
        - stiffness_matrix @ angles,
    )

    return np.concatenate((angular_velocities, angular_accelerations))
