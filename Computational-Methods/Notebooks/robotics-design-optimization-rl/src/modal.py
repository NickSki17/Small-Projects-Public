
"""
Modal analysis for a 3-DOF rotational torsional system.

For undamped free vibration:

    J theta_ddot + K_theta theta = 0

Assuming harmonic motion produces the generalized eigenvalue problem:

    K_theta phi = lambda J phi

where:

    lambda = omega**2

J          : rotational inertia matrix
K_theta    : torsional stiffness matrix
omega      : natural angular frequency in rad/s
phi        : rotational mode shape

Mode shapes are normalized such that:

    phi.T @ J @ phi = I
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh


def solve_modes(
    inertia_matrix: ArrayLike,
    torsional_stiffness_matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve the generalized eigenvalue problem for rotational modes.

    Parameters
    ----------
    inertia_matrix : array-like
        3x3 symmetric positive-definite rotational inertia matrix J.

    torsional_stiffness_matrix : array-like
        3x3 symmetric torsional stiffness matrix K_theta.

    Returns
    -------
    natural_frequencies : np.ndarray
        Three natural angular frequencies in rad/s, sorted ascending.

    mode_shapes : np.ndarray
        Corresponding mode shapes. Each column is a mode shape.
        The columns are inertia-normalized:

            mode_shapes.T @ J @ mode_shapes = I

    Raises
    ------
    ValueError
        If the matrices have invalid dimensions, contain nonfinite
        values, violate symmetry, or do not define a positive-definite
        inertia and stiffness system.
    """
    inertia_matrix = np.asarray(inertia_matrix, dtype=float)
    torsional_stiffness_matrix = np.asarray(
        torsional_stiffness_matrix,
        dtype=float,
    )

    if inertia_matrix.shape != (3, 3):
        raise ValueError("inertia_matrix must have shape (3, 3).")

    if torsional_stiffness_matrix.shape != (3, 3):
        raise ValueError(
            "torsional_stiffness_matrix must have shape (3, 3)."
        )

    if not np.all(np.isfinite(inertia_matrix)):
        raise ValueError(
            "inertia_matrix must contain only finite values."
        )

    if not np.all(np.isfinite(torsional_stiffness_matrix)):
        raise ValueError(
            "torsional_stiffness_matrix must contain only finite values."
        )

    if not np.allclose(inertia_matrix, inertia_matrix.T):
        raise ValueError("inertia_matrix must be symmetric.")

    if not np.allclose(
        torsional_stiffness_matrix,
        torsional_stiffness_matrix.T,
    ):
        raise ValueError(
            "torsional_stiffness_matrix must be symmetric."
        )

    inertia_eigenvalues = np.linalg.eigvalsh(inertia_matrix)

    if np.any(inertia_eigenvalues <= 0.0):
        raise ValueError(
            "inertia_matrix must be positive definite."
        )

    stiffness_eigenvalues = np.linalg.eigvalsh(
        torsional_stiffness_matrix
    )

    if np.any(stiffness_eigenvalues <= 0.0):
        raise ValueError(
            "torsional_stiffness_matrix must be positive definite."
        )

    eigenvalues, mode_shapes = eigh(
        torsional_stiffness_matrix,
        inertia_matrix,
    )

    # For a positive-definite inertia and stiffness system,
    # all generalized eigenvalues must be positive.
    if np.any(eigenvalues <= 0.0):
        raise ValueError(
            "All generalized eigenvalues must be positive."
        )

    natural_frequencies = np.sqrt(eigenvalues)

    return natural_frequencies, mode_shapes


def angular_to_hz(
    natural_frequencies: ArrayLike,
) -> NDArray[np.float64]:
    """
    Convert natural angular frequencies from rad/s to Hz.

    Parameters
    ----------
    natural_frequencies : array-like
        One-dimensional array of angular frequencies in rad/s.

    Returns
    -------
    np.ndarray
        Frequencies in Hz.
    """
    natural_frequencies = np.asarray(
        natural_frequencies,
        dtype=float,
    )

    if natural_frequencies.ndim != 1:
        raise ValueError(
            "natural_frequencies must be a one-dimensional array."
        )

    if not np.all(np.isfinite(natural_frequencies)):
        raise ValueError(
            "natural_frequencies must contain only finite values."
        )

    if np.any(natural_frequencies < 0.0):
        raise ValueError(
            "natural_frequencies must be nonnegative."
        )

    return natural_frequencies / (2.0 * np.pi)
