"""
Modal analysis for a 3-DOF linear mass-spring-damper system.

For undamped free vibration, the governing equation is

    M x_ddot + K x = 0

Assuming harmonic motion gives the generalized eigenvalue problem

    K phi = lambda M phi

where

    lambda = omega^2

The natural angular frequencies are therefore

    omega = sqrt(lambda)

This module performs undamped modal analysis using the mass and
stiffness matrices constructed by dynamics.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def solve_modes(
    mass_matrix: NDArray[np.float64],
    stiffness_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve the generalized eigenvalue problem for the natural modes.

    Parameters
    ----------
    mass_matrix : np.ndarray
        3x3 mass matrix M. It must be symmetric positive definite.

    stiffness_matrix : np.ndarray
        3x3 stiffness matrix K. It must be symmetric.

    Returns
    -------
    natural_frequencies : np.ndarray
        Natural angular frequencies [omega_1, omega_2, omega_3]
        in rad/s, sorted from lowest to highest.

    mode_shapes : np.ndarray
        Corresponding mode shapes. Each column is one mode shape.
        The columns are mass-normalized such that

            phi.T @ M @ phi = I.
    """
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    stiffness_matrix = np.asarray(stiffness_matrix, dtype=float)

    if mass_matrix.shape != (3, 3):
        raise ValueError("mass_matrix must have shape (3, 3).")

    if stiffness_matrix.shape != (3, 3):
        raise ValueError("stiffness_matrix must have shape (3, 3).")

    if not np.all(np.isfinite(mass_matrix)):
        raise ValueError("mass_matrix must contain only finite values.")

    if not np.all(np.isfinite(stiffness_matrix)):
        raise ValueError(
            "stiffness_matrix must contain only finite values."
        )

    if not np.allclose(mass_matrix, mass_matrix.T):
        raise ValueError("mass_matrix must be symmetric.")

    if not np.allclose(stiffness_matrix, stiffness_matrix.T):
        raise ValueError("stiffness_matrix must be symmetric.")

    mass_eigenvalues = np.linalg.eigvalsh(mass_matrix)

    if np.any(mass_eigenvalues <= 0.0):
        raise ValueError("mass_matrix must be positive definite.")

    eigenvalues, mode_shapes = eigh(
        stiffness_matrix,
        mass_matrix,
    )

    if np.any(eigenvalues <= 0.0):
        raise ValueError(
            "All eigenvalues must be positive for this fixed-wall system."
        )

    natural_frequencies = np.sqrt(eigenvalues)

    return natural_frequencies, mode_shapes


def angular_to_hz(
    natural_frequencies: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert angular natural frequencies from rad/s to Hz.

    Parameters
    ----------
    natural_frequencies : np.ndarray
        Natural angular frequencies in rad/s.

    Returns
    -------
    np.ndarray
        Natural frequencies in Hz.
    """
    natural_frequencies = np.asarray(natural_frequencies, dtype=float)

    if natural_frequencies.ndim != 1:
        raise ValueError("natural_frequencies must be a one-dimensional array.")

    if not np.all(np.isfinite(natural_frequencies)):
        raise ValueError(
            "natural_frequencies must contain only finite values."
        )

    if np.any(natural_frequencies < 0.0):
        raise ValueError(
            "natural_frequencies must be nonnegative."
        )

    return natural_frequencies / (2.0 * np.pi)