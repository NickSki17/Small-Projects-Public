"""
vibration_analysis.py

Free-vibration analysis of an n-mass, fixed-fixed spring chain.

Physical model
--------------
n identical masses m, connected in series by n+1 identical springs of
stiffness k, with the first and last masses also connected to fixed walls:

    wall --k-- m1 --k-- m2 --k-- ... --k-- mn --k-- wall

Damping is neglected: this is an undamped, conservative system. Governing
equation in matrix form (M = m*I, since all masses are identical):

    M x'' + K x = 0   <=>   x'' = -(K/m) x

State-space (first-order) form, with z = [x; x_dot]:

    z' = A z,      A = [[0, I], [-K/m, 0]]

Eigenvalue interpretation
--------------------------
For an undamped conservative system, the eigenvalues of A are purely
imaginary and occur in conjugate pairs, lambda = +/- i*omega_k. The
positive imaginary parts correspond to the system's natural angular
frequencies (rad/s). There are exactly n distinct positive natural
frequencies for n masses.

Fourier analysis
-----------------
The free response of any single mass, given generic (non-modal) initial
conditions, is a superposition of all n normal modes. The single-sided
amplitude spectrum (FFT) of a mass's position time history should show
peaks at frequencies omega_k / (2*pi) [Hz], which is the numerical
cross-check performed in `main()`: simulated spectral peaks are compared
against the frequencies computed analytically from the eigenvalue problem.

Assumptions and limitations
-----------------------------
- Uniform mass m and stiffness k across all elements. Heterogeneous
  properties would require `generate_k_matrix` to accept arrays instead
  of scalars.
- No damping matrix C: energy is conserved and oscillations do not decay.
  This is why the eigenvalues are purely imaginary rather than complex
  with negative real parts.
- Free vibration only (no external forcing): the system responds purely
  to its random initial conditions.
- Numerical eigenvalues may contain small imaginary-part errors from finite
  precision. A tolerance is used to exclude values that are effectively zero;
  the retained positive frequencies are returned without rounding so that
  numerical precision is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
from scipy.linalg import eig

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# --------------------------------------------------------------------------
# Baseline simulation parameters (documented for reproducibility)
# --------------------------------------------------------------------------
DEFAULT_K = 1.0
DEFAULT_M = 1.0
DEFAULT_SEED = 42
BASELINE_N_LIST = (2, 3, 5, 10)
BASELINE_N_SIM = 20
BASELINE_T = 100.0
BASELINE_DT = 0.02


@dataclass(frozen=True)
class SimulationResult:
    """Results from an n-mass free-vibration simulation."""

    t: np.ndarray
    y: np.ndarray
    n: int


# --------------------------------------------------------------------------
# System matrices
# --------------------------------------------------------------------------

def generate_k_matrix(
    n: int,
    k: float = DEFAULT_K,
    m: float = DEFAULT_M,
) -> np.ndarray:
    """
    Build the n x n matrix representing K/m for a fixed-fixed identical
    spring-mass chain (tridiagonal, main diagonal 2k/m, off-diagonal -k/m).

    The main diagonal is 2k/m for every mass because each interior mass
    is flanked by two springs of stiffness k; the boundary masses connect
    to a wall spring on one side and a chain spring on the other, which is
    still stiffness k, preserving the uniform-diagonal structure for this
    fixed-fixed configuration.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")

    main_diag = np.full(n, 2.0 * k / m)
    off_diag = np.full(n - 1, -k / m)

    return (
        np.diag(main_diag)
        + np.diag(off_diag, k=1)
        + np.diag(off_diag, k=-1)
    )


def generate_state_matrix(k_over_m: np.ndarray) -> np.ndarray:
    """Build the 2n x 2n first-order system matrix A for z' = A z."""
    n = k_over_m.shape[0]
    a_mat = np.zeros((2 * n, 2 * n))
    a_mat[:n, n:] = np.eye(n)
    a_mat[n:, :n] = -k_over_m
    return a_mat


def _first_order_rhs(
    t: float,
    z: np.ndarray,
    k_over_m: np.ndarray,
) -> np.ndarray:
    """
    ODE right-hand side for solve_ivp: z' = A z, expressed without forming
    A explicitly. The precomputed k_over_m matrix is reused on every RHS
    evaluation.
    """
    n = k_over_m.shape[0]
    pos = z[:n]
    vel = z[n:]

    dzdt = np.empty_like(z)
    dzdt[:n] = vel
    dzdt[n:] = -k_over_m @ pos

    return dzdt


# --------------------------------------------------------------------------
# Eigenvalue / natural frequency analysis
# --------------------------------------------------------------------------

def compute_natural_frequencies(
    n: int,
    k: float = DEFAULT_K,
    m: float = DEFAULT_M,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Return the n positive natural frequencies (rad/s) of the
    n-mass fixed-fixed chain, sorted ascending.
    """
    k_over_m = generate_k_matrix(n, k, m)
    a_mat = generate_state_matrix(k_over_m)
    eigenvalues = eig(a_mat, right=False)

    imag_parts = np.imag(eigenvalues)
    positive = imag_parts[imag_parts > tol]

    return np.sort(positive)


# --------------------------------------------------------------------------
# Time-domain simulation
# --------------------------------------------------------------------------

def simulate_free_vibration(
    n: int,
    t_final: float = BASELINE_T,
    dt: float = BASELINE_DT,
    k: float = DEFAULT_K,
    m: float = DEFAULT_M,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """
    Simulate free vibration of the n-mass chain from random initial
    conditions (small random displacements, random velocities), integrated
    with an explicit Runge-Kutta method (RK45).

    A fixed random seed is used so results are reproducible run to run.
    Output is sampled on a uniform grid via `np.linspace`, which guarantees
    both endpoints are included and the point count is deterministic.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if t_final <= 0 or dt <= 0:
        raise ValueError(
            f"t_final and dt must be positive, "
            f"got t_final={t_final}, dt={dt}"
        )

    rng = np.random.default_rng(seed)
    z0 = np.empty(2 * n)
    z0[:n] = (rng.random(n) - 0.5) * 0.1
    z0[n:] = rng.random(n) - 0.5

    k_over_m = generate_k_matrix(n, k, m)
    n_points = int(round(t_final / dt)) + 1
    t_eval = np.linspace(0.0, t_final, n_points)

    sol = solve_ivp(
        _first_order_rhs,
        t_span=(0.0, t_final),
        y0=z0,
        t_eval=t_eval,
        args=(k_over_m,),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return SimulationResult(t=sol.t, y=sol.y, n=n)


# --------------------------------------------------------------------------
# Fourier analysis
# --------------------------------------------------------------------------

def compute_spectrum(
    time_series: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided amplitude spectrum of a real time series.

    Returns:
        frequencies: Positive frequencies in Hz.
        amplitude: Corresponding single-sided amplitudes.
    """
    time_series = np.asarray(time_series, dtype=float)

    if not np.all(np.isfinite(time_series)):
        raise ValueError("time_series must contain only finite values.")

    n_points = len(time_series)

    if n_points < 2 or dt <= 0:
        return np.array([]), np.array([])

    spectrum_full = fft(time_series)
    freqs_full = fftfreq(n_points, dt)

    n_positive = n_points // 2 + 1
    freqs = freqs_full[:n_positive]
    amplitude = np.abs(spectrum_full[:n_positive]) / n_points

    if n_points > 2:
        if n_points % 2 == 0:
            amplitude[1:-1] *= 2.0
        else:
            amplitude[1:] *= 2.0

    return freqs, amplitude


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_trajectories(
    sim: SimulationResult,
    mass_indices: tuple[int, ...],
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in mass_indices:
        ax.plot(
            sim.t,
            sim.y[i, :],
            label=f"Mass {i + 1}",
            linewidth=1.2,
        )

    ax.set_title(f"Free-Vibration Mass Trajectories (n = {sim.n})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"trajectories_n{sim.n}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Trajectory plot saved to {out_path}")

    plt.close(fig)


def plot_spectra(
    sim: SimulationResult,
    mass_indices: tuple[int, ...],
    expected_freqs_hz: np.ndarray,
    save: bool = True,
) -> None:
    dt_actual = sim.t[1] - sim.t[0]
    fig, ax = plt.subplots(figsize=(10, 5))

    max_amplitude = 0.0

    for i in mass_indices:
        freqs, amplitude = compute_spectrum(sim.y[i, :], dt_actual)

        if freqs.size:
            ax.plot(
                freqs,
                amplitude,
                label=f"Mass {i + 1}",
                linewidth=1.2,
            )
            max_amplitude = max(
                max_amplitude,
                float(np.max(amplitude)),
            )

    for j, f_exp in enumerate(expected_freqs_hz):
        ax.axvline(
            f_exp,
            color="r",
            linestyle="--",
            alpha=0.4,
            label="Expected natural frequency" if j == 0 else None,
        )

    ax.set_title(
        f"Frequency Spectrum vs. Predicted Natural Frequencies (n = {sim.n})"
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")

    if expected_freqs_hz.size:
        ax.set_xlim(0, expected_freqs_hz[-1] * 1.2)

    if max_amplitude > 0:
        ax.set_ylim(0, max_amplitude * 1.1)

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"spectrum_n{sim.n}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Spectrum plot saved to {out_path}")

    plt.close(fig)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def main() -> None:
    print("=== Natural Frequency Analysis (Eigenvalue Problem) ===")
    print(f"Parameters: k = {DEFAULT_K}, m = {DEFAULT_M}\n")

    for n in BASELINE_N_LIST:
        freqs = compute_natural_frequencies(n, DEFAULT_K, DEFAULT_M)
        print(
            f"n = {n:3d}: natural frequencies (rad/s) = "
            f"{np.round(freqs, 4)}"
        )

    print("\n=== Time-Domain Simulation and Fourier Verification ===")
    print(
        f"n = {BASELINE_N_SIM}, T = {BASELINE_T} s, "
        f"dt = {BASELINE_DT} s, seed = {DEFAULT_SEED}"
    )

    sim = simulate_free_vibration(
        BASELINE_N_SIM,
        t_final=BASELINE_T,
        dt=BASELINE_DT,
        k=DEFAULT_K,
        m=DEFAULT_M,
        seed=DEFAULT_SEED,
    )

    mass_indices = (
        0,
        BASELINE_N_SIM // 2,
        BASELINE_N_SIM - 1,
    )

    expected_omegas = compute_natural_frequencies(
        BASELINE_N_SIM,
        DEFAULT_K,
        DEFAULT_M,
    )
    expected_freqs_hz = expected_omegas / (2 * np.pi)

    print(
        f"Expected natural frequencies (Hz): "
        f"{np.round(expected_freqs_hz, 4)}"
    )

    plot_trajectories(sim, mass_indices)
    plot_spectra(sim, mass_indices, expected_freqs_hz)

    print(
        "\nCheck: FFT peaks should occur near the dashed lines marking "
        "the analytically predicted natural frequencies."
    )


if __name__ == "__main__":
    main()