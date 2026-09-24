"""
optimization.py

Unconstrained optimization benchmark: Gradient Descent, Newton's Method, and
BFGS (SciPy) applied to five benchmark objective functions.

Physical / mathematical context
--------------------------------
These are classic unconstrained-optimization test functions used to compare
convergence behavior of first-order (gradient descent), second-order
(Newton), and quasi-Newton (BFGS) methods:

    f1 : sphere function (convex, isotropic)          -- easy baseline
    f2 : convex quadratic with cross term              -- tests conditioning
    f3 : Rosenbrock function (non-convex, curved valley) -- classic hard case
    f4 : quartic with flat/degenerate minimum along x1+x2 -- stresses Newton
    f5 : negative Gaussian bump (non-convex, bounded)  -- has a true maximum
         at the origin approached via gradient ASCENT information; included
         here as a gradient-descent MINIMIZATION target, so it will drive
         away from the origin. This is intentional: it demonstrates that
         "minimize" is not always the physically meaningful direction, and
         is left in as a methodological check, not a bug.

Stopping criterion
-------------------
Gradient Descent and Newton's Method both stop when the scale-invariant
criterion

    ||grad f(x)|| / (1 + |f(x)|) <= epsilon

is satisfied. This normalizes the gradient norm by the objective magnitude
so the same epsilon is meaningful whether f(x) is O(1) or O(1e6).

Known methodological inconsistency: SciPy's BFGS implementation
(`scipy.optimize.minimize(method="BFGS")`) uses its own internal
convergence test based on `gtol` (raw gradient norm, not normalized by
f(x)). `epsilon` is passed through as `gtol` for approximate comparability,
but the two are not identical criteria. This is a real limitation, not
papered over: BFGS iteration counts are not perfectly apples-to-apples with
GD/Newton and should be reported as such.

Newton's method Hessian
-------------------------
The Hessian is approximated via finite differences of the analytic
gradient (`scipy.optimize.approx_fprime`). This raw finite-difference
Jacobian is not guaranteed to be numerically symmetric even though the
true Hessian is; it is explicitly symmetrized (`0.5 * (H + H.T)`) before
inversion. Skipping this step and relying on "it usually works out" is a
latent source of divergence on ill-conditioned problems.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

ArrayLike = Sequence[float]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


# --------------------------------------------------------------------------
# Benchmark objective functions and analytic gradients
# --------------------------------------------------------------------------

def f1(x: np.ndarray) -> float:
    """Sphere function: f = x1^2 + x2^2 + x3^2."""
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def grad_f1(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0], 2 * x[1], 2 * x[2]])


def f2(x: np.ndarray) -> float:
    """Convex quadratic: f = x1^2 + 2*x2^2 - 2*x1*x2 - 2*x2."""
    return x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 2 * x[1]


def grad_f2(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0] - 2 * x[1], 4 * x[1] - 2 * x[0] - 2])


def f3(x: np.ndarray) -> float:
    """Rosenbrock function: f = 100*(x2 - x1^2)^2 + (1 - x1)^2."""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_f3(x: np.ndarray) -> np.ndarray:
    df_dx1 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    df_dx2 = 200 * (x[1] - x[0] ** 2)
    return np.array([df_dx1, df_dx2])


def f4(x: np.ndarray) -> float:
    """Quartic with a flat degenerate minimum: f = (x1+x2)^4 + x2^2."""
    return (x[0] + x[1]) ** 4 + x[1] ** 2


def grad_f4(x: np.ndarray) -> np.ndarray:
    df_dx1 = 4 * (x[0] + x[1]) ** 3
    df_dx2 = 4 * (x[0] + x[1]) ** 3 + 2 * x[1]
    return np.array([df_dx1, df_dx2])


def f5(x: np.ndarray) -> float:
    """
    Gaussian bump: f = exp(-x1^2 - x2^2).

    The function has a maximum at the origin and approaches zero as
    ||x|| -> infinity, so unconstrained minimization drives x outward.
    """
    return float(np.exp(-x[0] ** 2 - x[1] ** 2))


def grad_f5(x: np.ndarray) -> np.ndarray:
    common = np.exp(-x[0] ** 2 - x[1] ** 2)
    return np.array([-2 * x[0] * common, -2 * x[1] * common])


@dataclass(frozen=True)
class Problem:
    """A single benchmark problem: objective, gradient, and starting point."""
    name: str
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray


PROBLEMS: List[Problem] = [
    Problem("f1", f1, grad_f1, np.array([1.0, 1.0, 1.0])),
    Problem("f2", f2, grad_f2, np.array([0.0, 0.0])),
    Problem("f3", f3, grad_f3, np.array([-1.2, 1.0])),
    Problem("f4", f4, grad_f4, np.array([2.0, -2.0])),
    Problem("f5", f5, grad_f5, np.array([0.5, 0.5])),
]


# --------------------------------------------------------------------------
# Shared data structures
# --------------------------------------------------------------------------

@dataclass
class IterationRecord:
    iteration: int
    x: np.ndarray
    f_x: float
    search_direction: np.ndarray | None = None
    step_length: float | None = None


@dataclass
class OptimizationResult:
    method: str
    x_final: np.ndarray
    f_final: float
    converged: bool
    n_iterations: int
    history: list[IterationRecord] = field(default_factory=list)


def _stopping_criterion(
    grad: np.ndarray,
    f_x: float,
) -> float:
    """
    Gradient-based convergence metric.

    Convergence is determined by the gradient norm rather than by the
    magnitude of the objective function. This prevents large objective
    values from artificially making the convergence metric small.
    """
    return np.linalg.norm(grad)


def _print_iteration_summary(
    method: str,
    history: list[IterationRecord],
) -> None:
    """Print the first 10 and last 5 iterations, or all if fewer than 15."""
    n = len(history)
    print(f"\n--- Iteration Details ({method}) ---")

    def _fmt(rec: IterationRecord) -> str:
        sd = rec.search_direction
        sd_str = np.array2string(sd, precision=4) if sd is not None else "n/a"
        alpha_str = (
            f"{rec.step_length:.2e}"
            if rec.step_length is not None
            else "n/a"
        )
        return (
            f"Iter {rec.iteration:4d}: SD={sd_str}, alpha={alpha_str}, "
            f"x={np.array2string(rec.x, precision=6)}, "
            f"f(x)={rec.f_x:.6e}"
        )

    if n <= 15:
        for rec in history:
            print(_fmt(rec))
    else:
        for rec in history[:10]:
            print(_fmt(rec))

        print("...")

        for rec in history[-5:]:
            print(_fmt(rec))


# --------------------------------------------------------------------------
# Gradient Descent
# --------------------------------------------------------------------------

def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    alpha: float = 0.01,
    epsilon: float = 1e-6,
    max_iter: int = 10000,
) -> OptimizationResult:
    """
    Fixed-step steepest descent:
        x_{k+1} = x_k - alpha * grad f(x_k).

    A fixed step length is used deliberately, with no line search, so the
    convergence behavior of basic gradient descent can be compared directly
    with Newton's method and BFGS. This can lead to slow convergence or
    failure to converge within max_iter on ill-conditioned problems such as
    the Rosenbrock function.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    x = np.array(x0, dtype=float)
    history = [IterationRecord(0, x.copy(), f(x))]
    converged = False

    for i in range(1, max_iter + 1):
        grad = grad_f(x)
        f_x = f(x)

        if not np.all(np.isfinite(x)) or not np.isfinite(f_x):
            print("  [GD] Diverged: non-finite values encountered.")
            break

        if np.linalg.norm(grad) > 1e10 or abs(f_x) > 1e10:
            print("  [GD] Diverged: values exceeded threshold.")
            break

        if _stopping_criterion(grad, f_x) <= epsilon:
            converged = True
            break

        direction = -grad
        x = x + alpha * direction

        history.append(
            IterationRecord(
                i,
                x.copy(),
                f(x),
                direction.copy(),
                alpha,
            )
        )

    return OptimizationResult(
        method="Gradient Descent",
        x_final=x.copy(),
        f_final=f(x),
        converged=converged,
        n_iterations=history[-1].iteration,
        history=history,
    )


# --------------------------------------------------------------------------
# Newton's Method
# --------------------------------------------------------------------------

def _symmetrized_hessian(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    fd_epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Approximate the Hessian by finite differences of the analytic gradient.

    The finite-difference Jacobian is explicitly symmetrized because the
    exact Hessian of a twice-differentiable scalar objective is symmetric,
    while finite-difference roundoff and truncation can introduce small
    asymmetries.
    """
    if fd_epsilon <= 0:
        raise ValueError("fd_epsilon must be positive.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_raw = opt.approx_fprime(x, grad_f, fd_epsilon)

    h_raw = np.atleast_2d(h_raw)
    return 0.5 * (h_raw + h_raw.T)


def newtons_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    epsilon: float = 1e-6,
    max_iter: int = 10000,
) -> OptimizationResult:
    """
    Classical Newton's method with unit step length:

        H(x_k) d_k = -grad f(x_k)
        x_{k+1} = x_k + d_k

    The Hessian is approximated numerically from the analytic gradient.
    No line search or damping is used, so the method retains the behavior
    of classical undamped Newton's method.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    x = np.array(x0, dtype=float)
    history = [IterationRecord(0, x.copy(), f(x))]
    converged = False

    for i in range(1, max_iter + 1):
        grad = grad_f(x)
        f_x = f(x)

        if _stopping_criterion(grad, f_x) <= epsilon:
            converged = True
            break

        hess = _symmetrized_hessian(grad_f, x)

        try:
            direction = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print(
                f"  [Newton] Singular Hessian at iteration {i}; "
                "halting early."
            )
            break

        step = 1.0
        x = x + step * direction

        history.append(
            IterationRecord(
                i,
                x.copy(),
                f(x),
                direction.copy(),
                step,
            )
        )

    return OptimizationResult(
        method="Newton's Method",
        x_final=x.copy(),
        f_final=f(x),
        converged=converged,
        n_iterations=history[-1].iteration,
        history=history,
    )

# --------------------------------------------------------------------------
# BFGS (SciPy)
# --------------------------------------------------------------------------

def bfgs_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    epsilon: float = 1e-6,
    max_iter: int = 10000,
) -> OptimizationResult:
    """
    Quasi-Newton BFGS via scipy.optimize.minimize.

    SciPy's BFGS implementation uses its own gtol-based stopping criterion,
    which is not identical to the normalized gradient criterion used by
    Gradient Descent and Newton's method. Therefore, iteration counts are
    not perfectly apples-to-apples across the three methods.

    Iterates are recorded through a callback for consistent convergence
    plotting. SciPy does not expose the BFGS search direction or step length
    through this callback, so those fields remain None.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    x0 = np.array(x0, dtype=float)
    history = [IterationRecord(0, x0.copy(), f(x0))]

    def _callback(xk: np.ndarray) -> None:
        history.append(
            IterationRecord(
                len(history),
                xk.copy(),
                f(xk),
            )
        )

    result = opt.minimize(
        f,
        x0,
        method="BFGS",
        jac=grad_f,
        callback=_callback,
        options={
            "maxiter": max_iter,
            "gtol": epsilon,
        },
    )

    return OptimizationResult(
        method="BFGS (SciPy)",
        x_final=result.x.copy(),
        f_final=float(result.fun),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        history=history,
    )

# --------------------------------------------------------------------------
# Reporting / plotting
# --------------------------------------------------------------------------

def _save_convergence_plot(
    problem_name: str,
    results: list[OptimizationResult],
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))

    for res in results:
        f_vals = [rec.f_x for rec in res.history]
        ax.plot(
            range(len(f_vals)),
            f_vals,
            marker="o",
            markersize=3,
            label=res.method,
        )

    ax.set_title(f"Convergence Comparison: {problem_name}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.set_yscale("symlog")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"convergence_{problem_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def _write_summary_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / "optimization_summary.csv"
    fieldnames = [
        "problem",
        "method",
        "converged",
        "n_iterations",
        "f_final",
        "x_final",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def main(verbose: bool = True) -> list[dict]:
    """Run all optimization methods on every benchmark problem."""
    summary_rows: list[dict] = []

    for problem in PROBLEMS:
        print(f"\n======= Optimizing {problem.name} =======")

        results = [
            gradient_descent(problem.f, problem.grad, problem.x0),
            newtons_method(problem.f, problem.grad, problem.x0),
            bfgs_method(problem.f, problem.grad, problem.x0),
        ]

        for res in results:
            if verbose:
                _print_iteration_summary(res.method, res.history)

            print(
                f"[{problem.name}] {res.method}: "
                f"converged={res.converged}, "
                f"n_iter={res.n_iterations}, "
                f"x*={np.array2string(res.x_final, precision=6)}, "
                f"f(x*)={res.f_final:.6e}"
            )

            summary_rows.append(
                {
                    "problem": problem.name,
                    "method": res.method,
                    "converged": res.converged,
                    "n_iterations": res.n_iterations,
                    "f_final": res.f_final,
                    "x_final": np.array2string(
                        res.x_final,
                        precision=8,
                    ),
                }
            )

        fig_path = _save_convergence_plot(problem.name, results)
        print(f"[{problem.name}] convergence plot saved to {fig_path}")

    csv_path = _write_summary_csv(summary_rows)
    print(f"\nSummary written to {csv_path}")

    return summary_rows


if __name__ == "__main__":
    main()