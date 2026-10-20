"""
Constrained design optimization for a 3-DOF rotational torsional system.

The trained neural-network surrogate is used for fast design evaluation.
The selected design must subsequently be checked against the original
physics model.

Design variables:
    [J1, J2, J3, k_theta1, k_theta2, k_theta3]

Surrogate outputs:
    [omega1, omega2, omega3, theta2_max, theta3_max,
     alpha_max, theta_rms]

Objective:
    Minimize total rotational inertia.

Constraints:
    Natural frequencies must be no lower than baseline predictions.
    Response metrics must be no higher than baseline predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import (
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)

from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.surrogate import SurrogateMLP


INPUT_COLUMNS = (
    "J1",
    "J2",
    "J3",
    "k_theta1",
    "k_theta2",
    "k_theta3",
)

TARGET_COLUMNS = (
    "omega1",
    "omega2",
    "omega3",
    "theta2_max",
    "theta3_max",
    "alpha_max",
    "theta_rms",
)

FREQUENCY_COUNT = 3
DESIGN_DIM = 6
OUTPUT_DIM = 7
FEASIBILITY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class OptimizationOutcome:
    """Results from a surrogate-based design optimization."""

    baseline_design: NDArray[np.float64]
    baseline_outputs: NDArray[np.float64]
    optimized_design: NDArray[np.float64]
    optimized_outputs: NDArray[np.float64]
    solver_result: OptimizeResult | None
    surrogate_feasible: bool


def _get_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return validated lower and upper design-variable bounds."""
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)

    if lower.shape != (DESIGN_DIM,) or upper.shape != (DESIGN_DIM,):
        raise ValueError("Design bounds must each contain six values.")

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Design bounds must be finite.")

    if np.any(upper <= lower):
        raise ValueError("Every upper bound must exceed its lower bound.")

    return lower, upper


def scale_design(design: ArrayLike) -> NDArray[np.float64]:
    """Map a physical design to dimensionless coordinates in [0, 1]."""
    lower, upper = _get_bounds()
    design = np.asarray(design, dtype=float)

    if design.shape != (DESIGN_DIM,):
        raise ValueError("design must contain exactly six values.")

    if not np.all(np.isfinite(design)):
        raise ValueError("design must contain only finite values.")

    if np.any(design < lower) or np.any(design > upper):
        raise ValueError("design falls outside the configured bounds.")

    return (design - lower) / (upper - lower)


def unscale_design(
    scaled_design: ArrayLike,
) -> NDArray[np.float64]:
    """Map dimensionless coordinates in [0, 1] to physical variables."""
    lower, upper = _get_bounds()
    scaled_design = np.asarray(scaled_design, dtype=float)

    if scaled_design.shape != (DESIGN_DIM,):
        raise ValueError("scaled_design must contain exactly six values.")

    if not np.all(np.isfinite(scaled_design)):
        raise ValueError("scaled_design must contain only finite values.")

    if np.any(scaled_design < 0.0) or np.any(scaled_design > 1.0):
        raise ValueError("scaled_design must lie between zero and one.")

    return lower + scaled_design * (upper - lower)


def total_inertia(design: ArrayLike) -> float:
    """Return the sum of the three rotational inertias."""
    design = np.asarray(design, dtype=float)

    if design.shape != (DESIGN_DIM,):
        raise ValueError("design must contain exactly six values.")

    if not np.all(np.isfinite(design)):
        raise ValueError("design must contain only finite values.")

    if np.any(design[:3] <= 0.0):
        raise ValueError("Rotational inertias must be positive.")

    return float(np.sum(design[:3]))


def predict_design(
    model: SurrogateMLP,
    design: ArrayLike,
) -> NDArray[np.float64]:
    """Predict the seven physical outputs for one design."""
    design = np.asarray(design, dtype=float)

    if design.shape != (DESIGN_DIM,):
        raise ValueError("design must contain exactly six values.")

    if not np.all(np.isfinite(design)):
        raise ValueError("design must contain only finite values.")

    lower, upper = _get_bounds()

    if np.any(design < lower) or np.any(design > upper):
        raise ValueError("design falls outside the configured bounds.")

    model.eval()

    inputs = torch.as_tensor(
        np.array(design, dtype=np.float32, copy=True).reshape(1, -1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()[0]

    if predictions.shape != (OUTPUT_DIM,):
        raise RuntimeError("Surrogate returned an unexpected output shape.")

    predictions = np.asarray(predictions, dtype=float)

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Surrogate produced non-finite predictions.")

    return predictions


def constraint_margins(
    predicted_outputs: ArrayLike,
    baseline_outputs: ArrayLike,
) -> NDArray[np.float64]:
    """
    Return dimensionless inequality-constraint margins.

    SLSQP requires inequality constraints to be nonnegative.

    For natural frequencies:
        predicted / baseline - 1 >= 0

    For response metrics:
        1 - predicted / baseline >= 0
    """
    predicted = np.asarray(predicted_outputs, dtype=float)
    baseline = np.asarray(baseline_outputs, dtype=float)

    if predicted.shape != (OUTPUT_DIM,):
        raise ValueError("predicted_outputs must contain seven values.")

    if baseline.shape != (OUTPUT_DIM,):
        raise ValueError("baseline_outputs must contain seven values.")

    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted_outputs must be finite.")

    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline_outputs must be finite.")

    if np.any(baseline <= 0.0):
        raise ValueError("All baseline outputs must be positive.")

    frequency_margins = (
        predicted[:FREQUENCY_COUNT] / baseline[:FREQUENCY_COUNT] - 1.0
    )

    response_margins = (
        1.0
        - predicted[FREQUENCY_COUNT:] / baseline[FREQUENCY_COUNT:]
    )

    return np.concatenate((frequency_margins, response_margins))


def optimize_design(
    model: SurrogateMLP,
    baseline_design: ArrayLike | None = None,
    n_starts: int = 5,
    seed: int = 42,
    max_iterations: int = 5000,
    tolerance: float = 1e-9,
    solver: str = "SLSQP",
) -> OptimizationOutcome:
    """
    Minimize total inertia subject to baseline-relative constraints.

    Multiple starting points reduce dependence on one local search path.
    The baseline remains a valid fallback if no better feasible candidate
    is found.

    Note:
        Feasibility here refers to surrogate predictions, not the true
        physical simulation.
    """
    lower, upper = _get_bounds()

    if n_starts < 1:
        raise ValueError("n_starts must be at least one.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")

    solver = solver.upper()

    if solver not in {"SLSQP", "COBYQA"}:
        raise ValueError(
            f"Unsupported solver {solver!r}. "
            "Choose 'SLSQP' or 'COBYQA'."
        )

    if baseline_design is None:
        baseline = 0.5 * (lower + upper)
    else:
        baseline = np.asarray(baseline_design, dtype=float)

    # Validate the baseline and evaluate it once.
    baseline_scaled = scale_design(baseline)
    baseline_outputs = predict_design(model, baseline)

    if np.any(baseline_outputs <= 0.0):
        raise ValueError(
            "Baseline predictions must be positive for relative constraints."
        )

    # The baseline is feasible by construction: each output equals itself.
    best_design = baseline.copy()
    best_outputs = baseline_outputs.copy()
    best_result: OptimizeResult | None = None
    best_objective = total_inertia(baseline)
    best_is_feasible = True

    rng = np.random.default_rng(seed)
    starts = [baseline_scaled]

    for _ in range(n_starts - 1):
        starts.append(rng.uniform(0.0, 1.0, size=DESIGN_DIM))

    maximum_inertia = float(np.sum(upper[:3]))

    def objective(scaled: NDArray[np.float64]) -> float:
        design = unscale_design(scaled)
        return total_inertia(design) / maximum_inertia

    def constraints(scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        design = unscale_design(scaled)
        outputs = predict_design(model, design)
        return constraint_margins(outputs, baseline_outputs)

    
    for start_index, start in enumerate(starts, start=1):
        if solver == "SLSQP":
            solver_constraints = [
                {
                    "type": "ineq",
                    "fun": constraints,
                }
            ]

            options = {
                "maxiter": max_iterations,
                "ftol": tolerance,
                "disp": False,
            }

        else:
            solver_constraints = [
                NonlinearConstraint(
                    constraints,
                    lb=np.zeros(OUTPUT_DIM),
                    ub=np.full(OUTPUT_DIM, np.inf),
                )
            ]

            options = {
                "maxiter": max_iterations,
                "feasibility_tol": FEASIBILITY_TOLERANCE,
                "final_tr_radius": 1e-6,
                "disp": False,
            }

        result = minimize(
            objective,
            x0=start,
            method=solver,
            bounds=[(0.0, 1.0)] * DESIGN_DIM,
            constraints=solver_constraints,
            options=options,
        )

        scaled_candidate = np.asarray(result.x, dtype=np.float64)

        if (
            scaled_candidate.shape != (DESIGN_DIM,)
            or not np.all(np.isfinite(scaled_candidate))
        ):
            print(
                f"WARNING: {solver} returned an invalid scaled candidate: "
                f"{scaled_candidate}"
            )
            continue

        bound_violation = max(
            0.0,
            float(np.max(-scaled_candidate)),
            float(np.max(scaled_candidate - 1.0)),
        )

        if bound_violation > 1e-8:
            print(
                f"WARNING: {solver} returned a candidate outside "
                f"[0, 1]; maximum violation={bound_violation:.3e}; "
                f"x={scaled_candidate}"
            )
            continue

        # Correct only tiny floating-point excursions after checking them.
        scaled_candidate = np.clip(scaled_candidate, 0.0, 1.0)
        candidate = unscale_design(scaled_candidate)

        # Evaluate this candidate independently of the solver's status.
        candidate_outputs = predict_design(model, candidate)
        margins = constraint_margins(
            candidate_outputs,
            baseline_outputs,
        )

        feasible = bool(
            np.all(np.isfinite(margins))
            and np.all(margins >= -FEASIBILITY_TOLERANCE)
            and np.all(candidate >= lower)
            and np.all(candidate <= upper)
        )


        candidate_objective = total_inertia(candidate)

        print(
            f"Start {start_index}/{len(starts)}: "
            f"success={result.success}, "
            f"message={result.message!r}, "
            f"objective={candidate_objective:.8f}, "
            f"minimum_constraint_margin={np.min(margins):.8e}, "
            f"feasible={feasible}"
        )

        if feasible and candidate_objective < best_objective:
            best_design = candidate.copy()
            best_outputs = candidate_outputs.copy()
            best_objective = candidate_objective
            best_result = result
            best_is_feasible = True


    if best_result is None:
        print(
            "WARNING: No improved feasible candidate was accepted. "
            "Returning the baseline design."
        )
    else:
        print(f"Accepted solver success: {best_result.success}")
        print(f"Accepted solver message: {best_result.message}")
        print(f"Accepted objective: {best_result.fun:.8f}")


    final_margins = constraint_margins(
        best_outputs,
        baseline_outputs,
    )

    constraint_names = (
        "omega1_minimum",
        "omega2_minimum",
        "omega3_minimum",
        "theta2_maximum",
        "theta3_maximum",
        "alpha_maximum",
        "theta_rms_maximum",
    )

    print("\nACCEPTED DESIGN CONSTRAINT MARGINS")
    for name, margin in zip(constraint_names, final_margins):
        print(f"{name}: {margin:+.8e}")

    print(
        "Minimum margin: "
        f"{np.min(final_margins):+.8e}"
    )
    print(
        "Feasibility tolerance: "
        f"{FEASIBILITY_TOLERANCE:.8e}"
    )


    return OptimizationOutcome(
        baseline_design=baseline.copy(),
        baseline_outputs=baseline_outputs.copy(),
        optimized_design=best_design.copy(),
        optimized_outputs=best_outputs.copy(),
        solver_result=best_result,
        surrogate_feasible=best_is_feasible,
    )