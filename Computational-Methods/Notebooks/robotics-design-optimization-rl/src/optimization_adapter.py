
"""Bridge surrogate inference, existing optimization, and physics verification.

Run from the repository root:
    python -m src.optimization_adapter

The existing SLSQP implementation remains in src/optimization.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.generate_dataset import evaluate_design
from src.optimization import optimize_design, constraint_margins
from src.surrogate_inference import load_surrogate
from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS

INPUT_COLUMNS = [
    "J1",
    "J2",
    "J3",
    "k_theta1",
    "k_theta2",
    "k_theta3",
]

TARGET_COLUMNS = [
    "omega1",
    "omega2",
    "omega3",
    "theta2_max",
    "theta3_max",
    "alpha_max",
    "theta_rms",
]

DEFAULT_MODEL_DIR = Path("artifacts/surrogate_benchmark")
OUTPUT_DIR = Path("artifacts/surrogate_optimization")
PHYSICS_FEASIBILITY_TOLERANCE = 1e-6


class InferenceModelAdapter:
    """Expose the inference interface as a PyTorch-compatible model.

    src.optimization.py expects a model with eval() and a callable
    that accepts a tensor and returns a tensor. This adapter provides
    that interface without changing the existing optimizer.
    """

    def __init__(self, inference: Any) -> None:
        if not callable(getattr(inference, "predict_array", None)):
            raise TypeError(
                "The loaded inference object must expose predict_array()."
            )
        self.inference = inference

    def eval(self) -> InferenceModelAdapter:
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

        device = inputs.device
        array = inputs.detach().cpu().numpy()

        predictions = np.asarray(
            self.inference.predict_array(array),
            dtype=np.float64,
        )

        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        if predictions.shape != (array.shape[0], len(TARGET_COLUMNS)):
            raise ValueError(
                "Unexpected surrogate output shape: "
                f"{predictions.shape}; expected "
                f"({array.shape[0]}, {len(TARGET_COLUMNS)})."
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Surrogate produced non-finite predictions.")

        return torch.as_tensor(
            predictions,
            dtype=torch.float32,
            device=device,
        )


def physics_outputs(design: np.ndarray) -> np.ndarray:
    """Evaluate the original physics model in surrogate target order."""

    design = np.asarray(design, dtype=np.float64)

    if design.shape != (len(INPUT_COLUMNS),):
        raise ValueError(
            f"Expected a design with shape (6,), got {design.shape}."
        )

    if not np.all(np.isfinite(design)):
        raise ValueError("Design contains non-finite values.")

    result = evaluate_design(design)

    if not isinstance(result, dict):
        raise TypeError(
            "evaluate_design() must return a dictionary of physical metrics."
        )

    missing = [name for name in TARGET_COLUMNS if name not in result]
    if missing:
        raise KeyError(
            "Physics evaluation is missing required surrogate targets: "
            f"{missing}. Available keys: {list(result)}"
        )

    outputs = np.asarray(
        [result[name] for name in TARGET_COLUMNS],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(outputs)):
        raise ValueError("Physics evaluation produced non-finite outputs.")

    return outputs


def compare_outputs(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> pd.DataFrame:
    """Compare predictions with physics outputs, target by target."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)

    if predicted.shape != (len(TARGET_COLUMNS),):
        raise ValueError("Unexpected prediction vector shape.")

    if actual.shape != (len(TARGET_COLUMNS),):
        raise ValueError("Unexpected physics output vector shape.")

    absolute_error = np.abs(predicted - actual)

    # Relative error is undefined when the reference value is zero.
    relative_error_percent = np.full_like(actual, np.nan)
    nonzero = np.abs(actual) > np.finfo(np.float64).eps
    relative_error_percent[nonzero] = (
        100.0 * absolute_error[nonzero] / np.abs(actual[nonzero])
    )

    return pd.DataFrame(
        {
            "target": TARGET_COLUMNS,
            "surrogate_prediction": predicted,
            "physics_result": actual,
            "absolute_error": absolute_error,
            "relative_error_percent": relative_error_percent,
        }
    )



def run_optimization(
    model_dir: Path = DEFAULT_MODEL_DIR,
    n_starts: int = 5,
    seed: int = 42,
    max_iterations: int = 5000,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Run SLSQP and COBYQA, verify candidates, and select the best."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading surrogate from: {model_dir.resolve()}")
    inference = load_surrogate(model_dir)
    model = InferenceModelAdapter(inference)

    solver_names = ("SLSQP", "COBYQA")
    outcomes = {}
    solver_runs = {}
    candidate_records = {}

    # Use identical configuration and seed for both solvers.
    for solver in solver_names:
        print(f"\n{'=' * 60}")
        print(f"RUNNING {solver}")
        print(f"{'=' * 60}")

        try:
            outcome = optimize_design(
                model,
                n_starts=n_starts,
                seed=seed,
                max_iterations=max_iterations,
                tolerance=tolerance,
                solver=solver,
            )

            outcomes[solver] = outcome
            result = outcome.solver_result

            solver_runs[solver] = {
                "completed": True,
                "solver_success": (
                    bool(result.success) if result is not None else False
                ),
                "solver_message": (
                    str(result.message)
                    if result is not None
                    else "No improved surrogate-feasible candidate accepted."
                ),
                "iterations": (
                    int(result.nit) if result is not None else None
                ),
                "function_evaluations": (
                    int(result.nfev) if result is not None else None
                ),
                "surrogate_feasible": bool(outcome.surrogate_feasible),
            }

        except Exception as exc:
            # Allow the other solver to run if one method fails.
            solver_runs[solver] = {
                "completed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"{solver} failed: {type(exc).__name__}: {exc}")

    if not outcomes:
        raise RuntimeError(
            "Both optimization solvers failed. "
            "No candidate could be evaluated."
        )

    # Both solvers use the same baseline definition. Obtain the baseline
    # and evaluate it independently with the original physics model.
    first_outcome = next(iter(outcomes.values()))

    baseline_design = np.asarray(
        first_outcome.baseline_design, dtype=np.float64
    ).reshape(-1)

    baseline_surrogate = np.asarray(
        first_outcome.baseline_outputs, dtype=np.float64
    ).reshape(-1)

    baseline_physics = physics_outputs(baseline_design)
    baseline_inertia = float(np.sum(baseline_design[:3]))

    baseline_comparison = compare_outputs(
        baseline_surrogate, baseline_physics
    )
    baseline_comparison.insert(0, "design", "baseline")

    all_comparisons = [baseline_comparison]
    candidate_records["baseline"] = {
        "design": baseline_design.copy(),
        "surrogate_outputs": baseline_surrogate.copy(),
        "physics_outputs": baseline_physics.copy(),
        "physics_margins": np.zeros(len(TARGET_COLUMNS)),
        "inertia": baseline_inertia,
        "physics_feasible": True,
        "improves_baseline": False,
    }

    # Verify each solver's candidate against the ORIGINAL physics model.
    lower = np.asarray(LOWER_BOUNDS, dtype=np.float64)
    upper = np.asarray(UPPER_BOUNDS, dtype=np.float64)

    feasible_candidates = []

    for solver, outcome in outcomes.items():
        design = np.asarray(
            outcome.optimized_design, dtype=np.float64
        ).reshape(-1)

        surrogate_outputs = np.asarray(
            outcome.optimized_outputs, dtype=np.float64
        ).reshape(-1)

        actual_outputs = physics_outputs(design)

        # Compare actual physics outputs against the baseline physics
        # outputs, rather than against baseline surrogate predictions.
        physics_margins = constraint_margins(
            actual_outputs, baseline_physics
        )

        in_bounds = bool(
            np.all(design >= lower) and np.all(design <= upper)
        )

        physics_feasible = bool(
            in_bounds
            and np.all(
                physics_margins >= -PHYSICS_FEASIBILITY_TOLERANCE
            )
        )

        inertia = float(np.sum(design[:3]))
        improves_baseline = inertia < baseline_inertia - 1e-12

        accepted = bool(
            outcome.surrogate_feasible
            and physics_feasible
            and improves_baseline
        )

        result = outcome.solver_result

        solver_runs[solver].update({
            "design": dict(zip(INPUT_COLUMNS, design.tolist())),
            "total_inertia": inertia,
            "minimum_physics_margin": float(np.min(physics_margins)),
            "in_bounds": in_bounds,
            "physics_feasible": physics_feasible,
            "improves_baseline": improves_baseline,
            "accepted_for_selection": accepted,
            "solver_success": (
                bool(result.success) if result is not None else False
            ),
            "solver_message": (
                str(result.message)
                if result is not None
                else "No improved surrogate-feasible candidate accepted."
            ),
        })

        candidate_records[solver] = {
            "design": design,
            "surrogate_outputs": surrogate_outputs,
            "physics_outputs": actual_outputs,
            "physics_margins": physics_margins,
            "inertia": inertia,
            "physics_feasible": physics_feasible,
            "improves_baseline": improves_baseline,
            "accepted": accepted,
        }

        comparison = compare_outputs(surrogate_outputs, actual_outputs)
        comparison.insert(0, "design", solver)
        all_comparisons.append(comparison)

        print(f"\n{solver} verification:")
        print(f"  Total inertia: {inertia:.10f}")
        print(f"  In bounds: {in_bounds}")
        print(f"  Surrogate feasible: {outcome.surrogate_feasible}")
        print(f"  Physics feasible: {physics_feasible}")
        print(f"  Minimum physics margin: {np.min(physics_margins):+.8e}")
        print(f"  Improves baseline: {improves_baseline}")
        print(f"  Accepted for selection: {accepted}")

        if accepted:
            feasible_candidates.append(solver)

    # Select the lowest-inertia candidate that passes BOTH surrogate
    # feasibility and independent physics feasibility.
    if feasible_candidates:
        selected_solver = min(
            feasible_candidates,
            key=lambda name: candidate_records[name]["inertia"],
        )
        selected = candidate_records[selected_solver]
        selection_reason = (
            "Lowest-inertia candidate passing surrogate and physics checks."
        )
    else:
        selected_solver = "BASELINE"
        selected = candidate_records["baseline"]
        selection_reason = (
            "Neither solver produced a lower-inertia candidate "
            "that passed all acceptance checks; retained baseline."
        )

    selected_design = selected["design"]
    selected_surrogate = selected["surrogate_outputs"]
    selected_physics = selected["physics_outputs"]

    # Export comparisons for baseline and both solver candidates.
    pd.concat(all_comparisons, ignore_index=True).to_csv(
        OUTPUT_DIR / "physics_verification.csv",
        index=False,
    )

    design_rows = []
    for name, record in candidate_records.items():
        design = record["design"]
        design_rows.append({
            "candidate": name,
            **dict(zip(INPUT_COLUMNS, design.tolist())),
            "total_inertia": record["inertia"],
            "physics_feasible": record["physics_feasible"],
            "selected": name == selected_solver,
        })

    pd.DataFrame(design_rows).to_csv(
        OUTPUT_DIR / "design_comparison.csv",
        index=False,
    )

    summary = {
        "model_directory": str(model_dir),
        "objective": "minimize_total_inertia",
        "configuration": {
            "n_starts": n_starts,
            "seed": seed,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "physics_feasibility_tolerance": (
                PHYSICS_FEASIBILITY_TOLERANCE
            ),
        },
        "solver_runs": solver_runs,
        "selection": {
            "selected_solver": selected_solver,
            "reason": selection_reason,
            "total_inertia": selected["inertia"],
            "physics_feasible": selected["physics_feasible"],
            "improves_baseline": selected["improves_baseline"],
        },
        "baseline": {
            "design": dict(zip(INPUT_COLUMNS, baseline_design.tolist())),
            "total_inertia": baseline_inertia,
            "surrogate_outputs": dict(
                zip(TARGET_COLUMNS, baseline_surrogate.tolist())
            ),
            "physics_outputs": dict(
                zip(TARGET_COLUMNS, baseline_physics.tolist())
            ),
        },
        "selected_design": {
            "design": dict(zip(INPUT_COLUMNS, selected_design.tolist())),
            "surrogate_outputs": dict(
                zip(TARGET_COLUMNS, selected_surrogate.tolist())
            ),
            "physics_outputs": dict(
                zip(TARGET_COLUMNS, selected_physics.tolist())
            ),
            "physics_margins": dict(
                zip(
                    (
                        "omega1_minimum",
                        "omega2_minimum",
                        "omega3_minimum",
                        "theta2_maximum",
                        "theta3_maximum",
                        "alpha_maximum",
                        "theta_rms_maximum",
                    ),
                    selected["physics_margins"].tolist(),
                )
            ),
        },
    }

    (OUTPUT_DIR / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("FINAL SELECTION")
    print(f"{'=' * 60}")
    print(f"Selected: {selected_solver}")
    print(f"Reason: {selection_reason}")
    print(f"Baseline inertia: {baseline_inertia:.10f}")
    print(f"Selected inertia: {selected['inertia']:.10f}")
    print(f"Selected design: {selected_design}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")

    return summary



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize a surrogate and verify its candidate with physics."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory accepted by src.surrogate_inference.load_surrogate.",
    )
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    if args.n_starts < 1:
        parser.error("--n-starts must be at least 1.")

    run_optimization(
        model_dir=args.model_dir,
        n_starts=args.n_starts,
        seed=args.seed,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
