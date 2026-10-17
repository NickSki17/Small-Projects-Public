
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
from src.optimization import optimize_design
from src.surrogate_inference import load_surrogate


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
    """Optimize the surrogate, verify the design, and export results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading surrogate from: {model_dir.resolve()}")
    inference = load_surrogate(model_dir)
    model = InferenceModelAdapter(inference)

    # Preserve the existing objective, constraints, bounds, and
    # multistart SLSQP implementation.
    outcome = optimize_design(
        model,
        n_starts=n_starts,
        seed=seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    baseline_design = np.asarray(
        outcome.baseline_design, dtype=np.float64
    ).reshape(-1)

    optimized_design = np.asarray(
        outcome.optimized_design, dtype=np.float64
    ).reshape(-1)

    baseline_surrogate = np.asarray(
        outcome.baseline_outputs, dtype=np.float64
    ).reshape(-1)

    optimized_surrogate = np.asarray(
        outcome.optimized_outputs, dtype=np.float64
    ).reshape(-1)

    if baseline_design.shape != (6,) or optimized_design.shape != (6,):
        raise ValueError("Optimizer returned an invalid design shape.")

    # Independent physics evaluations; neither result comes from
    # the surrogate or from the optimizer's predicted outputs.
    baseline_physics = physics_outputs(baseline_design)
    optimized_physics = physics_outputs(optimized_design)

    baseline_comparison = compare_outputs(
        baseline_surrogate, baseline_physics
    )
    optimized_comparison = compare_outputs(
        optimized_surrogate, optimized_physics
    )

    baseline_comparison.insert(0, "design", "baseline")
    optimized_comparison.insert(0, "design", "optimized")

    comparison = pd.concat(
        [baseline_comparison, optimized_comparison],
        ignore_index=True,
    )

    comparison.to_csv(
        OUTPUT_DIR / "physics_verification.csv",
        index=False,
    )

    design_results = pd.DataFrame(
        {
            "variable": INPUT_COLUMNS,
            "baseline_design": baseline_design,
            "optimized_design": optimized_design,
            "change": optimized_design - baseline_design,
        }
    )
    design_results.to_csv(
        OUTPUT_DIR / "design_comparison.csv",
        index=False,
    )

    solver_result = outcome.solver_result

    summary = {
        "model_directory": str(model_dir),
        "objective": "minimize_total_inertia",
        "optimization": {
            "solver_success": (
                bool(solver_result.success)
                if solver_result is not None
                else False
            ),
            "solver_message": (
                str(solver_result.message)
                if solver_result is not None
                else "No solver result returned."
            ),
            "surrogate_feasible": bool(outcome.surrogate_feasible),
            "n_starts": n_starts,
            "seed": seed,
        },
        "baseline": {
            "design": dict(zip(INPUT_COLUMNS, baseline_design.tolist())),
            "surrogate_outputs": dict(
                zip(TARGET_COLUMNS, baseline_surrogate.tolist())
            ),
            "physics_outputs": dict(
                zip(TARGET_COLUMNS, baseline_physics.tolist())
            ),
        },
        "optimized": {
            "design": dict(zip(INPUT_COLUMNS, optimized_design.tolist())),
            "surrogate_outputs": dict(
                zip(TARGET_COLUMNS, optimized_surrogate.tolist())
            ),
            "physics_outputs": dict(
                zip(TARGET_COLUMNS, optimized_physics.tolist())
            ),
        },
        "verification": {
            "max_absolute_error_baseline": float(
                np.max(np.abs(baseline_surrogate - baseline_physics))
            ),
            "max_absolute_error_optimized": float(
                np.max(np.abs(optimized_surrogate - optimized_physics))
            ),
            "physics_evaluated": True,
            # Physics feasibility must be checked separately against
            # the original design constraints.
            "physics_feasibility_verified": False,
        },
    }

    (OUTPUT_DIR / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    
    print("\nOPTIMIZER DIAGNOSTICS")

    if solver_result is None:
        print("Accepted solver result: None")
        print("The baseline design was retained.")
    else:
        print(f"Accepted solver success: {solver_result.success}")
        print(f"Accepted solver message: {solver_result.message}")
        print(f"Accepted solver iterations: {solver_result.nit}")
        print(
            "Accepted solver function evaluations: "
            f"{solver_result.nfev}"
        )
        print(
            "Accepted solver objective: "
            f"{solver_result.fun:.8f}"
        )
        print(
            "Accepted solver candidate (scaled): "
            f"{solver_result.x}"
        )

    print(f"Surrogate feasible: {outcome.surrogate_feasible}")
    print(f"Accepted design (physical units): {optimized_design}")
    print(f"Baseline design (physical units): {baseline_design}")

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
