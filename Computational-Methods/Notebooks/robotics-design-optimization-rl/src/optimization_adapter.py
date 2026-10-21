
"""Bridge surrogate inference, existing optimization, and physics verification.

Run from the repository root:
    python -m src.optimization_adapter

The existing SLSQP implementation remains in src/optimization.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.generate_dataset import evaluate_design
from src.optimization import (
    FEASIBILITY_TOLERANCE,
    SLSQP_FINITE_DIFFERENCE_STEP,
    optimize_design,
    constraint_margins,
    load_validation_allowances,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts/surrogate_benchmark"
DEFAULT_DATASET = PROJECT_ROOT / "data/full_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/surrogate_optimization"
PHYSICS_FEASIBILITY_TOLERANCE = 1e-6
CONSTRAINT_NAMES = (
    "omega1_minimum", "omega2_minimum", "omega3_minimum",
    "theta2_maximum", "theta3_maximum", "alpha_maximum", "theta_rms_maximum",
)


def classify_candidate(design, surrogate_margins, physics_margins, lower, upper,
                       baseline_inertia, solver_success=False):
    """Classify independently of solver termination; margins are physical units."""
    design = np.asarray(design, dtype=float).reshape(-1)
    surrogate_margins = np.asarray(surrogate_margins, dtype=float).reshape(-1)
    physics_margins = np.asarray(physics_margins, dtype=float).reshape(-1)
    lower, upper = np.asarray(lower), np.asarray(upper)
    finite = (design.shape == lower.shape and surrogate_margins.shape == (7,)
              and physics_margins.shape == (7,) and np.all(np.isfinite(design))
              and np.all(np.isfinite(surrogate_margins)) and np.all(np.isfinite(physics_margins)))
    in_bounds = bool(finite and np.all(design >= lower) and np.all(design <= upper))
    surrogate_feasible = bool(finite and np.all(surrogate_margins >= -FEASIBILITY_TOLERANCE))
    physics_feasible = bool(finite and np.all(physics_margins >= -PHYSICS_FEASIBILITY_TOLERANCE))
    inertia = float(np.sum(design[:3])) if finite else None
    improves = bool(inertia is not None and inertia < baseline_inertia - 1e-12)
    accepted = bool(in_bounds and surrogate_feasible and physics_feasible and improves)
    reasons = []
    if not finite: reasons.append("invalid_or_nonfinite_candidate")
    if finite and not in_bounds: reasons.append("out_of_bounds")
    if finite and not surrogate_feasible: reasons.append("surrogate_screen_failed")
    if finite and not physics_feasible: reasons.append("independent_physics_failed")
    if finite and not improves: reasons.append("does_not_improve_baseline")
    return {"in_bounds": in_bounds, "surrogate_feasible": surrogate_feasible,
            "physics_feasible": physics_feasible, "improves_baseline": improves,
            "accepted_for_selection": accepted, "rejection_reasons": reasons,
            "solver_success": bool(solver_success), "total_inertia": inertia}


def select_candidate(records):
    """Return the accepted candidate name with minimum physical total inertia."""
    eligible = [(name, record["total_inertia"]) for name, record in records.items()
                if record.get("accepted_for_selection")]
    return min(eligible, key=lambda item: item[1])[0] if eligible else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_output_dir():
    path = OUTPUT_DIR / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S.%fZ")
    path.mkdir(parents=True, exist_ok=False)
    return path


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
    calibration_dir: Path | None = None,
    dataset_path: Path = DEFAULT_DATASET,
    n_starts: int = 5,
    seed: int = 42,
    max_iterations: int = 50000,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Run SLSQP and COBYQA, verify candidates, and select the best."""

    if calibration_dir is None:
        raise FileNotFoundError("Validation calibration is required. Run `python -m scripts.diagnose_validation_region` and pass its run directory with --calibration-dir.")
    def project_path(value: Path) -> Path:
        value = Path(value)
        return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()

    model_dir = project_path(model_dir)
    dataset_path = project_path(dataset_path)
    calibration_dir = project_path(calibration_dir)

    print(f"Loading surrogate from: {model_dir.resolve()}")
    inference = load_surrogate(model_dir)
    model = InferenceModelAdapter(inference)

    solver_names = ("SLSQP", "COBYQA")
    outcomes = {}
    solver_runs = {}
    candidate_records = {}

    global_allowances, low_allowances, low_inertia_cutoff = load_validation_allowances(
        calibration_dir, model_dir, dataset_path
    )
    lower = np.asarray(LOWER_BOUNDS, dtype=np.float64)
    upper = np.asarray(UPPER_BOUNDS, dtype=np.float64)
    baseline_design_for_run = 0.5 * (lower + upper)
    baseline_physics_for_run = physics_outputs(baseline_design_for_run)
    output_dir = _fresh_output_dir()

    # Use identical configuration and seed for both solvers.
    for solver in solver_names:
        print(f"\n{'=' * 60}")
        print(f"RUNNING {solver}")
        print(f"{'=' * 60}")

        try:
            outcome = optimize_design(
                model,
                baseline_design=baseline_design_for_run,
                n_starts=n_starts,
                seed=seed,
                max_iterations=max_iterations,
                tolerance=tolerance,
                solver=solver,
                validation_allowances=global_allowances,
                low_inertia_allowances=low_allowances,
                low_inertia_cutoff=low_inertia_cutoff,
                constraint_reference_outputs=baseline_physics_for_run,
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

    baseline_physics = baseline_physics_for_run
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
      candidates = outcome.candidates or ({"design": outcome.optimized_design,
          "outputs": outcome.optimized_outputs, "result": outcome.solver_result,
          "surrogate_feasible": outcome.surrogate_feasible,
          "margins": np.zeros(7)},)
      for candidate_index, candidate_record in enumerate(candidates, start=1):
        design = np.asarray(candidate_record["design"], dtype=np.float64).reshape(-1)
        surrogate_outputs = np.asarray(candidate_record["outputs"], dtype=np.float64).reshape(-1)

        # Reject malformed/out-of-bound candidates before invoking physics.
        if design.shape != lower.shape or not np.all(np.isfinite(design)) or np.any(design < lower) or np.any(design > upper):
            solver_runs[solver].setdefault("candidate_records", []).append({
                "candidate": f"{solver}_start_{candidate_index}",
                "solver_success": bool(getattr(candidate_record.get("result"), "success", False)),
                "in_bounds": False, "physics_feasible": False,
                "surrogate_feasible": bool(candidate_record.get("surrogate_feasible", False)),
                "accepted_for_selection": False, "rejection_reason": "invalid_or_out_of_bounds",
            })
            continue

        actual_outputs = physics_outputs(design)

        # Compare actual physics outputs against the baseline physics
        # outputs, rather than against baseline surrogate predictions.
        physics_margins = constraint_margins(
            actual_outputs, baseline_physics
        )

        inertia = float(np.sum(design[:3]))
        result = candidate_record.get("result")
        candidate_name = f"{solver}_start_{candidate_index}"
        classification = classify_candidate(
            design, candidate_record["margins"], physics_margins, lower, upper,
            baseline_inertia, getattr(result, "success", False),
        )
        in_bounds = classification["in_bounds"]
        physics_feasible = classification["physics_feasible"]
        improves_baseline = classification["improves_baseline"]
        accepted = classification["accepted_for_selection"]

        solver_runs[solver].setdefault("candidate_records", []).append({
            "candidate": candidate_name,
            "solver_success": bool(result.success) if result is not None else False,
            "solver_termination_status": str(result.message) if result is not None else "no solver result",
            "solver_iterations": int(result.nit) if result is not None and getattr(result, "nit", None) is not None else None,
            "solver_function_evaluations": int(result.nfev) if result is not None and getattr(result, "nfev", None) is not None else None,
            "solver_objective_normalized": float(result.fun) if result is not None and np.isfinite(result.fun) else None,
            "design": dict(zip(INPUT_COLUMNS, design.tolist())),
            "total_inertia": inertia,
            "surrogate_predictions": dict(zip(TARGET_COLUMNS, candidate_record["outputs"].tolist())),
            "validation_allowances": dict(zip(TARGET_COLUMNS, candidate_record["allowances"].tolist())),
            "surrogate_margins": dict(zip(CONSTRAINT_NAMES, candidate_record["margins"].tolist())),
            "physics_margins": dict(zip(CONSTRAINT_NAMES, physics_margins.tolist())),
            "minimum_physics_margin": float(np.min(physics_margins)),
            "in_bounds": in_bounds,
            "surrogate_feasible": classification["surrogate_feasible"],
            "physics_feasible": physics_feasible,
            "improves_baseline": improves_baseline,
            "accepted_for_selection": accepted,
            "rejection_reason": None if accepted else ";".join(classification["rejection_reasons"]),
        })

        candidate_records[candidate_name] = {
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
        comparison.insert(0, "design", candidate_name)
        all_comparisons.append(comparison)

        print(f"\n{candidate_name} verification:")
        print(f"  Total inertia: {inertia:.10f}")
        print(f"  In bounds: {in_bounds}")
        print(f"  Surrogate feasible: {candidate_record['surrogate_feasible']}")
        print(f"  Physics feasible: {physics_feasible}")
        print(f"  Minimum physics margin: {np.min(physics_margins):+.8e}")
        physics_constraint_names = (
            "omega1_minimum",
            "omega2_minimum",
            "omega3_minimum",
            "theta2_maximum",
            "theta3_maximum",
            "alpha_maximum",
            "theta_rms_maximum",
        )

        print("  Individual physics margins:")
        for name, margin in zip(
            physics_constraint_names,
            physics_margins,
        ):
            print(f"    {name}: {margin:+.8e}")
        print(f"  Improves baseline: {improves_baseline}")
        print(f"  Accepted for selection: {accepted}")

        if accepted:
            feasible_candidates.append(candidate_name)

    # Select the lowest-inertia candidate that passes BOTH surrogate
    # feasibility and independent physics feasibility.
    if feasible_candidates:
        selected_solver = min(feasible_candidates, key=lambda name: candidate_records[name]["inertia"])
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
        output_dir / "physics_verification.csv",
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
        output_dir / "design_comparison.csv",
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
            "slsqp_finite_difference_step_scaled": SLSQP_FINITE_DIFFERENCE_STEP,
            "physics_feasibility_tolerance": (
                PHYSICS_FEASIBILITY_TOLERANCE
            ),
            "validation_allowances_empirical": dict(zip(TARGET_COLUMNS, global_allowances.tolist())),
            "low_inertia_allowances_empirical": dict(zip(TARGET_COLUMNS, low_allowances.tolist())),
            "low_inertia_cutoff_total_inertia": low_inertia_cutoff,
            "allowance_method": "directional maximum error on existing validation partition; finite-sample empirical safeguard, not a guarantee",
            "low_inertia_definition": "J1+J2+J3 <= validation 10th percentile cutoff",
            "surrogate_feasibility_tolerance": FEASIBILITY_TOLERANCE,
        },
        "solver_runs": solver_runs,
        "candidates": {
            solver: details.get("candidate_records", [])
            for solver, details in solver_runs.items()
        },
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

    summary["run_directory"] = str(output_dir.resolve())
    (output_dir / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    versions = {}
    for package in ("numpy", "scipy", "pandas", "torch", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "inputs": {"dataset_path": str(dataset_path), "model_directory": str(model_dir),
            "calibration_directory": str(calibration_dir)},
        "dataset_sha256": _sha256(dataset_path) if dataset_path.is_file() else None,
        "checkpoint_sha256": _sha256(model_dir / "selected_model_state.pt") if (model_dir / "selected_model_state.pt").is_file() else None,
        "model_metadata_sha256": _sha256(model_dir / "selected_model_metadata.json") if (model_dir / "selected_model_metadata.json").is_file() else None,
        "calibration_metadata_sha256": _sha256(calibration_dir / "diagnostic_metadata.json"),
        "calibration_summary_sha256": _sha256(calibration_dir / "validation_error_summary.csv"),
        "calibration_directions_sha256": _sha256(calibration_dir / "validation_constraint_direction.csv"),
        "source_revision": revision, "source_tree_dirty": dirty,
        "python": __import__("sys").version, "dependencies": versions,
        "configuration": {"solver_methods": list(solver_names), "n_starts": n_starts,
            "seed": seed, "max_iterations": max_iterations, "tolerance": tolerance,
            "slsqp_finite_difference_step_scaled": SLSQP_FINITE_DIFFERENCE_STEP,
            "lower_bounds": lower.tolist(), "upper_bounds": upper.tolist()},
        "baseline": {"design": baseline_design.tolist(), "total_inertia_kg_m2": baseline_inertia},
        "selected": {"name": selected_solver, "design": selected_design.tolist(),
            "total_inertia_kg_m2": selected["inertia"], "acceptance_reason": selection_reason},
        "outputs_sha256": {name: _sha256(output_dir / name) for name in
            ("physics_verification.csv", "design_comparison.csv", "optimization_summary.json")},
        "unknowns": ["full per-evaluation solver trace", "historical provenance of pre-existing model artifacts"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("FINAL SELECTION")
    print(f"{'=' * 60}")
    print(f"Selected: {selected_solver}")
    print(f"Reason: {selection_reason}")
    print(f"Baseline inertia: {baseline_inertia:.10f}")
    print(f"Selected inertia: {selected['inertia']:.10f}")
    print(f"Selected design: {selected_design}")
    print(f"Results saved to: {output_dir.resolve()}")

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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--calibration-dir", type=Path, required=True,
                        help="New validation-only diagnostic run directory; required, no weaker fallback.")
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=50000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    if args.n_starts < 1:
        parser.error("--n-starts must be at least 1.")

    run_optimization(
        model_dir=args.model_dir,
        calibration_dir=args.calibration_dir,
        dataset_path=args.dataset,
        n_starts=args.n_starts,
        seed=args.seed,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
