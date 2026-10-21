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
import hashlib
import json
from pathlib import Path
import pandas as pd

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
# The surrogate inference path evaluates the network in float32.  SLSQP's
# default absolute difference step is too small in normalized design space
# and can quantize constraint differences to zero.  This value was checked
# against forward/central differences and controlled starts on the saved
# baseline case; it is an absolute step in the [0, 1]-scaled coordinates.
SLSQP_FINITE_DIFFERENCE_STEP = 1e-5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OptimizationOutcome:
    """Results from a surrogate-based design optimization."""

    baseline_design: NDArray[np.float64]
    baseline_outputs: NDArray[np.float64]
    optimized_design: NDArray[np.float64]
    optimized_outputs: NDArray[np.float64]
    solver_result: OptimizeResult | None
    surrogate_feasible: bool
    candidates: tuple[dict, ...] = ()


def load_validation_allowances(
    calibration_dir: Path,
    model_dir: Path,
    dataset_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Load empirical margins only when model/data/split provenance matches.

    These finite-sample maxima are safeguards, not statistical guarantees.
    Low-inertia means total inertia at or below the validation 10th-percentile
    cutoff recorded by the diagnostic.
    """
    def project_path(value: Path) -> Path:
        value = Path(value)
        return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()

    calibration_dir = project_path(calibration_dir)
    model_dir = project_path(model_dir)
    dataset_path = project_path(dataset_path)
    summary_path = calibration_dir / "validation_error_summary.csv"
    metadata_path = calibration_dir / "diagnostic_metadata.json"
    direction_path = calibration_dir / "validation_constraint_direction.csv"
    model_metadata_path = model_dir / "selected_model_metadata.json"
    required_paths = (dataset_path, summary_path, metadata_path, direction_path, model_metadata_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Required optimization inputs are missing: " + ", ".join(missing_paths)
            + ". Generate the dataset, selected model, and validation calibration in the documented order."
        )
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    directions = pd.read_csv(direction_path)

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    if metadata.get("test_partition_evaluated") is not False:
        raise ValueError("Calibration metadata must confirm test-set isolation.")
    if metadata.get("error_sign_convention") != "prediction - actual":
        raise ValueError("Calibration error sign convention is missing or incompatible.")
    if list(metadata.get("target_columns", [])) != list(TARGET_COLUMNS):
        raise ValueError("Calibration target ordering does not match optimization.")
    if metadata.get("inertia_definition") != "J1 + J2 + J3":
        raise ValueError("Calibration inertia definition is incompatible.")
    if metadata.get("model") != model_metadata.get("model") or metadata.get("seed") != model_metadata.get("seed"):
        raise ValueError("Calibration model identity/seed does not match the loaded model.")
    if model_metadata.get("dataset_sha256") and model_metadata["dataset_sha256"] != digest(dataset_path):
        raise ValueError("Selected model metadata refers to a different dataset fingerprint.")
    if model_metadata.get("checkpoint_sha256") and model_metadata["checkpoint_sha256"] != digest(model_dir / str(model_metadata.get("model_file", ""))):
        raise ValueError("Selected model checkpoint does not match its metadata fingerprint.")
    expected_hashes = {
        "dataset_sha256": dataset_path,
        "checkpoint_sha256": model_dir / str(model_metadata.get("model_file", "")),
        "model_metadata_sha256": model_metadata_path,
        "error_summary_sha256": summary_path,
    }
    for key, path in expected_hashes.items():
        if not path.is_file() or metadata.get(key) != digest(path):
            raise ValueError(f"Calibration provenance mismatch for {key}; regenerate calibration from the current inputs.")
    if not direction_path.is_file() or metadata.get("constraint_direction_sha256") != digest(direction_path):
        raise ValueError("Calibration constraint-direction report is missing or has changed.")
    for target in TARGET_COLUMNS:
        rows = directions[(directions["region"] == "all_validation") & (directions["target"] == target)]
        expected_direction = "frequency_overprediction" if target in TARGET_COLUMNS[:FREQUENCY_COUNT] else "maximum_response_underprediction"
        if len(rows) != 1 or rows.iloc[0]["dangerous_error_direction"] != expected_direction:
            raise ValueError(f"Calibration dangerous-error direction is inconsistent for {target}.")

    from scripts.train_surrogate import split_dataset
    from scripts.train_surrogate import TARGET_UNITS
    dataset = pd.read_csv(dataset_path)
    _, validation, _ = split_dataset(dataset, int(model_metadata["seed"]))
    validation_ids = ",".join(map(str, validation["design_id"].tolist()))
    expected_ids_hash = hashlib.sha256(validation_ids.encode("utf-8")).hexdigest()
    if metadata.get("validation_design_ids_sha256") != expected_ids_hash:
        raise ValueError("Calibration validation split fingerprint does not match current dataset/model seed.")
    if model_metadata.get("validation_design_ids_sha256") and model_metadata["validation_design_ids_sha256"] != expected_ids_hash:
        raise ValueError("Selected model and calibration use different validation design identities.")
    if metadata.get("validation_rows") != len(validation):
        raise ValueError("Calibration validation row count does not match the current split.")
    inertia = validation[["J1", "J2", "J3"]].sum(axis=1)
    expected_cutoff = float(inertia.quantile(float(metadata.get("low_inertia_fraction", 0.10))))
    cutoff = float(metadata["low_inertia_cutoff"])
    if not np.isclose(cutoff, expected_cutoff, rtol=0.0, atol=1e-12):
        raise ValueError("Calibration low-inertia cutoff does not match the current validation split.")

    def directional_allowances(region_name: str) -> NDArray[np.float64]:
        region = summary[summary["region"] == region_name]
        by_target = region.set_index("target")
        if len(region) != OUTPUT_DIM or set(TARGET_COLUMNS) != set(by_target.index):
            raise ValueError(f"Validation calibration is missing {region_name} target rows.")
        for target in TARGET_COLUMNS:
            if str(by_target.loc[target, "units"]) != TARGET_UNITS[target]:
                raise ValueError(f"Calibration units are incompatible for {target}.")
        expected_count = int(metadata["validation_rows"] if region_name == "all_validation" else metadata["low_inertia_validation_rows"])
        if not np.all(region["sample_count"].to_numpy(dtype=int) == expected_count):
            raise ValueError(f"Calibration sample counts are inconsistent for {region_name}.")
        values = np.array([
            max(0.0, float(by_target.loc[name, "max_overprediction"]))
            for name in TARGET_COLUMNS[:FREQUENCY_COUNT]
        ] + [
            max(0.0, -float(by_target.loc[name, "max_underprediction"]))
            for name in TARGET_COLUMNS[FREQUENCY_COUNT:]
        ])
        if not np.all(np.isfinite(values)):
            raise ValueError("Validation allowances must be finite.")
        return values
    # Prediction minus actual: overprediction is dangerous for frequencies;
    # underprediction magnitude is dangerous for response maxima.
    all_validation = directional_allowances("all_validation")
    low_inertia = directional_allowances("lowest_inertia_validation_10pct")
    return all_validation, low_inertia, cutoff


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
    allowances: ArrayLike | None = None,
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

    allowance = np.zeros(OUTPUT_DIM) if allowances is None else np.asarray(allowances, dtype=float)
    if allowance.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(allowance)) or np.any(allowance < 0):
        raise ValueError("allowances must be seven finite nonnegative values.")
    conservative = predicted.copy()
    conservative[:FREQUENCY_COUNT] -= allowance[:FREQUENCY_COUNT]
    conservative[FREQUENCY_COUNT:] += allowance[FREQUENCY_COUNT:]
    frequency_margins = (
        conservative[:FREQUENCY_COUNT] / baseline[:FREQUENCY_COUNT] - 1.0
    )

    response_margins = (
        1.0
        - conservative[FREQUENCY_COUNT:] / baseline[FREQUENCY_COUNT:]
    )

    return np.concatenate((frequency_margins, response_margins))


def conservative_allowance_envelope(
    validation_allowances: ArrayLike | None,
    low_inertia_allowances: ArrayLike | None,
) -> NDArray[np.float64]:
    """Return one constant empirical safeguard across the search domain.

    The low-inertia validation rows are a subset of all validation rows, so
    the all-validation directional maxima should envelope their regional
    maxima. Taking the elementwise maximum also protects against malformed
    or independently regenerated summaries. A single envelope avoids a
    discontinuous constraint jump when a trial crosses the inertia cutoff,
    which otherwise violates COBYQA's smooth local-model assumption.
    These empirical maxima are not formal error bounds.
    """
    supplied = [value for value in (validation_allowances, low_inertia_allowances)
                if value is not None]
    if not supplied:
        return np.zeros(OUTPUT_DIM, dtype=np.float64)

    arrays = [np.asarray(value, dtype=np.float64) for value in supplied]
    if any(array.shape != (OUTPUT_DIM,) for array in arrays):
        raise ValueError("allowances must each contain seven values.")
    if any(not np.all(np.isfinite(array)) or np.any(array < 0.0)
           for array in arrays):
        raise ValueError("allowances must be finite and nonnegative.")
    return np.maximum.reduce(arrays)


def optimize_design(
    model: SurrogateMLP,
    baseline_design: ArrayLike | None = None,
    n_starts: int = 5,
    seed: int = 42,
    max_iterations: int = 50000,
    tolerance: float = 1e-9,
    solver: str = "SLSQP",
    validation_allowances: ArrayLike | None = None,
    low_inertia_allowances: ArrayLike | None = None,
    low_inertia_cutoff: float | None = None,
    constraint_reference_outputs: ArrayLike | None = None,
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
    constraint_reference = (
        baseline_outputs if constraint_reference_outputs is None
        else np.asarray(constraint_reference_outputs, dtype=float)
    )
    if constraint_reference.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(constraint_reference)) or np.any(constraint_reference <= 0.0):
        raise ValueError("constraint_reference_outputs must contain seven positive finite values.")

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
    all_candidates: list[dict] = []

    # Keep the surrogate constraints continuous across the low-inertia
    # cutoff. The low-region empirical maxima are already included in this
    # elementwise envelope; switching allowances at the cutoff creates a
    # discontinuity that destabilizes COBYQA's local quadratic models.
    optimization_allowances = conservative_allowance_envelope(
        validation_allowances, low_inertia_allowances
    )

    def trial_design(scaled: ArrayLike) -> tuple[NDArray[np.float64], float]:
        """Map solver trial points and quantify any bound violation."""
        scaled = np.asarray(scaled, dtype=np.float64)

        if scaled.shape != (DESIGN_DIM,):
            raise ValueError(
                f"Expected {DESIGN_DIM} scaled variables; "
                f"received shape {scaled.shape}."
            )

        if not np.all(np.isfinite(scaled)):
            raise ValueError("Scaled design contains non-finite values.")
        violation = max(0.0, float(np.max(-scaled)), float(np.max(scaled - 1.0)))
        projected = np.clip(scaled, 0.0, 1.0)
        return _get_bounds()[0] + projected * (_get_bounds()[1] - _get_bounds()[0]), violation

    def objective(scaled: NDArray[np.float64]) -> float:
        design, violation = trial_design(scaled)
        if violation > 0.0:
            return 1.0 + violation * violation
        return total_inertia(design) / maximum_inertia

    def constraints(scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        design, violation = trial_design(scaled)
        if violation > 0.0:
            return np.full(OUTPUT_DIM, -violation)
        outputs = predict_design(model, design)
        return constraint_margins(outputs, constraint_reference, optimization_allowances)

    
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
                "eps": SLSQP_FINITE_DIFFERENCE_STEP,
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
                # COBYQA's evaluation budget is separate from its iteration
                # count; keep it bounded so multistart runs stay reviewable.
                "maxfev": min(max_iterations, 50000),
                "feasibility_tol": 1e-8,
                "initial_tr_radius": 0.2,
                "final_tr_radius": 1e-6,
                "disp": False,
            }

        try:
            result = minimize(
                objective,
                x0=start,
                method=solver,
                bounds=[(0.0, 1.0)] * DESIGN_DIM,
                constraints=solver_constraints,
                options=options,
            )
        except Exception as exc:
            print(f"{solver} start {start_index} failed: {type(exc).__name__}: {exc}")
            continue

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
        margins = constraint_margins(candidate_outputs, constraint_reference, optimization_allowances)

        feasible = bool(
            np.all(np.isfinite(margins))
            and np.all(margins >= -FEASIBILITY_TOLERANCE)
            and np.all(candidate >= lower)
            and np.all(candidate <= upper)
        )


        candidate_objective = total_inertia(candidate)
        candidate_allowances = optimization_allowances
        all_candidates.append({"design": candidate.copy(), "outputs": candidate_outputs.copy(),
                               "allowances": candidate_allowances.copy(),
                               "margins": margins.copy(), "objective": candidate_objective,
                               "surrogate_feasible": feasible, "result": result,
                               "in_bounds": True})

        print(
            f"Start {start_index}/{len(starts)}: "
            f"success={result.success}, "
            f"message={result.message!r}, "
            f"nit={getattr(result, 'nit', None)}, "
            f"nfev={getattr(result, 'nfev', None)}, "
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


    final_margins = constraint_margins(best_outputs, constraint_reference, optimization_allowances)

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
        surrogate_feasible=bool(best_result is not None),
        candidates=tuple(all_candidates),
    )
