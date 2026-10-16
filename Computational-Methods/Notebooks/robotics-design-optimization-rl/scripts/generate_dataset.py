
"""
Generate a reproducible dataset for the 3-DOF rotational torsional
vibration system.

Run from the project root:

    python -m scripts.generate_dataset

Optional smaller run for a quick check:

    python -m scripts.generate_dataset --samples 20

The default dataset is saved to data/full_dataset.csv.
Existing pilot files are not overwritten.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.dataset_config import (
    DATASET_SAMPLES,
    DAMPINGS,
    INITIAL_STATE,
    LOWER_BOUNDS,
    RTOL,
    ATOL,
    SEED,
    T_END,
    T_START,
    TIME_STEP,
    UPPER_BOUNDS,
)
from src.dynamics import (
    build_inertia_matrix,
    build_rotational_damping_matrix,
    build_torsional_stiffness_matrix,
)
from src.metrics import compute_response_metrics
from src.modal import solve_modes
from src.simulation import simulate_system


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "full_dataset.csv"

PARAMETER_NAMES = (
    "J1",
    "J2",
    "J3",
    "k_theta1",
    "k_theta2",
    "k_theta3",
)

METRIC_NAMES = (
    "theta_max",
    "theta1_max",
    "theta2_max",
    "theta3_max",
    "alpha_max",
    "theta_rms",
)

FREQUENCY_NAMES = ("omega1", "omega2", "omega3")

CANDIDATE_TARGETS = (
    "theta2_max",
    "theta3_max",
    "alpha_max",
    "theta_rms",
)


def generate_designs(n_samples: int) -> np.ndarray:
    """Generate design parameters using seeded Latin hypercube sampling."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    lower_bounds = np.asarray(LOWER_BOUNDS, dtype=float)
    upper_bounds = np.asarray(UPPER_BOUNDS, dtype=float)

    if lower_bounds.shape != (6,) or upper_bounds.shape != (6,):
        raise ValueError("Parameter bounds must each contain six values.")

    if (
        not np.all(np.isfinite(lower_bounds))
        or not np.all(np.isfinite(upper_bounds))
        or np.any(upper_bounds <= lower_bounds)
    ):
        raise ValueError("Parameter bounds must be finite and ordered.")

    sampler = qmc.LatinHypercube(d=6, seed=SEED)
    unit_samples = sampler.random(n=n_samples)

    designs = qmc.scale(
        unit_samples,
        lower_bounds,
        upper_bounds,
    )

    if not np.all(np.isfinite(designs)):
        raise RuntimeError("Design generation produced non-finite values.")

    if np.any(designs < lower_bounds) or np.any(designs > upper_bounds):
        raise RuntimeError("Generated designs fall outside parameter bounds.")

    return designs


def evaluate_design(design: np.ndarray) -> dict[str, float]:
    """Evaluate modal frequencies and passive response metrics for one design."""
    design = np.asarray(design, dtype=float)

    if design.shape != (6,):
        raise ValueError("Each design must contain exactly six parameters.")

    if not np.all(np.isfinite(design)):
        raise ValueError("Design parameters must be finite.")

    inertias = design[:3]
    torsional_stiffnesses = design[3:]

    inertia_matrix = build_inertia_matrix(inertias)
    damping_matrix = build_rotational_damping_matrix(DAMPINGS)
    stiffness_matrix = build_torsional_stiffness_matrix(
        torsional_stiffnesses
    )

    natural_frequencies, _ = solve_modes(
        inertia_matrix,
        stiffness_matrix,
    )

    time_vector, state_history = simulate_system(
        inertia_matrix,
        damping_matrix,
        stiffness_matrix,
        INITIAL_STATE,
        (T_START, T_END),
        TIME_STEP,
        rtol=RTOL,
        atol=ATOL,
    )

    metrics = compute_response_metrics(
        time_vector,
        state_history,
        inertia_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    results = {
        **{
            name: float(value)
            for name, value in zip(FREQUENCY_NAMES, natural_frequencies)
        },
        **{name: float(value) for name, value in metrics.items()},
    }

    if not np.all(np.isfinite(list(results.values()))):
        raise RuntimeError("A design produced non-finite output metrics.")

    if any(results[name] <= 0.0 for name in FREQUENCY_NAMES):
        raise RuntimeError("A design produced a non-positive natural frequency.")

    return results


def validate_dataset(
    dataset: pd.DataFrame,
    designs: np.ndarray,
) -> None:
    """Raise an error if basic dataset integrity checks fail."""
    expected_columns = (
        ("design_id",)
        + PARAMETER_NAMES
        + FREQUENCY_NAMES
        + METRIC_NAMES
    )

    if tuple(dataset.columns) != expected_columns:
        raise RuntimeError("Dataset columns do not match the expected schema.")

    if len(dataset) != len(designs):
        raise RuntimeError("Dataset row count does not match design count.")

    numeric_values = dataset.select_dtypes(include="number").to_numpy()

    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError("Dataset contains non-finite numeric values.")

    lower_bounds = np.asarray(LOWER_BOUNDS, dtype=float)
    upper_bounds = np.asarray(UPPER_BOUNDS, dtype=float)
    parameter_values = dataset[list(PARAMETER_NAMES)].to_numpy()

    if np.any(parameter_values < lower_bounds):
        raise RuntimeError("Dataset contains parameters below their bounds.")

    if np.any(parameter_values > upper_bounds):
        raise RuntimeError("Dataset contains parameters above their bounds.")

    if dataset["design_id"].tolist() != list(range(1, len(dataset) + 1)):
        raise RuntimeError("Design IDs are missing, duplicated, or unordered.")

    if dataset[list(FREQUENCY_NAMES)].le(0.0).any().any():
        raise RuntimeError("Dataset contains non-positive natural frequencies.")

    if dataset[list(METRIC_NAMES)].lt(0.0).any().any():
        raise RuntimeError("Absolute response metrics cannot be negative.")

    if dataset.duplicated(subset=list(PARAMETER_NAMES)).any():
        raise RuntimeError("Dataset contains duplicate design vectors.")

    for target in CANDIDATE_TARGETS:
        if dataset[target].nunique() < 2:
            raise RuntimeError(
                f"Candidate target '{target}' has no observed variation."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the rotational torsional simulation dataset."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DATASET_SAMPLES,
        help=f"Number of designs to evaluate (default: {DATASET_SAMPLES}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path (default: data/full_dataset.csv).",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1.")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if output_path.resolve() in {
        (PROJECT_ROOT / "data" / "pilot_dataset.csv").resolve(),
        (PROJECT_ROOT / "data" / "pilot_convergence.csv").resolve(),
    }:
        parser.error("The output path cannot overwrite a pilot validation file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("ROTATIONAL TORSIONAL DATASET GENERATION")
    print("=" * 60)
    print(f"Designs: {args.samples}")
    print(f"Random seed: {SEED}")
    print(f"Time interval: {TIME_STEP} s")
    print(f"Simulation duration: {T_END - T_START} s")
    print(f"Output: {output_path}")
    print()

    designs = generate_designs(args.samples)
    records: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for index, design in enumerate(designs, start=1):
        response = evaluate_design(design)

        record = {
            "design_id": index,
            **{
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, design)
            },
            **response,
        }
        records.append(record)

        if index % 100 == 0 or index == args.samples:
            elapsed = time.perf_counter() - start_time
            print(
                f"Completed {index}/{args.samples} designs "
                f"({elapsed:.1f} s elapsed)."
            )

    dataset = pd.DataFrame(records)

    # Enforce a stable column order for downstream scripts.
    expected_columns = (
        ("design_id",)
        + PARAMETER_NAMES
        + FREQUENCY_NAMES
        + METRIC_NAMES
    )
    dataset = dataset[list(expected_columns)]

    validate_dataset(dataset, designs)

    # Save only after every design and dataset-level check passes.
    dataset.to_csv(output_path, index=False)

    elapsed = time.perf_counter() - start_time

    print()
    print("DATASET VALIDATION PASSED")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Total runtime: {elapsed:.2f} s")
    print(f"Mean runtime per design: {elapsed / len(dataset):.4f} s")
    print(f"Saved: {output_path}")

    print("\nNatural frequencies (rad/s)")
    print(
        dataset[list(FREQUENCY_NAMES)]
        .agg(["min", "median", "max", "std"])
        .to_string()
    )

    print("\nResponse metrics")
    print(
        dataset[list(METRIC_NAMES)]
        .agg(["min", "median", "max", "std"])
        .to_string()
    )

    print("\nCandidate target ranges")
    for target in CANDIDATE_TARGETS:
        values = dataset[target]
        print(
            f"{target}: min={values.min():.8g}, "
            f"max={values.max():.8g}, "
            f"unique={values.nunique()}"
        )

    if dataset["theta1_max"].nunique() == 1:
        print(
            "\nNote: theta1_max is constant in this dataset. "
            "Keep it as a diagnostic, not a surrogate target."
        )


if __name__ == "__main__":
    main()
