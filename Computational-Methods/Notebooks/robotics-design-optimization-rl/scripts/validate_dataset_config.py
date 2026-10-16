"""
Pilot validation for the rotational torsional dataset configuration.

Run from the robotics-design-optimization-rl project root:

    python scripts/validate_dataset_config.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.dataset_config import (
    ATOL,
    DAMPINGS,
    INITIAL_STATE,
    LOWER_BOUNDS,
    METRIC_RELATIVE_TOLERANCE,
    PILOT_SAMPLES,
    REFINED_TIME_STEP,
    RTOL,
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


DATA_DIR = Path("data")


def generate_designs() -> np.ndarray:
    """Generate reproducible rotational designs using Latin hypercube sampling."""
    sampler = qmc.LatinHypercube(d=6, seed=SEED)
    unit_samples = sampler.random(n=PILOT_SAMPLES)

    return qmc.scale(
        unit_samples,
        LOWER_BOUNDS,
        UPPER_BOUNDS,
    )


def evaluate_design(
    design: np.ndarray,
    time_step: float,
) -> dict[str, float]:
    """Run modal analysis and passive time-domain simulation for one design."""
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

    # Positional matrix arguments keep this call independent of whether
    # the first three parameter names use older or rotational terminology.
    time, state_history = simulate_system(
        inertia_matrix,
        damping_matrix,
        stiffness_matrix,
        INITIAL_STATE,
        (T_START, T_END),
        time_step,
        rtol=RTOL,
        atol=ATOL,
    )

    metrics = compute_response_metrics(
        time,
        state_history,
        inertia_matrix,
        damping_matrix,
        stiffness_matrix,
    )

    return {
        "omega1": float(natural_frequencies[0]),
        "omega2": float(natural_frequencies[1]),
        "omega3": float(natural_frequencies[2]),
        **metrics,
    }


def relative_difference(a: float, b: float) -> float:
    """Symmetric relative difference with protection near zero."""
    scale = max(abs(a), abs(b))

    if scale == 0.0:
        return 0.0

    return abs(a - b) / scale


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    designs = generate_designs()
    records: list[dict[str, float]] = []
    convergence_records: list[dict[str, float]] = []

    parameter_names = (
        "J1",
        "J2",
        "J3",
        "k_theta1",
        "k_theta2",
        "k_theta3",
    )
    metric_names = (
        "theta_max",
        "theta1_max",
        "theta2_max",
        "theta3_max",
        "alpha_max",
        "theta_rms",
    )

    for index, design in enumerate(designs, start=1):
        coarse = evaluate_design(design, TIME_STEP)
        refined = evaluate_design(design, REFINED_TIME_STEP)

        record = {
            "design_id": float(index),
            **{
                name: float(value)
                for name, value in zip(parameter_names, design)
            },
            **coarse,
        }
        records.append(record)

        convergence_record = {"design_id": float(index)}

        for metric in metric_names:
            coarse_value = coarse[metric]
            refined_value = refined[metric]

            convergence_record[
                f"{metric}_relative_difference"
            ] = relative_difference(coarse_value, refined_value)

            convergence_record[
                f"{metric}_absolute_difference"
            ] = abs(coarse_value - refined_value)

        convergence_records.append(convergence_record)

    results = pd.DataFrame(records)
    convergence = pd.DataFrame(convergence_records)

    print("\nROTATIONAL TORSIONAL PILOT DATASET SUMMARY")
    print("=" * 60)
    print(f"Designs evaluated: {len(results)}")
    print(f"Random seed: {SEED}")
    print(f"Coarse time step: {TIME_STEP} s")
    print(f"Refined time step: {REFINED_TIME_STEP} s")

    print("\nNatural angular frequencies (rad/s)")
    print(
        results[["omega1", "omega2", "omega3"]]
        .agg(["min", "median", "max"])
        .to_string()
    )

    print("\nResponse metric distributions")
    print(
        results[list(metric_names)]
        .agg(["min", "median", "max", "std"])
        .to_string()
    )

    relative_columns = [
        f"{metric}_relative_difference"
        for metric in metric_names
    ]
    absolute_columns = [
        f"{metric}_absolute_difference"
        for metric in metric_names
    ]

    print("\nMaximum relative differences: coarse vs. refined")
    print(convergence[relative_columns].max().to_string())

    print("\nMaximum absolute differences: coarse vs. refined")
    print(convergence[absolute_columns].max().to_string())

    print("\nTarget usefulness")
    candidate_targets = (
        "theta1_max",
        "theta2_max",
        "theta3_max",
        "alpha_max",
        "theta_rms",
    )

    target_relative_ranges: dict[str, float] = {}

    for metric in candidate_targets:
        values = results[metric].to_numpy(dtype=float)
        value_range = float(np.ptp(values))
        scale = max(float(np.max(np.abs(values))), 1e-15)
        relative_range = value_range / scale

        target_relative_ranges[metric] = relative_range

        print(
            f"{metric}: range={value_range:.8g}, "
            f"relative range={relative_range:.4%}"
        )

    print(
        "theta_max: retained as the global peak-response diagnostic; "
        "it may be dominated by the initial condition."
    )

    print(
        "Candidate surrogate targets are diagnostic only at this stage; "
        "target usefulness must be assessed before full dataset generation."
    )

    numeric_results = results.select_dtypes(include="number")

    all_finite = bool(
        np.isfinite(numeric_results.to_numpy()).all()
    )
    frequencies_positive = bool(
        (results[["omega1", "omega2", "omega3"]] > 0.0)
        .to_numpy()
        .all()
    )

    max_relative_difference = float(
        convergence[relative_columns].to_numpy().max()
    )
    convergence_passes = (
        max_relative_difference <= METRIC_RELATIVE_TOLERANCE
    )

    print("\nPilot validation checks")
    print(f"All numeric results finite: {all_finite}")
    print(f"All natural frequencies positive: {frequencies_positive}")
    print(
        "Sampling convergence threshold: "
        f"{METRIC_RELATIVE_TOLERANCE:.2%}"
    )
    print(f"Sampling convergence passes: {convergence_passes}")

    results.to_csv(DATA_DIR / "pilot_dataset.csv", index=False)
    convergence.to_csv(
        DATA_DIR / "pilot_convergence.csv",
        index=False,
    )

    print("\nSaved:")
    print("  data/pilot_dataset.csv")
    print("  data/pilot_convergence.csv")

    if not all_finite:
        raise RuntimeError("Pilot validation failed: non-finite results.")

    if not frequencies_positive:
        raise RuntimeError(
            "Pilot validation failed: non-positive natural frequencies."
        )

    if not convergence_passes:
        raise RuntimeError(
            "Pilot validation failed: sampling convergence threshold exceeded."
        )


if __name__ == "__main__":
    main()