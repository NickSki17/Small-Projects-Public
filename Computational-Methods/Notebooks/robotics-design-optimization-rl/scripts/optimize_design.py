
"""
Run surrogate-based torsional design optimization and validate the result.

Run from the project root:
    python -m scripts.optimize_design

The optimizer minimizes total rotational inertia subject to constraints
defined relative to the baseline surrogate predictions.

The optimized design is then evaluated independently using the original
physics model through scripts.generate_dataset.evaluate_design().

Outputs:
    artifacts/design_optimization/
        design_comparison.csv
        optimization_summary.json
        figures/
            design_comparison.png
            output_comparison.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from scripts.generate_dataset import evaluate_design
from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    FEASIBILITY_TOLERANCE,
    OptimizationOutcome,
    constraint_margins,
    optimize_design,
)
from src.surrogate import SurrogateMLP


DEFAULT_MODEL_DIR = Path("artifacts/surrogate")
DEFAULT_OUTPUT_DIR = Path("artifacts/design_optimization")


@dataclass(frozen=True)
class DesignStudyResult:
    """Optimization results plus independent physical validation."""

    outcome: OptimizationOutcome
    baseline_physics_outputs: NDArray[np.float64]
    optimized_physics_outputs: NDArray[np.float64]
    true_constraint_margins: NDArray[np.float64]
    true_physics_feasible: bool


def load_surrogate(
    model_dir: Path,
) -> tuple[SurrogateMLP, dict[str, Any]]:
    """Load a trained surrogate and its metadata without retraining."""
    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model_state.pt"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("input_columns") != list(INPUT_COLUMNS):
        raise ValueError(
            "Surrogate input_columns do not match src.optimization.INPUT_COLUMNS."
        )

    if metadata.get("target_columns") != list(TARGET_COLUMNS):
        raise ValueError(
            "Surrogate target_columns do not match src.optimization.TARGET_COLUMNS."
        )

    if metadata.get("input_dim") != len(INPUT_COLUMNS):
        raise ValueError("Surrogate metadata has an unexpected input_dim.")

    if metadata.get("output_dim") != len(TARGET_COLUMNS):
        raise ValueError("Surrogate metadata has an unexpected output_dim.")

    model = SurrogateMLP(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_dims=tuple(metadata["hidden_dims"]),
    )

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata


def physics_output_vector(
    design: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate one design with the original numerical physics model.

    Return values in exactly the same order as TARGET_COLUMNS.
    """
    design = np.asarray(design, dtype=float)

    if design.shape != (len(INPUT_COLUMNS),):
        raise ValueError("design must contain exactly six values.")

    if not np.all(np.isfinite(design)):
        raise ValueError("design must contain only finite values.")

    physics_results = evaluate_design(design)

    missing = [
        name for name in TARGET_COLUMNS if name not in physics_results
    ]
    if missing:
        raise KeyError(
            f"Physics evaluation is missing required outputs: {missing}"
        )

    outputs = np.asarray(
        [physics_results[name] for name in TARGET_COLUMNS],
        dtype=float,
    )

    if not np.all(np.isfinite(outputs)):
        raise RuntimeError("Physics evaluation returned non-finite outputs.")

    return outputs


def run_design_study(
    model: SurrogateMLP,
    baseline_design: NDArray[np.float64] | None = None,
    n_starts: int = 5,
    seed: int = 42,
    max_iterations: int = 50000,
    tolerance: float = 1e-9,
) -> DesignStudyResult:
    """
    Optimize using the surrogate, then check the result against true physics.

    The optimizer's surrogate feasibility flag is preserved separately from
    the true-physics feasibility result calculated here.
    """
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)

    if lower.shape != (len(INPUT_COLUMNS),):
        raise ValueError("LOWER_BOUNDS must contain six values.")

    if upper.shape != (len(INPUT_COLUMNS),):
        raise ValueError("UPPER_BOUNDS must contain six values.")

    if baseline_design is None:
        baseline = 0.5 * (lower + upper)
    else:
        baseline = np.asarray(baseline_design, dtype=float)

    # optimize_design performs its own baseline shape, finiteness, and
    # bounds checks. Keep its baseline as the source of truth.
    outcome = optimize_design(
        model,
        baseline_design=baseline,
        n_starts=n_starts,
        seed=seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    baseline_physics = physics_output_vector(outcome.baseline_design)
    optimized_physics = physics_output_vector(outcome.optimized_design)

    if np.any(baseline_physics <= 0.0):
        raise ValueError(
            "True baseline outputs must be positive for relative constraints."
        )

    true_margins = constraint_margins(
        optimized_physics,
        baseline_physics,
    )

    true_feasible = bool(
        np.all(true_margins >= -FEASIBILITY_TOLERANCE)
        and np.all(outcome.optimized_design >= lower)
        and np.all(outcome.optimized_design <= upper)
    )

    return DesignStudyResult(
        outcome=outcome,
        baseline_physics_outputs=baseline_physics,
        optimized_physics_outputs=optimized_physics,
        true_constraint_margins=true_margins,
        true_physics_feasible=true_feasible,
    )


def build_comparison_table(
    result: DesignStudyResult,
) -> pd.DataFrame:
    """Build a CSV-ready table of design variables and response predictions."""
    outcome = result.outcome
    rows: list[dict[str, float | str]] = []

    cases = (
        (
            "baseline",
            outcome.baseline_design,
            outcome.baseline_outputs,
            result.baseline_physics_outputs,
        ),
        (
            "optimized",
            outcome.optimized_design,
            outcome.optimized_outputs,
            result.optimized_physics_outputs,
        ),
    )

    for label, design, surrogate_outputs, physics_outputs in cases:
        row: dict[str, float | str] = {"case": label}

        for index, name in enumerate(INPUT_COLUMNS):
            row[name] = float(design[index])

        for index, name in enumerate(TARGET_COLUMNS):
            row[f"{name}_surrogate"] = float(surrogate_outputs[index])
            row[f"{name}_physics"] = float(physics_outputs[index])

        row["total_inertia"] = float(np.sum(design[:3]))
        rows.append(row)

    return pd.DataFrame(rows)


def save_design_comparison_plot(
    output_path: Path,
    result: DesignStudyResult,
) -> None:
    """Compare baseline and optimized design variables in normalized units."""
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)

    baseline = (
        result.outcome.baseline_design - lower
    ) / (upper - lower)

    optimized = (
        result.outcome.optimized_design - lower
    ) / (upper - lower)

    positions = np.arange(len(INPUT_COLUMNS))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        positions - width / 2,
        baseline,
        width,
        label="Baseline",
    )
    axis.bar(
        positions + width / 2,
        optimized,
        width,
        label="Optimized",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(INPUT_COLUMNS, rotation=25, ha="right")
    axis.set_ylabel("Normalized design variable (0–1)")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Baseline vs. Optimized Design")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_output_comparison_plot(
    output_path: Path,
    result: DesignStudyResult,
) -> None:
    """
    Plot output ratios relative to each model's own baseline prediction.

    Frequency ratios should be >= 1; response ratios should be <= 1
    under the baseline-relative constraints.
    """
    baseline_surrogate = result.outcome.baseline_outputs
    optimized_surrogate = result.outcome.optimized_outputs
    baseline_physics = result.baseline_physics_outputs
    optimized_physics = result.optimized_physics_outputs

    surrogate_ratios = optimized_surrogate / baseline_surrogate
    physics_ratios = optimized_physics / baseline_physics

    positions = np.arange(len(TARGET_COLUMNS))
    width = 0.36

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        positions - width / 2,
        surrogate_ratios,
        width,
        label="Surrogate: optimized / baseline",
    )
    axis.bar(
        positions + width / 2,
        physics_ratios,
        width,
        label="Physics: optimized / baseline",
    )
    axis.axhline(1.0, linestyle="--", linewidth=1)

    axis.set_xticks(positions)
    axis.set_xticklabels(TARGET_COLUMNS, rotation=25, ha="right")
    axis.set_ylabel("Optimized output / baseline output")
    axis.set_title("Surrogate Predictions vs. True-Physics Validation")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_results(
    result: DesignStudyResult,
    output_dir: Path,
) -> None:
    """Save numerical comparisons, a JSON summary, and diagnostic plots."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(result)
    table.to_csv(output_dir / "design_comparison.csv", index=False)

    outcome = result.outcome
    solver = outcome.solver_result

    summary = {
        "objective": "minimize total rotational inertia",
        "constraint_definition": (
            "Natural frequencies >= baseline; response metrics <= baseline"
        ),
        "baseline_design": {
            name: float(value)
            for name, value in zip(
                INPUT_COLUMNS,
                outcome.baseline_design,
                strict=True,
            )
        },
        "optimized_design": {
            name: float(value)
            for name, value in zip(
                INPUT_COLUMNS,
                outcome.optimized_design,
                strict=True,
            )
        },
        "baseline_total_inertia": float(
            np.sum(outcome.baseline_design[:3])
        ),
        "optimized_total_inertia": float(
            np.sum(outcome.optimized_design[:3])
        ),
        "inertia_reduction_percent": float(
            100.0
            * (
                1.0
                - np.sum(outcome.optimized_design[:3])
                / np.sum(outcome.baseline_design[:3])
            )
        ),
        "baseline_surrogate_outputs": dict(
            zip(TARGET_COLUMNS, map(float, outcome.baseline_outputs), strict=True)
        ),
        "optimized_surrogate_outputs": dict(
            zip(TARGET_COLUMNS, map(float, outcome.optimized_outputs), strict=True)
        ),
        "baseline_physics_outputs": dict(
            zip(
                TARGET_COLUMNS,
                map(float, result.baseline_physics_outputs),
                strict=True,
            )
        ),
        "optimized_physics_outputs": dict(
            zip(
                TARGET_COLUMNS,
                map(float, result.optimized_physics_outputs),
                strict=True,
            )
        ),
        "surrogate_feasible": bool(outcome.surrogate_feasible),
        "true_physics_feasible": bool(result.true_physics_feasible),
        "true_physics_constraint_margins": dict(
            zip(
                (
                    "omega1",
                    "omega2",
                    "omega3",
                    "theta2_max",
                    "theta3_max",
                    "alpha_max",
                    "theta_rms",
                ),
                map(float, result.true_constraint_margins),
                strict=True,
            )
        ),
        "solver": {
            "success": bool(solver.success) if solver is not None else None,
            "message": str(solver.message) if solver is not None else (
                "Baseline retained; no better feasible candidate selected."
            ),
            "iterations": int(solver.nit) if solver is not None else None,
        },
        "interpretation": (
            "True-physics feasibility is determined by evaluating the selected "
            "design with the original numerical model. A successful solver "
            "status alone does not establish engineering feasibility."
        ),
    }

    (output_dir / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    save_design_comparison_plot(
        figures_dir / "design_comparison.png",
        result,
    )
    save_output_comparison_plot(
        figures_dir / "output_comparison.png",
        result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a torsional mechanical design using a trained surrogate "
            "and validate the result with the original physics model."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing metadata.json and model_state.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for optimization results and figures.",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=5,
        help="Number of SLSQP starting points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for multistart optimization.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50000,
        help="Maximum SLSQP iterations per start.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="SLSQP objective/constraint tolerance.",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(
        "This legacy optimizer does not apply validation-calibrated constraints "
        "or enforce independent-physics acceptance. Use `python -m "
        "src.optimization_adapter --calibration-dir <diagnostic-run>` instead."
    )


if __name__ == "__main__":
    main()
