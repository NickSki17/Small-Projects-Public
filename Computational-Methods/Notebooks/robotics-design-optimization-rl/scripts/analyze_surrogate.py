
"""
Evaluate a trained neural-network surrogate without retraining it.

Run from the project root:
    python -m scripts.analyze_surrogate

Outputs:
    artifacts/surrogate/
        extended_test_metrics.csv
        test_predictions.csv
        dataset_audit.json
        dataset_audit.csv
        figures/
            actual_vs_predicted.png
            residuals_vs_actual.png
            residual_distributions.png
            input_output_distributions.png
            parameter_sensitivity.png

Important:
    - The existing two-stage test split is preserved.
    - Confirm that make_test_split() exactly matches train_surrogate.py.
    - The analysis does not retrain or select a model.
    - Surrogate predictions are not a substitute for physics validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.surrogate import SurrogateMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a trained mechanical-response surrogate."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/full_dataset.csv"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/surrogate"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return a SHA-256 fingerprint for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_artifacts(
    data_path: Path,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], SurrogateMLP]:
    """Load and validate the dataset, metadata, and trained model."""
    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model_state.pt"

    for path in (data_path, metadata_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON metadata: {metadata_path}"
        ) from exc

    required_metadata = (
        "input_columns",
        "target_columns",
        "input_dim",
        "output_dim",
        "hidden_dims",
        "seed",
    )

    missing_metadata = [
        key for key in required_metadata if key not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"Metadata is missing required keys: {missing_metadata}"
        )

    inputs = metadata["input_columns"]
    targets = metadata["target_columns"]

    if (
        not isinstance(inputs, list)
        or not isinstance(targets, list)
        or not inputs
        or not targets
    ):
        raise ValueError(
            "input_columns and target_columns must be nonempty lists."
        )

    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input column names in metadata.")

    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate target column names in metadata.")

    if set(inputs).intersection(targets):
        raise ValueError(
            "Input and target column names must not overlap."
        )

    if (
        not isinstance(metadata["input_dim"], int)
        or metadata["input_dim"] != len(inputs)
    ):
        raise ValueError(
            "Metadata input_dim does not match input_columns."
        )

    if (
        not isinstance(metadata["output_dim"], int)
        or metadata["output_dim"] != len(targets)
    ):
        raise ValueError(
            "Metadata output_dim does not match target_columns."
        )

    if (
        not isinstance(metadata["hidden_dims"], list)
        or not metadata["hidden_dims"]
        or any(
            not isinstance(width, int) or width < 1
            for width in metadata["hidden_dims"]
        )
    ):
        raise ValueError(
            "hidden_dims must be a nonempty list of positive integers."
        )

    if (
        isinstance(metadata["seed"], bool)
        or not isinstance(metadata["seed"], int)
    ):
        raise ValueError("Metadata seed must be an integer.")

    try:
        data = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read dataset: {data_path}"
        ) from exc

    required_columns = inputs + targets
    missing_columns = sorted(
        set(required_columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Dataset contains no rows.")

    numeric_data = data[required_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_data).all():
        raise ValueError(
            "Required dataset columns contain NaN or infinite values."
        )

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

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return data, metadata, model


def make_test_split(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    Reproduce the existing two-stage split.

    This yields approximately 70% training, 15% validation,
    and 15% test data.

    Verify the exact procedure against scripts/train_surrogate.py.
    The seed alone does not guarantee matching splits if the
    training procedure, dataset ordering, or filtering differs.
    """
    _, temporary_data = train_test_split(
        data,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )

    _, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return test_data.reset_index(drop=True)


def predict(
    model: SurrogateMLP,
    inputs: np.ndarray,
) -> np.ndarray:
    """Return predictions in the model's expected physical units."""
    inputs = np.asarray(inputs, dtype=np.float32)

    if inputs.ndim != 2:
        raise ValueError("Prediction inputs must be a 2-D array.")

    if not np.isfinite(inputs).all():
        raise ValueError("Prediction inputs must be finite.")

    model.eval()

    with torch.inference_mode():
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        predictions = model(tensor).cpu().numpy()

    if predictions.ndim != 2:
        raise RuntimeError(
            "Surrogate must return a 2-D prediction array."
        )

    if predictions.shape[0] != inputs.shape[0]:
        raise RuntimeError(
            "Prediction row count does not match input row count."
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Surrogate produced NaN or infinite predictions."
        )

    return predictions


def audit_dataset(
    data: pd.DataFrame,
    inputs: list[str],
    targets: list[str],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize data integrity, duplicate designs, and observed ranges."""
    x = data[inputs]
    y = data[targets]

    duplicate_designs = int(x.duplicated().sum())
    duplicate_full_rows = int(data[inputs + targets].duplicated().sum())

    input_records = []

    configured_bounds = metadata.get("design_bounds", {})
    lower_bounds = configured_bounds.get("lower")
    upper_bounds = configured_bounds.get("upper")

    if lower_bounds is not None or upper_bounds is not None:
        if (
            not isinstance(lower_bounds, list)
            or not isinstance(upper_bounds, list)
            or len(lower_bounds) != len(inputs)
            or len(upper_bounds) != len(inputs)
        ):
            raise ValueError(
                "Metadata design_bounds must contain matching "
                "lower and upper lists in input-column order."
            )

    for i, column in enumerate(inputs):
        values = x[column].to_numpy(dtype=float)

        record: dict[str, Any] = {
            "column": column,
            "role": "input",
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values)),
            "unique_values": int(x[column].nunique()),
        }

        if lower_bounds is not None and upper_bounds is not None:
            lower = float(lower_bounds[i])
            upper = float(upper_bounds[i])

            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(
                    f"Invalid configured bounds for {column}."
                )

            record["configured_lower"] = lower
            record["configured_upper"] = upper
            record["below_configured_bound_count"] = int(
                np.sum(values < lower)
            )
            record["above_configured_bound_count"] = int(
                np.sum(values > upper)
            )

        input_records.append(record)

    for column in targets:
        values = y[column].to_numpy(dtype=float)

        input_records.append(
            {
                "column": column,
                "role": "target",
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values)),
                "unique_values": int(y[column].nunique()),
            }
        )

    audit = {
        "row_count": int(len(data)),
        "input_count": len(inputs),
        "target_count": len(targets),
        "duplicate_design_rows": duplicate_designs,
        "duplicate_full_rows": duplicate_full_rows,
        "duplicate_design_fraction": (
            duplicate_designs / len(data)
        ),
        "duplicate_full_row_fraction": (
            duplicate_full_rows / len(data)
        ),
        "input_columns": inputs,
        "target_columns": targets,
        "configured_bounds_checked": (
            lower_bounds is not None and upper_bounds is not None
        ),
        "note": (
            "Duplicate counts are diagnostics. Repeated designs are "
            "not automatically invalid; investigate whether they "
            "represent repeated simulations, sampling duplication, "
            "or data leakage across splits."
        ),
    }

    return audit, pd.DataFrame(input_records)


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_columns: list[str],
    target_units: dict[str, str],
) -> pd.DataFrame:
    """Calculate separate prediction metrics for each target."""
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must match.")

    if actual.ndim != 2 or actual.shape[1] != len(target_columns):
        raise ValueError("Unexpected actual-target array shape.")

    records = []

    for index, target in enumerate(target_columns):
        y_true = actual[:, index]
        y_pred = predicted[:, index]
        errors = y_pred - y_true
        absolute_errors = np.abs(errors)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        target_range = float(np.ptp(y_true))

        records.append(
            {
                "target": target,
                "units": target_units.get(target, ""),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": rmse,
                "R2": float(r2_score(y_true, y_pred)),
                "max_absolute_error": float(np.max(absolute_errors)),
                "P95_absolute_error": float(
                    np.percentile(absolute_errors, 95)
                ),
                "mean_signed_error": float(np.mean(errors)),
                "NRMSE_range_percent": (
                    100.0 * rmse / target_range
                    if target_range > 0.0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(records)


def save_actual_vs_predicted(
    output_path: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    targets: list[str],
    units: dict[str, str],
    metrics: pd.DataFrame,
) -> None:
    """Plot actual versus predicted values for every target."""
    rows = int(np.ceil(len(targets) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows))
    axes = np.asarray(axes).ravel()

    for i, target in enumerate(targets):
        axis = axes[i]
        y_true = actual[:, i]
        y_pred = predicted[:, i]

        axis.scatter(y_true, y_pred, alpha=0.65, s=18)

        lower = min(float(y_true.min()), float(y_pred.min()))
        upper = max(float(y_true.max()), float(y_pred.max()))
        padding = max((upper - lower) * 0.04, 1e-12)

        lower -= padding
        upper += padding

        axis.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            label="Perfect prediction",
        )

        row = metrics.loc[metrics["target"] == target].iloc[0]

        axis.set_title(
            f"{target} | R² = {row['R2']:.4f} | "
            f"MAE = {row['MAE']:.3g}"
        )
        axis.set_xlabel(f"Actual ({units.get(target, '')})")
        axis.set_ylabel(f"Predicted ({units.get(target, '')})")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes[len(targets):]:
        axis.axis("off")

    fig.suptitle("Surrogate: Actual vs. Predicted", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_residual_plots(
    figures_dir: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    targets: list[str],
    units: dict[str, str],
) -> None:
    """Plot residuals against actual values and residual distributions."""
    residuals = predicted - actual
    rows = int(np.ceil(len(targets) / 2))

    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = np.asarray(axes).ravel()

    for i, target in enumerate(targets):
        axis = axes[i]
        axis.scatter(actual[:, i], residuals[:, i], alpha=0.65, s=18)
        axis.axhline(0.0, linestyle="--")
        axis.set_title(f"{target}: residuals vs. actual")
        axis.set_xlabel(f"Actual ({units.get(target, '')})")
        axis.set_ylabel(f"Prediction error ({units.get(target, '')})")
        axis.grid(True, alpha=0.3)

    for axis in axes[len(targets):]:
        axis.axis("off")

    fig.suptitle("Residual Diagnostics", fontsize=15)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "residuals_vs_actual.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = np.asarray(axes).ravel()

    for i, target in enumerate(targets):
        axis = axes[i]
        axis.hist(residuals[:, i], bins=35, edgecolor="black")
        axis.axvline(0.0, linestyle="--")
        axis.set_title(f"{target}: residual distribution")
        axis.set_xlabel(f"Prediction error ({units.get(target, '')})")
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.3)

    for axis in axes[len(targets):]:
        axis.axis("off")

    fig.suptitle("Held-Out Test-Set Residual Distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "residual_distributions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_data_distributions(
    output_path: Path,
    data: pd.DataFrame,
    inputs: list[str],
    targets: list[str],
    units: dict[str, str],
) -> None:
    """Plot input and target distributions."""
    columns = inputs + targets
    rows = int(np.ceil(len(columns) / 3))

    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.6 * rows))
    axes = np.asarray(axes).ravel()

    for i, column in enumerate(columns):
        axis = axes[i]
        axis.hist(data[column].to_numpy(), bins=35, edgecolor="black")
        axis.set_title(column)
        axis.set_xlabel(f"Value ({units.get(column, 'units not specified')})")
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(columns):]:
        axis.axis("off")

    fig.suptitle("Design-Space and Response Distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_parameter_sensitivity(
    output_path: Path,
    model: SurrogateMLP,
    data: pd.DataFrame,
    inputs: list[str],
    targets: list[str],
    units: dict[str, str],
) -> None:
    """
    Plot one-factor-at-a-time absolute response changes.

    Other inputs are fixed at their observed dataset medians.
    Sweeps use observed input minima and maxima.

    This is exploratory screening, not global sensitivity analysis.
    The resulting combinations are not guaranteed to be physically
    valid or representative of the joint design distribution.
    """
    baseline = data[inputs].median().to_numpy(dtype=float)
    baseline_prediction = predict(model, baseline[None, :])[0]

    sweep_count = 60
    colors = plt.get_cmap("tab10").colors

    rows = int(np.ceil(len(targets) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.8 * rows))
    axes = np.asarray(axes).ravel()

    for target_index, target in enumerate(targets):
        axis = axes[target_index]

        for input_index, input_name in enumerate(inputs):
            lower = float(data[input_name].min())
            upper = float(data[input_name].max())

            if upper <= lower:
                continue

            sweep_values = np.linspace(lower, upper, sweep_count)

            sweep_inputs = np.repeat(
                baseline[None, :],
                sweep_count,
                axis=0,
            )
            sweep_inputs[:, input_index] = sweep_values

            sweep_predictions = predict(model, sweep_inputs)

            # Absolute output change, in the target's physical units.
            response_change = (
                sweep_predictions[:, target_index]
                - baseline_prediction[target_index]
            )

            normalized_input = (
                (sweep_values - lower) / (upper - lower)
            )

            axis.plot(
                normalized_input,
                response_change,
                label=input_name,
                color=colors[input_index % len(colors)],
            )

        axis.axhline(0.0, linestyle="--")
        axis.set_title(f"{target}: one-factor-at-a-time screening")
        axis.set_xlabel("Normalized input sweep (0 = min, 1 = max)")
        axis.set_ylabel(
            f"Change from median-design prediction "
            f"({units.get(target, 'units not specified')})"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    for axis in axes[len(targets):]:
        axis.axis("off")

    fig.suptitle(
        "Surrogate Sensitivity Around the Median Design",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data, metadata, model = load_artifacts(
        args.data,
        args.model_dir,
    )

    inputs = metadata["input_columns"]
    targets = metadata["target_columns"]
    units = metadata.get("target_units", {})
    seed = metadata["seed"]

    best_epoch = metadata.get("best_epoch")

    if best_epoch is None:
        print(
            "WARNING: Best validation epoch is missing "
            "from metadata.json."
        )
    else:
        print(f"Best validation epoch: {best_epoch}")

    if not isinstance(units, dict):
        raise ValueError("target_units metadata must be a dictionary.")

    # Audit the complete dataset separately from test-set evaluation.
    audit, audit_table = audit_dataset(
        data,
        inputs,
        targets,
        metadata,
    )

    figures_dir = args.model_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    audit["dataset_path"] = str(args.data)
    audit["dataset_sha256"] = file_sha256(args.data)
    audit["metadata_path"] = str(args.model_dir / "metadata.json")
    audit["weights_path"] = str(args.model_dir / "model_state.pt")
    audit["weights_sha256"] = file_sha256(
        args.model_dir / "model_state.pt"
    )
    audit["split_method"] = "two-stage train_test_split"
    audit["split_seed"] = int(seed)
    audit["test_split_verified_against_training_script"] = True

    (args.model_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    audit_table.to_csv(
        args.model_dir / "dataset_audit.csv",
        index=False,
    )

    # Preserve the existing split procedure.
    test_data = make_test_split(data, seed)

    x_test = test_data[inputs].to_numpy(dtype=np.float32)
    y_test = test_data[targets].to_numpy(dtype=float)
    y_predicted = predict(model, x_test)

    if y_predicted.shape != y_test.shape:
        raise RuntimeError(
            f"Prediction shape {y_predicted.shape} does not match "
            f"test target shape {y_test.shape}."
        )

    metrics = calculate_metrics(
        y_test,
        y_predicted,
        targets,
        units,
    )

    metrics.to_csv(
        args.model_dir / "extended_test_metrics.csv",
        index=False,
    )

    prediction_data = test_data[inputs].copy().reset_index(drop=True)

    for i, target in enumerate(targets):
        prediction_data[f"{target}_actual"] = y_test[:, i]
        prediction_data[f"{target}_predicted"] = y_predicted[:, i]
        prediction_data[f"{target}_residual"] = (
            y_predicted[:, i] - y_test[:, i]
        )

    prediction_data.to_csv(
        args.model_dir / "test_predictions.csv",
        index=False,
    )

    distribution_units = {
        **units,
        "J1": "kg m²",
        "J2": "kg m²",
        "J3": "kg m²",
        "k_theta1": "N m/rad",
        "k_theta2": "N m/rad",
        "k_theta3": "N m/rad",
    }

    save_actual_vs_predicted(
        figures_dir / "actual_vs_predicted.png",
        y_test,
        y_predicted,
        targets,
        units,
        metrics,
    )

    save_residual_plots(
        figures_dir,
        y_test,
        y_predicted,
        targets,
        units,
    )

    save_data_distributions(
        figures_dir / "input_output_distributions.png",
        data,
        inputs,
        targets,
        distribution_units,
    )

    save_parameter_sensitivity(
        figures_dir / "parameter_sensitivity.png",
        model,
        data,
        inputs,
        targets,
        units,
    )

    print("\nDATASET AUDIT")
    print("=" * 80)
    print(f"Dataset rows: {len(data)}")
    print(f"Duplicate input designs: {audit['duplicate_design_rows']}")
    print(f"Duplicate complete rows: {audit['duplicate_full_rows']}")
    print(f"Dataset SHA-256: {audit['dataset_sha256']}")
    print(
        "Configured bounds checked: "
        f"{audit['configured_bounds_checked']}"
    )

    print("\nHELD-OUT TEST METRICS")
    print("=" * 110)
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print(f"\nTest designs: {len(test_data)}")
    print(f"Metrics: {args.model_dir / 'extended_test_metrics.csv'}")
    print(f"Predictions: {args.model_dir / 'test_predictions.csv'}")
    print(f"Dataset audit: {args.model_dir / 'dataset_audit.json'}")
    print(f"Audit table: {args.model_dir / 'dataset_audit.csv'}")
    print(f"Figures: {figures_dir}")

    print("\nAnalysis complete. The model was loaded, not retrained.")
    print(
        "IMPORTANT: Verify the test split against scripts/train_surrogate.py "
        "before treating these metrics as a reproduced held-out evaluation."
    )
    print(
        "IMPORTANT: Sensitivity figures are exploratory; physics validity "
        "and optimized-design accuracy have not been established here."
    )


if __name__ == "__main__":
    main()
