"""
Analyze a trained neural-network surrogate without retraining it.

Run from the project root:
    python -m scripts.analyze_surrogate

Outputs:
    artifacts/surrogate/
        extended_test_metrics.csv
        test_predictions.csv
        figures/
            actual_vs_predicted.png
            residuals_vs_actual.png
            residual_distributions.png
            input_output_distributions.png
            parameter_sensitivity.png

The original held-out test split is reproduced using the seed saved
in metadata.json. The test set is not used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def load_artifacts(
    data_path: Path,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict, SurrogateMLP]:
    """Load the dataset, metadata, and trained model weights."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model_state.pt"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = pd.read_csv(data_path)

    input_columns = metadata["input_columns"]
    target_columns = metadata["target_columns"]
    required_columns = input_columns + target_columns

    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    numeric_data = data[required_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_data).all():
        raise ValueError("Dataset contains non-finite required values.")

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

    return data, metadata, model


def make_test_split(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Reproduce the split used by scripts/train_surrogate.py."""
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
    """Return predictions in the original physical units."""
    with torch.no_grad():
        tensor = torch.as_tensor(
            np.array(inputs, copy=True),
            dtype=torch.float32,
        )
        predictions = model(tensor).cpu().numpy()

    if not np.isfinite(predictions).all():
        raise RuntimeError("The surrogate produced non-finite predictions.")

    return predictions


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_columns: list[str],
    target_units: dict[str, str],
) -> pd.DataFrame:
    """Calculate error metrics separately for each target."""
    records = []

    for index, target in enumerate(target_columns):
        y_true = actual[:, index]
        y_pred = predicted[:, index]
        errors = y_pred - y_true
        absolute_errors = np.abs(errors)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        target_range = float(np.max(y_true) - np.min(y_true))

        # NRMSE is expressed as a percentage of the held-out
        # actual-value range. It is undefined for a constant target.
        nrmse_percent = (
            100.0 * rmse / target_range
            if target_range > 0.0
            else float("nan")
        )

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
                "NRMSE_range_percent": nrmse_percent,
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
    """Plot actual versus predicted values for all targets."""
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
            f"{target} | R² = {row['R2']:.5f} | "
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
    """Plot residuals against actual values and their distributions."""
    residuals = predicted - actual
    rows = int(np.ceil(len(targets) / 2))

    # Residuals against actual values
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

    # Residual distributions
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

    fig.suptitle("Distribution of Test-Set Prediction Errors", fontsize=15)
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
    """Show the distribution of all design variables and response targets."""
    columns = inputs + targets
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.6 * rows))
    axes = np.asarray(axes).ravel()

    for i, column in enumerate(columns):
        axis = axes[i]
        axis.hist(data[column].to_numpy(), bins=35, edgecolor="black")
        axis.set_title(column)
        axis.set_xlabel(
            f"Value ({units.get(column, 'design variable units')})"
        )
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
    Plot one-factor-at-a-time surrogate sensitivity around the median design.

    Each input is swept across its observed dataset range while all other
    inputs remain at their dataset medians. Output changes are shown as
    percentages relative to the surrogate prediction at the median design.

    This is a local screening visualization, not a global sensitivity study.
    """
    baseline = data[inputs].median().to_numpy(dtype=float)
    baseline_prediction = predict(model, baseline[None, :])[0]

    # Avoid unstable percentage changes if a baseline prediction is near zero.
    scales = np.maximum(np.abs(baseline_prediction), 1e-12)

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
            sweep_values = np.linspace(lower, upper, sweep_count)

            sweep_inputs = np.repeat(
                baseline[None, :],
                sweep_count,
                axis=0,
            )
            sweep_inputs[:, input_index] = sweep_values

            sweep_predictions = predict(model, sweep_inputs)
            percentage_change = (
                100.0
                * (sweep_predictions[:, target_index]
                   - baseline_prediction[target_index])
                / scales[target_index]
            )

            normalized_input = (
                (sweep_values - lower) / (upper - lower)
                if upper > lower
                else np.zeros_like(sweep_values)
            )

            axis.plot(
                normalized_input,
                percentage_change,
                label=input_name,
                color=colors[input_index % len(colors)],
            )

        axis.axhline(0.0, linestyle="--")
        axis.set_title(f"{target}: one-factor-at-a-time sensitivity")
        axis.set_xlabel("Normalized input sweep (0 = observed min, 1 = max)")
        axis.set_ylabel("Change from median-design prediction (%)")
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

    data, metadata, model = load_artifacts(args.data, args.model_dir)

    inputs = metadata["input_columns"]
    targets = metadata["target_columns"]
    units = metadata.get("target_units", {})
    seed = int(metadata["seed"])

    test_data = make_test_split(data, seed)

    x_test = test_data[inputs].to_numpy(dtype=np.float32)
    y_test = test_data[targets].to_numpy(dtype=float)
    y_predicted = predict(model, x_test)

    metrics = calculate_metrics(
        y_test,
        y_predicted,
        targets,
        units,
    )

    figures_dir = args.model_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    distribution_units = {
        **units,
        "J1": "kg m²",
        "J2": "kg m²",
        "J3": "kg m²",
        "k_theta1": "N m/rad",
        "k_theta2": "N m/rad",
        "k_theta3": "N m/rad",
    }

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

    print("\nEXTENDED HELD-OUT TEST METRICS")
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
    print(f"Figures: {figures_dir}")
    print("\nAnalysis complete. The model was loaded, not retrained.")


if __name__ == "__main__":
    main()